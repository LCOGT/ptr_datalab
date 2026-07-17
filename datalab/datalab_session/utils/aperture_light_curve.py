import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from dateutil.parser import ParserError, parse as parse_date

from datalab.datalab_session.utils.comparison_stars import (
    ComparisonMeasurement,
    ComparisonStar,
    candidate_stars_from_catalog,
    measure_candidate_on_frame,
    select_comparison_stars,
)
from datalab.datalab_session.utils.centroiding import calculate_background_model, centroid
from datalab.datalab_session.utils.fits_metadata import (
    FrameGeometry,
    arcsec_to_pixels,
    frame_gain,
    frame_geometry,
    frame_read_noise,
    optional_float,
    world_to_pixel,
)
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag
from datalab.datalab_session.utils.geometry import (
    angular_distance_arcsec,
    distance_pixels,
    minimum_angular_neighbor_distance_arcsec,
)
from datalab.datalab_session.utils.photometry_diagnostics import (
    candidate_overlay_jpeg_bytes,
    comparison_star_validation_diagnostics,
)
from datalab.datalab_session.utils.photometry import measure_aperture

log = logging.getLogger()
log.setLevel(logging.INFO)

SOURCE_CATALOG_RA_KEY = "ra"
SOURCE_CATALOG_DEC_KEY = "dec"
SOURCE_CATALOG_MAG_KEY = "mag"
SOURCE_CATALOG_FLUX_KEY = "flux"
# The only CAT columns the pipeline reads. CAT tables carry many more columns, and whole rows kept
# per frame for the full run are a measurable share of operation memory on dense fields.
SOURCE_CATALOG_COLUMNS = (
    "id",
    "name",
    SOURCE_CATALOG_RA_KEY,
    SOURCE_CATALOG_DEC_KEY,
    SOURCE_CATALOG_MAG_KEY,
    SOURCE_CATALOG_FLUX_KEY,
)
EDGE_MARGIN_PX = 2.0
TARGET_PROXIMITY_FACTOR = 2.0
# A target recenter is accepted only if the centroid moves less than this many pixels from the
# WCS-predicted position. Larger shifts (or a failed centroid) mean the centroid was pulled onto
# a neighbour or host-galaxy structure.
TARGET_RECENTER_MAX_SHIFT_PX = 6.0
DEFAULT_CROSSMATCH_ARCSEC = 1.0
DEFAULT_APERTURE_RADIUS = 7.64
DEFAULT_ANNULUS_INNER_RADIUS = 12.73
DEFAULT_ANNULUS_OUTER_RADIUS = 19.10
DEFAULT_MIN_COMPARISONS = 5
DEFAULT_MAX_COMPARISONS = 10
# A comparison candidate is established if its cross-matched cluster is detected in at least this
# fraction of frames. Catalog detection near the limiting magnitude is noisy, and heterogeneous
# multi-telescope sets (different sites/FOV/depth) rarely catalog the same star in *every* frame,
# so requiring presence in all frames discards good stars and collapses the candidate pool. Selected
# stars are still measured via WCS on all frames, so the ensemble stays consistent frame-to-frame.
COMPARISON_FRAME_COVERAGE_FRACTION = 0.8


class LightCurveError(ValueError):
    pass


@dataclass(frozen=True)
class FrameContext:
    """
        Validated FITS frame metadata needed by the aperture photometry pipeline.

        Deliberately holds no pixel data: full-resolution images are streamed through the pixel
        pass one frame at a time (see _measure_frame_pixels), so peak memory stays flat no matter
        how many frames are submitted.
    """
    fits_path: str
    date_obs: datetime
    header: Mapping[str, Any]
    second_hdu_rows: tuple[Mapping[str, Any], ...]
    width: int
    height: int


@dataclass(frozen=True)
class TargetMeasurement:
    """
        Aperture measurement for the target source in a single frame.
    """
    x: float
    y: float
    net_source_counts: float
    source_uncertainty: float
    mean_background_per_pixel: float
    peak_pixel_value: float
    effective_source_pixels: float
    effective_background_pixels: float


@dataclass(frozen=True)
class FrameResult:
    """
        Target and comp star measurements for a single FITS frame.
    """
    fits_path: str
    date_obs: datetime
    target_measurement: TargetMeasurement
    comparison_measurements: tuple[ComparisonMeasurement, ...]


@dataclass(frozen=True)
class LightCurveRow:
    """
        A single row of the calibrated light curve for the target source.
    """
    fits_path: str
    date_obs: datetime
    target_centroid_x: float
    target_centroid_y: float
    target_net_source_counts: float
    target_source_uncertainty: float
    comparison_ensemble_total_counts: float
    comparison_ensemble_uncertainty: float
    target_differential_flux: float
    target_differential_flux_uncertainty: float
    target_calibrated_apparent_magnitude: float
    target_calibrated_apparent_magnitude_uncertainty: float


@dataclass(frozen=True)
class LightCurveResult:
    """
        Complete aperture photometry result returned by the generate_light_curve function, including light curve rows, selected comparison stars, and diagnostics.
    """
    frames: list[FrameResult]
    selected_comparison_stars: list[ComparisonStar]
    light_curve_rows: list[LightCurveRow]
    diagnostics: list[str]
    diagnostics_by_fits_basename: dict[str, list[str]]
    diagnostic_image_jpegs_by_fits_basename: dict[str, bytes]


def generate_light_curve(
    fits_paths: list[str],
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius: float,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    min_comparisons: int = 5,
    max_comparisons: int = 10,
) -> LightCurveResult:
    """
        Generates a calibrated target light curve from local input FITS files, using comparison
        stars from the source catalog.

        Validates frame metadata and builds the comparison-star candidate catalog from headers and
        CAT tables alone, then streams pixel data one frame at a time to measure the target and
        every candidate, selects a comparison ensemble, and produces calibrated light curve rows
        with diagnostics for the frontend. At most one frame's full-resolution pixels are in
        memory at any point, so memory does not grow with the number of input frames.
    """
    log.info(
        "Aperture Photometry pipeline starting: "
        f"fits_count={len(fits_paths)}, target_ra={target_ra_deg:.8f}, target_dec={target_dec_deg:.8f}, "
        f"aperture_radius={aperture_radius:.3f}, "
        f"annulus_inner_radius={annulus_inner_radius:.3f}, "
        f"annulus_outer_radius={annulus_outer_radius:.3f}, "
        f"min_comparisons={min_comparisons}, max_comparisons={max_comparisons}"
    )
    _validate_inputs(
        fits_paths=fits_paths,
        aperture_radius=aperture_radius,
        annulus_inner_radius=annulus_inner_radius,
        annulus_outer_radius=annulus_outer_radius,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
    )

    diagnostics: list[str] = []
    frames = _validated_frame_contexts(fits_paths)

    diagnostics_by_fits_basename: dict[str, list[str]] = {
        os.path.basename(frame.fits_path): []
        for frame in frames
    }

    catalog = _build_field_star_catalog(
        frames=frames,
        target_ra_deg=target_ra_deg,
        target_dec_deg=target_dec_deg,
        aperture_radius=aperture_radius,
        annulus_outer_radius=annulus_outer_radius,
    )
    log.info(
        "Aperture Photometry comparison catalog built: "
        f"valid_candidates={len(catalog)}"
    )
    candidate_stars = candidate_stars_from_catalog(catalog)

    target_measurements: dict[str, TargetMeasurement] = {}
    measurements_by_candidate: dict[str, dict[str, ComparisonMeasurement]] = {
        candidate.candidate_id: {} for candidate in candidate_stars
    }
    failed_candidate_ids: set[str] = set()
    for frame in frames:
        target, frame_measurements, newly_failed = _measure_frame_pixels(
            frame=frame,
            candidate_stars=candidate_stars,
            skip_candidate_ids=failed_candidate_ids,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
            aperture_radius=aperture_radius,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
        )
        target_measurements[frame.fits_path] = target
        failed_candidate_ids |= newly_failed
        for candidate_id in newly_failed:
            measurements_by_candidate.pop(candidate_id, None)
        for candidate_id, measurement in frame_measurements.items():
            measurements_by_candidate[candidate_id][frame.fits_path] = measurement
        log.info(
            "Aperture Photometry target measurement: "
            f"frame={frame.fits_path}, centroid=({target.x:.3f}, {target.y:.3f}), "
            f"net_counts={target.net_source_counts:.6f}, uncertainty={target.source_uncertainty:.6f}, "
            f"background={target.mean_background_per_pixel:.6f}, peak={target.peak_pixel_value:.6f}"
        )

    target_mag_proxy = _target_magnitude_proxy(target_measurements.values())
    log.info(f"Aperture Photometry target magnitude proxy: {target_mag_proxy:.6f}")
    selection = select_comparison_stars(
        frames=frames,
        candidates=candidate_stars,
        measurements_by_candidate=measurements_by_candidate,
        target_mag_proxy=target_mag_proxy,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
        error_class=LightCurveError,
    )
    log.info(
        "Aperture Photometry comparison stars selected: "
        f"selected_count={len(selection.selected_stars)}, "
        f"candidate_ids={[star.candidate_id for star in selection.selected_stars]}, "
        f"selection_diagnostics={len(selection.diagnostics)}"
    )

    reference_magnitudes = np.asarray(
        [star.reference_magnitude for star in selection.selected_stars],
        dtype=float,
    )
    ensemble_reference_flux = float(np.sum(10 ** (-0.4 * reference_magnitudes)))
    if ensemble_reference_flux <= 0.0:
        raise LightCurveError("Comparison-star magnitude calibration produced a non-positive ensemble reference flux.")
    ensemble_reference_mag = -2.5 * math.log10(ensemble_reference_flux)
    log.info(
        "Aperture Photometry calibration reference: "
        f"ensemble_reference_flux={ensemble_reference_flux:.12e}, "
        f"ensemble_reference_mag={ensemble_reference_mag:.6f}"
    )

    frame_results: list[FrameResult] = []
    light_curve_rows: list[LightCurveRow] = []
    diagnostic_image_jpegs_by_fits_basename: dict[str, bytes] = {}
    for frame in frames:
        target = target_measurements[frame.fits_path]
        # Reuse the per-frame measurements captured during selection rather than re-measuring the
        # same apertures on the same frames.
        comparison_measurements = tuple(
            selection.measurements_by_candidate[star.candidate_id][frame.fits_path]
            for star in selection.selected_stars
        )
        comparison_counts = np.asarray(
            [measurement.net_source_counts for measurement in comparison_measurements],
            dtype=float,
        )
        comparison_uncertainties = np.asarray(
            [measurement.source_uncertainty for measurement in comparison_measurements],
            dtype=float,
        )
        ensemble_flux = float(np.sum(comparison_counts))
        ensemble_variance = float(np.sum(np.square(comparison_uncertainties)))
        if not math.isfinite(ensemble_flux) or ensemble_flux <= 0.0:
            raise LightCurveError(f"Ensemble comparison flux is invalid for frame {frame.fits_path}.")

        target_variance = target.source_uncertainty * target.source_uncertainty
        target_rel_flux = target.net_source_counts / ensemble_flux
        if target.net_source_counts > 0.0:
            target_rel_flux_sigma = abs(target_rel_flux) * math.sqrt(
                target_variance / (target.net_source_counts * target.net_source_counts)
                + ensemble_variance / (ensemble_flux * ensemble_flux)
            )
        else:
            # Non-positive target counts (e.g. WCS-fallback measuring on blank sky) make the
            # relative-flux error undefined and dividing by net_source_counts**2 would raise
            # ZeroDivisionError at exactly 0. The row's magnitude is NaN below in this case anyway.
            target_rel_flux_sigma = math.nan
        calibrated_flux = target_rel_flux * ensemble_reference_flux
        calibration_slope = -2.5
        frame_zero_point = 2.5 * math.log10(ensemble_flux) + ensemble_reference_mag
        if calibrated_flux > 0.0:
            calibrated_mag = calibration_slope * math.log10(target.net_source_counts) + frame_zero_point
            _, calibrated_mag_sigma = flux_to_mag(target_rel_flux, target_rel_flux_sigma)
        else:
            calibrated_mag = math.nan
            calibrated_mag_sigma = math.nan
        log.info(
            "Aperture Photometry frame calibration: "
            f"frame={frame.fits_path}, target_counts={target.net_source_counts:.6f}, "
            f"comparison_ensemble_counts={ensemble_flux:.6f}, "
            f"target_rel_flux={target_rel_flux:.12e}, target_rel_flux_sigma={target_rel_flux_sigma:.12e}, "
            f"frame_zero_point={frame_zero_point:.6f}, calibrated_flux={calibrated_flux:.12e}, "
            f"calibrated_mag={calibrated_mag:.6f}, "
            f"calibrated_mag_sigma={calibrated_mag_sigma:.6f}"
        )
        if not math.isfinite(calibrated_mag) or not math.isfinite(calibrated_mag_sigma):
            log.warning(
                "Aperture Photometry non-finite light-curve row: "
                f"frame={frame.fits_path}, calibrated_mag={calibrated_mag}, "
                f"calibrated_mag_sigma={calibrated_mag_sigma}. "
                "This row is present in backend output as null after JSON serialization and the frontend plot skips it."
            )
        frame_diagnostics = comparison_star_validation_diagnostics(
            frame=frame,
            stars=selection.selected_stars,
            measurements=comparison_measurements,
            frame_zero_point=frame_zero_point,
        )
        diagnostics.extend(frame_diagnostics)
        diagnostics_by_fits_basename[os.path.basename(frame.fits_path)].extend(frame_diagnostics)
        diagnostic_image_jpegs_by_fits_basename[os.path.basename(frame.fits_path)] = _render_frame_overlay(
            frame=frame,
            stars=selection.selected_stars,
            measurements=comparison_measurements,
            target_measurement=target,
            aperture_radius=aperture_radius,
        )

        frame_results.append(
            FrameResult(
                fits_path=frame.fits_path,
                date_obs=frame.date_obs,
                target_measurement=target,
                comparison_measurements=comparison_measurements,
            )
        )
        light_curve_rows.append(
            LightCurveRow(
                fits_path=frame.fits_path,
                date_obs=frame.date_obs,
                target_centroid_x=target.x,
                target_centroid_y=target.y,
                target_net_source_counts=target.net_source_counts,
                target_source_uncertainty=target.source_uncertainty,
                comparison_ensemble_total_counts=ensemble_flux,
                comparison_ensemble_uncertainty=math.sqrt(ensemble_variance),
                target_differential_flux=target_rel_flux,
                target_differential_flux_uncertainty=target_rel_flux_sigma,
                target_calibrated_apparent_magnitude=calibrated_mag,
                target_calibrated_apparent_magnitude_uncertainty=calibrated_mag_sigma,
            )
        )

    log.info(
        "Aperture Photometry pipeline completed: "
        f"frames={len(frame_results)}, light_curve_rows={len(light_curve_rows)}, "
        f"selected_comparison_stars={len(selection.selected_stars)}, diagnostics={len(diagnostics)}"
    )
    return LightCurveResult(
        frames=frame_results,
        selected_comparison_stars=list(selection.selected_stars),
        light_curve_rows=light_curve_rows,
        diagnostics=diagnostics,
        diagnostics_by_fits_basename=diagnostics_by_fits_basename,
        diagnostic_image_jpegs_by_fits_basename=diagnostic_image_jpegs_by_fits_basename,
    )


def _validate_inputs(
    *,
    fits_paths: Sequence[str],
    aperture_radius: float,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
    min_comparisons: int,
    max_comparisons: int,
) -> None:
    if not fits_paths:
        raise LightCurveError("fits_paths must be a non-empty list.")
    if aperture_radius <= 0:
        raise LightCurveError("aperture_radius must be > 0.")
    if annulus_inner_radius <= aperture_radius:
        raise LightCurveError("annulus_inner_radius must be greater than aperture_radius.")
    if annulus_outer_radius <= annulus_inner_radius:
        raise LightCurveError("annulus_outer_radius must be greater than annulus_inner_radius.")
    if min_comparisons <= 0 or max_comparisons <= 0 or min_comparisons > max_comparisons:
        raise LightCurveError("min_comparisons and max_comparisons must be positive and min_comparisons <= max_comparisons.")


def _validated_frame_contexts(fits_paths: Sequence[str]) -> list[FrameContext]:
    """
        Builds validated frame metadata for each input FITS path.

        Reads only the SCI header and the CAT table -- never SCI pixel data -- so validation memory
        and time stay flat regardless of frame count or sensor size. Frames that fail validation
        are ignored with a warning.
    """
    frames: list[FrameContext] = []
    for fits_path in fits_paths:
        log.info(f"Aperture Photometry validating FITS frame: {fits_path}")
        try:
            with fits.open(fits_path) as hdul:
                header = dict(hdul["SCI"].header)
                second_hdu_rows = tuple(_cat_rows(hdul["CAT"].data))

            if int(header.get("NAXIS", 0)) != 2:
                raise LightCurveError(f"Primary image for {fits_path} is not a 2D array.")
            width = int(header["NAXIS1"])
            height = int(header["NAXIS2"])

            date_obs_value = header.get("DATE-OBS")
            if not isinstance(date_obs_value, str) or not date_obs_value.strip():
                raise LightCurveError(f"Missing DATE-OBS in {fits_path}.")
            try:
                date_obs = parse_date(date_obs_value)
            except (ParserError, TypeError, ValueError, OverflowError) as exc:
                raise LightCurveError(f"Malformed DATE-OBS in {fits_path}: {date_obs_value!r}") from exc
            if date_obs.tzinfo is None:
                date_obs = date_obs.replace(tzinfo=timezone.utc)
            if not second_hdu_rows:
                raise LightCurveError(f"Second HDU is missing or empty for {fits_path}.")

            _validate_wcs(header, fits_path, (height, width))
            _validate_second_hdu(second_hdu_rows, fits_path)
            log.info(
                "Aperture Photometry frame validated: "
                f"frame={fits_path}, date_obs={date_obs.isoformat()}, "
                f"image_shape={(height, width)}, catalog_rows={len(second_hdu_rows)}"
            )
            frames.append(
                FrameContext(
                    fits_path=fits_path,
                    date_obs=date_obs,
                    header=header,
                    second_hdu_rows=second_hdu_rows,
                    width=width,
                    height=height,
                )
            )
        except LightCurveError as exc:
            log.warning(
                "Aperture Photometry ignoring input frame after validation error: "
                f"frame={fits_path}, error={exc}"
            )
        except Exception as exc:
            log.warning(
                "Aperture Photometry ignoring input frame after validation error: "
                f"frame={fits_path}, error={exc}"
            )

    if not frames:
        raise LightCurveError("Aperture photometry requires at least 1 valid input file.")

    frames = sorted(frames, key=lambda frame: frame.date_obs)
    log.info(
        "Aperture Photometry frames validated and sorted: "
        f"frame_count={len(frames)}, ordered_paths={[frame.fits_path for frame in frames]}"
    )
    return frames


def _load_frame_image(fits_path: str) -> np.ndarray:
    """
        Loads one frame's SCI pixel data as float32.

        float32 matches the archive's native SCI pixel type; asking for float64 here would double
        every frame's in-memory size (photometry sums already accumulate in double precision).
    """
    with fits.open(fits_path, memmap=False) as hdul:
        image = np.asarray(hdul["SCI"].data, dtype=np.float32)
    if image.ndim != 2:
        raise LightCurveError(f"Primary image for {fits_path} is not a 2D array.")
    return image


def _measure_frame_pixels(
    *,
    frame: FrameContext,
    candidate_stars: Sequence[ComparisonStar],
    skip_candidate_ids: set[str],
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius: float,
    annulus_inner_radius: float,
    annulus_outer_radius: float,
) -> tuple[TargetMeasurement, dict[str, ComparisonMeasurement], set[str]]:
    """
        Runs all pixel-dependent work for one frame: the target measurement and a measurement of
        every comparison candidate (minus skip_candidate_ids).

        The full-resolution image exists only inside this function, so it is released before the
        caller moves on to the next frame.

        Returns the target measurement, this frame's candidate measurements by candidate_id, and
        the ids of candidates that failed to measure on this frame.
    """
    image = _load_frame_image(frame.fits_path)
    # Build the frame's WCS and pixel-space aperture radii once, then reuse them for the target and
    # every candidate. These are frame constants, so recomputing them per candidate (as the old
    # arcsec_to_pixels/world_to_pixel calls did) just re-parsed the header WCS thousands of times.
    geometry = frame_geometry(frame.header, aperture_radius, annulus_inner_radius, annulus_outer_radius)
    target_measurement = _measure_target(
        frame=frame,
        image=image,
        geometry=geometry,
        target_ra_deg=target_ra_deg,
        target_dec_deg=target_dec_deg,
    )
    candidate_measurements: dict[str, ComparisonMeasurement] = {}
    failed_candidate_ids: set[str] = set()
    for candidate in candidate_stars:
        if candidate.candidate_id in skip_candidate_ids:
            continue
        try:
            candidate_measurements[candidate.candidate_id] = measure_candidate_on_frame(
                frame=frame,
                image=image,
                geometry=geometry,
                candidate=candidate,
                error_class=LightCurveError,
            )
        except LightCurveError:
            failed_candidate_ids.add(candidate.candidate_id)
    return target_measurement, candidate_measurements, failed_candidate_ids


def _render_frame_overlay(
    *,
    frame: FrameContext,
    stars: Sequence[ComparisonStar],
    measurements: Sequence[ComparisonMeasurement],
    target_measurement: TargetMeasurement,
    aperture_radius: float,
) -> bytes:
    """
        Reloads one frame's pixels and renders its diagnostic overlay, cropped at full resolution
        around the drawn circles before resampling.

        The full-resolution image exists only inside this function, so overlay rendering keeps
        peak memory flat no matter how many frames are submitted.
    """
    image = _load_frame_image(frame.fits_path)
    return candidate_overlay_jpeg_bytes(
        frame=frame,
        image=image,
        stars=stars,
        measurements=measurements,
        target_measurement=target_measurement,
        aperture_radius=aperture_radius,
    )


def _cat_rows(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    names = [name for name in (data.names or []) if name in SOURCE_CATALOG_COLUMNS]
    return [
        {
            name: data[name][index].item() if hasattr(data[name][index], "item") else data[name][index]
            for name in names
        }
        for index in range(len(data))
    ]


def _validate_wcs(header: Mapping[str, Any], fits_path: str, shape: tuple[int, int]) -> None:
    try:
        wcs = WCS(dict(header)).celestial
        if not wcs.has_celestial:
            raise ValueError("missing celestial WCS")

        center_x = shape[1] / 2.0
        center_y = shape[0] / 2.0
        skycoord = wcs.pixel_to_world(center_x, center_y)
        roundtrip_x, roundtrip_y = wcs.world_to_pixel(skycoord)
        if not all(math.isfinite(value) for value in (roundtrip_x, roundtrip_y)):
            raise ValueError("celestial WCS produced non-finite coordinates")
    except Exception as exc:  # pragma: no cover - error path covered by tests
        raise LightCurveError(f"Missing or unusable WCS in {fits_path}.") from exc


def _validate_second_hdu(rows: Sequence[Mapping[str, Any]], fits_path: str) -> None:
    row = rows[0]
    for key, label in (
        (SOURCE_CATALOG_RA_KEY, "RA"),
        (SOURCE_CATALOG_DEC_KEY, "Dec"),
        (SOURCE_CATALOG_MAG_KEY, "magnitude"),
        (SOURCE_CATALOG_FLUX_KEY, "flux"),
    ):
        if key not in row:
            raise LightCurveError(f"Second HDU in {fits_path} is missing required {label} column.")


def _measure_target(
    *,
    frame: FrameContext,
    image: np.ndarray,
    geometry: FrameGeometry,
    target_ra_deg: float,
    target_dec_deg: float,
) -> TargetMeasurement:
    """
        Converts the target RA and Dec to pixel coordinates, centroids the source, and measures
        aperture photometry. image is the frame's pixel data, passed separately from the metadata
        so the streaming pixel pass controls how long it stays in memory. geometry carries the
        frame's cached WCS and pixel-space aperture radii.

        The target is never allowed to drop a frame: if centroiding fails or the refinement drifts
        too far from the WCS position, it measures at the authoritative WCS position instead.

        Returns the target measurement for a single frame.
    """
    aperture_radius_px = geometry.aperture_radius_px
    annulus_inner_radius_px = geometry.annulus_inner_radius_px
    annulus_outer_radius_px = geometry.annulus_outer_radius_px

    try:
        initial_x, initial_y = geometry.world_to_pixel(target_ra_deg, target_dec_deg)
    except Exception as exc:
        raise LightCurveError(f"Target WCS localization failed for {frame.fits_path}.") from exc
    log.info(
        "Aperture Photometry target WCS localization: "
        f"frame={frame.fits_path}, initial_pixel=({initial_x:.3f}, {initial_y:.3f})"
    )

    centroid_result = centroid(
        image=image,
        x_click=initial_x,
        y_click=initial_y,
        radius=aperture_radius_px,
        r_back1=annulus_inner_radius_px,
        r_back2=annulus_outer_radius_px,
    )

    # A failed centroid, or a refinement that drifts more than TARGET_RECENTER_MAX_SHIFT_PX from the
    # WCS position, means it locked onto the host galaxy or a neighbour, so fall back to the WCS position.
    recenter_shift_px = math.hypot(centroid_result.x - initial_x, centroid_result.y - initial_y)
    if centroid_result.success and recenter_shift_px <= TARGET_RECENTER_MAX_SHIFT_PX:
        x_center, y_center = centroid_result.x, centroid_result.y
        background_model = centroid_result.background_model
    else:
        x_center, y_center = initial_x, initial_y
        # Re-estimate the background at the WCS position: a drifted annulus can straddle the host
        # galaxy and bias the sky level, which is exactly the pull we are rejecting.
        background_model = calculate_background_model(
            image,
            x_center,
            y_center,
            aperture_radius_px,
            annulus_inner_radius_px,
            annulus_outer_radius_px,
            remove_background_stars=True,
            use_plane_background=False,
        )
        reason = (
            "centroiding failed"
            if not centroid_result.success
            else f"centroid shift {recenter_shift_px:.2f}px exceeded {TARGET_RECENTER_MAX_SHIFT_PX:.2f}px limit"
        )
        log.warning(
            "Aperture Photometry target recenter skipped: "
            f"frame={frame.fits_path}, {reason}; measured at WCS position "
            f"({x_center:.3f}, {y_center:.3f})."
        )
    log.info(
        "Aperture Photometry target centroid: "
        f"frame={frame.fits_path}, position=({x_center:.3f}, {y_center:.3f})"
    )

    photometry = measure_aperture(
        image=image,
        x_center=x_center,
        y_center=y_center,
        aperture_radius_px=aperture_radius_px,
        background_model=background_model,
        gain=frame_gain(frame.header),
        read_noise=frame_read_noise(frame.header),
        dark=0.0,
        error_class=LightCurveError,
    )
    return TargetMeasurement(
        x=x_center,
        y=y_center,
        net_source_counts=photometry["net_source_counts"],
        source_uncertainty=photometry["source_uncertainty"],
        mean_background_per_pixel=photometry["mean_background_per_pixel"],
        peak_pixel_value=photometry["peak_pixel_value"],
        effective_source_pixels=photometry["effective_source_pixels"],
        effective_background_pixels=photometry["effective_background_pixels"],
    )


def _target_magnitude_proxy(measurements: Iterable[TargetMeasurement]) -> float:
    """
        The target's own measured instrumental magnitude, median over frames. Comparison stars are
        ranked against this (not the target's catalog magnitude) so both sides share a zero point.
    """
    instrumental_mags = [
        -2.5 * math.log10(measurement.net_source_counts)
        for measurement in measurements
        if measurement.net_source_counts > 0
    ]
    if not instrumental_mags:
        raise LightCurveError("Target photometry never produced positive source counts.")
    return float(np.median(np.asarray(instrumental_mags, dtype=float)))


def _build_field_star_catalog(
    *,
    frames: Sequence[FrameContext],
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius: float,
    annulus_outer_radius: float,
) -> list[dict[str, Any]]:
    """
        Builds comp star candidates from the source catalogs across valid frames.

        Returns candidates detected in at least COMPARISON_FRAME_COVERAGE_FRACTION of the frames that
        are not too close to the target or the edge of the image.
    """
    clusters: list[dict[str, Any]] = []
    target_pixels = {
        frame.fits_path: world_to_pixel(frame.header, target_ra_deg, target_dec_deg)
        for frame in frames
    }

    for frame in frames:
        rows: list[dict[str, Any]] = []
        for raw_row in frame.second_hdu_rows:
            try:
                rows.append(_extract_candidate_row(raw_row, frame.fits_path))
            except LightCurveError as exc:
                log.warning(f"rejected comparison candidate in {frame.fits_path}: {exc}")
        rejected_for_target = 0
        rejected_for_edge = 0
        if not rows:
            log.info(
                "Aperture Photometry comparison candidates processed: "
                f"frame={frame.fits_path}, extracted_rows=0, "
                "rejected_too_close_to_target=0, "
                f"rejected_too_close_to_edge=0, clusters_so_far={len(clusters)}"
            )
            continue

        ra_values = np.asarray([row["ra_deg"] for row in rows], dtype=float)
        dec_values = np.asarray([row["dec_deg"] for row in rows], dtype=float)
        x_values, y_values = WCS(dict(frame.header)).world_to_pixel_values(ra_values, dec_values)
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        for row, x, y in zip(rows, x_values, y_values):
            row["frame_path"] = frame.fits_path
            row["pixel_x"] = float(x)
            row["pixel_y"] = float(y)

        frame_aperture_radius_px = arcsec_to_pixels(frame.header, aperture_radius)
        frame_annulus_outer_radius_px = arcsec_to_pixels(frame.header, annulus_outer_radius)
        target_x, target_y = target_pixels[frame.fits_path]
        target_limit_px = max(TARGET_PROXIMITY_FACTOR * frame_aperture_radius_px, frame_annulus_outer_radius_px)
        too_close_to_target_mask = np.hypot(x_values - target_x, y_values - target_y) <= target_limit_px
        too_close_to_edge_mask = (
            (x_values - frame_annulus_outer_radius_px < EDGE_MARGIN_PX)
            | (y_values - frame_annulus_outer_radius_px < EDGE_MARGIN_PX)
            | (x_values + frame_annulus_outer_radius_px >= frame.width - EDGE_MARGIN_PX)
            | (y_values + frame_annulus_outer_radius_px >= frame.height - EDGE_MARGIN_PX)
        )

        rejected_for_target = int(np.count_nonzero(too_close_to_target_mask))
        rejected_for_edge = int(np.count_nonzero(~too_close_to_target_mask & too_close_to_edge_mask))

        for row, target_rejected, edge_rejected in zip(rows, too_close_to_target_mask, too_close_to_edge_mask):
            if target_rejected or edge_rejected:
                continue
            x = row["pixel_x"]
            y = row["pixel_y"]

            matched = False
            for cluster in clusters:
                if frame.fits_path in cluster["frame_paths"]:
                    continue
                if angular_distance_arcsec(row["ra_deg"], row["dec_deg"], cluster["ra_deg"], cluster["dec_deg"]) <= DEFAULT_CROSSMATCH_ARCSEC:
                    cluster["rows"].append(row)
                    cluster["frame_paths"].add(frame.fits_path)
                    cluster["mags"].append(row["mag"])
                    cluster["source_catalog_by_frame"][frame.fits_path] = {
                        "source_label": row["source_label"],
                        "flux": row["flux"],
                        "mag": row["mag"],
                    }
                    cluster["xys"].append((x, y))
                    matched = True
                    break
            if not matched:
                clusters.append(
                    {
                        "ra_deg": row["ra_deg"],
                        "dec_deg": row["dec_deg"],
                        "rows": [row],
                        "frame_paths": {frame.fits_path},
                        "mags": [row["mag"]],
                        "source_catalog_by_frame": {
                            frame.fits_path: {
                                "source_label": row["source_label"],
                                "flux": row["flux"],
                                "mag": row["mag"],
                            }
                        },
                        "xys": [(x, y)],
                    }
                )
        log.info(
            "Aperture Photometry comparison candidates processed: "
            f"frame={frame.fits_path}, extracted_rows={len(rows)}, "
            f"rejected_too_close_to_target={rejected_for_target}, "
            f"rejected_too_close_to_edge={rejected_for_edge}, clusters_so_far={len(clusters)}"
        )

    catalog: list[dict[str, Any]] = []
    rejected_for_coverage = 0
    required_coverage = max(1, math.ceil(COMPARISON_FRAME_COVERAGE_FRACTION * len(frames)))
    for idx, cluster in enumerate(
        sorted(clusters, key=lambda item: (round(item["ra_deg"], 8), round(item["dec_deg"], 8)))
    ):
        if len(cluster["frame_paths"]) < required_coverage:
            rejected_for_coverage += 1
            continue
        isolation = minimum_angular_neighbor_distance_arcsec(cluster, clusters)
        target_sep = min(
            distance_pixels(row["pixel_x"], row["pixel_y"], *target_pixels[row["frame_path"]])
            for row in cluster["rows"]
        )
        catalog.append(
            {
                "candidate_id": f"cand-{idx + 1:03d}",
                "ra_deg": float(np.mean([row["ra_deg"] for row in cluster["rows"]])),
                "dec_deg": float(np.mean([row["dec_deg"] for row in cluster["rows"]])),
                "second_hdu_magnitude": float(np.median(np.asarray(cluster["mags"], dtype=float))),
                "source_catalog_by_frame": dict(cluster["source_catalog_by_frame"]),
                "frame_coverage": len(cluster["frame_paths"]),
                "isolation_arcsec": isolation,
                "target_separation_px": target_sep,
            }
        )
    log.info(
        "Aperture Photometry comparison catalog summary: "
        f"clusters={len(clusters)}, required_coverage={required_coverage}/{len(frames)} frames, "
        f"rejected_insufficient_coverage={rejected_for_coverage}, "
        f"valid_catalog_candidates={len(catalog)}"
    )
    return catalog


def _extract_candidate_row(row: Mapping[str, Any], fits_path: str) -> dict[str, Any]:
    """
        Extracts and validates RA, Dec, magnitude, and flux from a source catalog row.

        Returns a normalized candidate row dictionary.
    """
    required_keys = (
        SOURCE_CATALOG_RA_KEY,
        SOURCE_CATALOG_DEC_KEY,
        SOURCE_CATALOG_MAG_KEY,
        SOURCE_CATALOG_FLUX_KEY,
    )
    if any(key not in row for key in required_keys):
        raise LightCurveError(f"Second HDU rows cannot support RA/Dec matching in {fits_path}.")
    ra_deg = float(row[SOURCE_CATALOG_RA_KEY])
    dec_deg = float(row[SOURCE_CATALOG_DEC_KEY])
    mag = float(row[SOURCE_CATALOG_MAG_KEY])
    flux = optional_float(row[SOURCE_CATALOG_FLUX_KEY])
    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg) or not math.isfinite(mag) or not math.isfinite(flux):
        raise LightCurveError(f"Second HDU row contains malformed RA/Dec/magnitude/flux values in {fits_path}.")
    return {
        "source_label": str(row.get("id", row.get("name", f"{ra_deg:.6f},{dec_deg:.6f}"))),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "mag": mag,
        "flux": flux,
    }
