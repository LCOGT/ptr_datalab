import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.wcs import WCS
from dateutil.parser import ParserError, parse as parse_date

from datalab.datalab_session.utils.comparison_stars import (
    ComparisonMeasurement,
    ComparisonStar,
    measure_candidate_on_frame,
    select_comparison_stars,
)
from datalab.datalab_session.utils.centroiding import centroid
from datalab.datalab_session.utils.fits_metadata import header_float, world_to_pixel
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag
from datalab.datalab_session.utils.geometry import (
    angular_distance_arcsec,
    distance_pixels,
    minimum_angular_neighbor_distance_arcsec,
)
from datalab.datalab_session.utils.photometry_diagnostics import (
    candidate_overlay_jpeg_base64,
    comparison_star_validation_diagnostics,
)
from datalab.datalab_session.utils.photometry import measure_aperture

log = logging.getLogger()
log.setLevel(logging.INFO)

SOURCE_CATALOG_RA_KEY = "ra"
SOURCE_CATALOG_DEC_KEY = "dec"
SOURCE_CATALOG_MAG_KEY = "mag"
SOURCE_CATALOG_FLUX_KEY = "flux"
EDGE_MARGIN_PX = 2.0
TARGET_PROXIMITY_FACTOR = 2.0
DEFAULT_CROSSMATCH_ARCSEC = 1.0
DEFAULT_GAIN = 1.0
DEFAULT_READ_NOISE = 0.0
DEFAULT_APERTURE_RADIUS_PX = 7.64
DEFAULT_ANNULUS_INNER_RADIUS_PX = 12.73
DEFAULT_ANNULUS_OUTER_RADIUS_PX = 19.10
DEFAULT_MIN_COMPARISONS = 5
DEFAULT_MAX_COMPARISONS = 10


class LightCurveError(ValueError):
    pass


def _pixel_to_world(header: Mapping[str, Any], x: float, y: float) -> tuple[float, float]:
    ra, dec = WCS(dict(header)).pixel_to_world_values(float(x), float(y))
    return float(ra), float(dec)


@dataclass(frozen=True)
class FrameContext:
    """
        Validates FITS frame data needed by the aperture photometry pipeline.
    """
    fits_path: str
    date_obs: datetime
    header: Mapping[str, Any]
    image: np.ndarray
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
    diagnostic_images_by_fits_basename: dict[str, str]


def generate_light_curve(
    input_handlers: list[Any],
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    comparison_strategy: str = "auto",
    min_comparisons: int = 5,
    max_comparisons: int = 10,
) -> LightCurveResult:
    """
        Generates a calibrated target light curve from input FITS files, using comparison stars from the source catalog.

        Validates frames, measures the target, builds a comp star catalog,
        selects a comparison ensemble, and produces calibrated light curve rows with diagnostics for the frontend.
    """
    log.info(
        "Aperture Photometry pipeline starting: "
        f"fits_count={len(input_handlers)}, target_ra={target_ra_deg:.8f}, target_dec={target_dec_deg:.8f}, "
        f"aperture_radius_px={aperture_radius_px:.3f}, "
        f"annulus_inner_radius_px={annulus_inner_radius_px:.3f}, "
        f"annulus_outer_radius_px={annulus_outer_radius_px:.3f}, "
        f"comparison_strategy={comparison_strategy}, "
        f"min_comparisons={min_comparisons}, max_comparisons={max_comparisons}"
    )
    _validate_inputs(
        input_handlers=input_handlers,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        comparison_strategy=comparison_strategy,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
    )

    diagnostics: list[str] = []
    frames = _validated_frame_contexts(input_handlers)

    diagnostics_by_fits_basename: dict[str, list[str]] = {
        os.path.basename(frame.fits_path): []
        for frame in frames
    }
    target_measurements = {
        frame.fits_path: _measure_target(
            frame=frame,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
            aperture_radius_px=aperture_radius_px,
            annulus_inner_radius_px=annulus_inner_radius_px,
            annulus_outer_radius_px=annulus_outer_radius_px,
        )
        for frame in frames
    }
    for frame in frames:
        target = target_measurements[frame.fits_path]
        log.info(
            "Aperture Photometry target measurement: "
            f"frame={frame.fits_path}, centroid=({target.x:.3f}, {target.y:.3f}), "
            f"net_counts={target.net_source_counts:.6f}, uncertainty={target.source_uncertainty:.6f}, "
            f"background={target.mean_background_per_pixel:.6f}, peak={target.peak_pixel_value:.6f}"
        )

    catalog = _build_field_star_catalog(
        frames=frames,
        target_ra_deg=target_ra_deg,
        target_dec_deg=target_dec_deg,
        aperture_radius_px=aperture_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
    )
    log.info(
        "Aperture Photometry comparison catalog built: "
        f"valid_candidates={len(catalog)}"
    )
    target_catalog_flux = _target_catalog_flux_proxy(
        frames=frames,
        target_measurements=target_measurements,
        target_ra_deg=target_ra_deg,
        target_dec_deg=target_dec_deg,
    )
    target_catalog_flux_text = f"{target_catalog_flux:.6f}" if target_catalog_flux is not None else "nan"
    log.info(
        "Aperture Photometry target catalog flux proxy: "
        f"{target_catalog_flux_text}"
    )
    selection = select_comparison_stars(
        frames=frames,
        catalog=catalog,
        target_catalog_flux=target_catalog_flux,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
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
    diagnostic_images_by_fits_basename: dict[str, str] = {}
    for frame in frames:
        target = target_measurements[frame.fits_path]
        comparison_measurements = tuple(
            measure_candidate_on_frame(
                frame=frame,
                candidate=star,
                aperture_radius_px=aperture_radius_px,
                annulus_inner_radius_px=annulus_inner_radius_px,
                annulus_outer_radius_px=annulus_outer_radius_px,
                error_class=LightCurveError,
            )
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
        target_rel_flux_sigma = abs(target_rel_flux) * math.sqrt(
            target_variance / (target.net_source_counts * target.net_source_counts)
            + ensemble_variance / (ensemble_flux * ensemble_flux)
        )
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
        diagnostic_images_by_fits_basename[os.path.basename(frame.fits_path)] = candidate_overlay_jpeg_base64(
            frame=frame,
            stars=selection.selected_stars,
            measurements=comparison_measurements,
            target_measurement=target,
            aperture_radius_px=aperture_radius_px,
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
        diagnostic_images_by_fits_basename=diagnostic_images_by_fits_basename,
    )


def _validate_inputs(
    *,
    input_handlers: Sequence[Any],
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    comparison_strategy: str,
    min_comparisons: int,
    max_comparisons: int,
) -> None:
    if not input_handlers:
        raise LightCurveError("input_handlers must be a non-empty list.")
    if aperture_radius_px <= 0:
        raise LightCurveError("aperture_radius_px must be > 0.")
    if annulus_inner_radius_px <= aperture_radius_px:
        raise LightCurveError("annulus_inner_radius_px must be greater than aperture_radius_px.")
    if annulus_outer_radius_px <= annulus_inner_radius_px:
        raise LightCurveError("annulus_outer_radius_px must be greater than annulus_inner_radius_px.")
    if min_comparisons <= 0 or max_comparisons <= 0 or min_comparisons > max_comparisons:
        raise LightCurveError("min_comparisons and max_comparisons must be positive and min_comparisons <= max_comparisons.")
    if comparison_strategy not in {"auto", "variability_first"}:
        raise LightCurveError("comparison_strategy must be 'auto' or 'variability_first'.")


def _validated_frame_contexts(input_handlers: Sequence[Any]) -> list[FrameContext]:
    frames: list[FrameContext] = []
    for input_handler in input_handlers:
        fits_path = input_handler.fits_file
        log.info(f"Aperture Photometry validating FITS frame: {fits_path}")
        try:
            image = np.asarray(input_handler.sci_data, dtype=float)
            if image.ndim != 2:
                raise LightCurveError(f"Primary image for {fits_path} is not a 2D array.")

            header = dict(input_handler.sci_hdu.header)
            date_obs_value = header.get("DATE-OBS")
            if not isinstance(date_obs_value, str) or not date_obs_value.strip():
                raise LightCurveError(f"Missing DATE-OBS in {fits_path}.")
            try:
                date_obs = parse_date(date_obs_value)
            except (ParserError, TypeError, ValueError, OverflowError) as exc:
                raise LightCurveError(f"Malformed DATE-OBS in {fits_path}: {date_obs_value!r}") from exc
            if date_obs.tzinfo is None:
                date_obs = date_obs.replace(tzinfo=timezone.utc)
            second_hdu_rows = tuple(_cat_rows(input_handler.get_hdu("CAT").data))
            if not second_hdu_rows:
                raise LightCurveError(f"Second HDU is missing or empty for {fits_path}.")

            _validate_wcs(header, fits_path, image.shape)
            _validate_second_hdu(second_hdu_rows, fits_path)
            log.info(
                "Aperture Photometry frame validated: "
                f"frame={fits_path}, date_obs={date_obs.isoformat()}, "
                f"image_shape={image.shape}, catalog_rows={len(second_hdu_rows)}"
            )
            frames.append(
                FrameContext(
                    fits_path=fits_path,
                    date_obs=date_obs,
                    header=header,
                    image=image,
                    second_hdu_rows=second_hdu_rows,
                    width=int(image.shape[1]),
                    height=int(image.shape[0]),
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


def _cat_rows(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    names = list(data.names or [])
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
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
) -> TargetMeasurement:
    """
        Converts the target RA and Dec to pixel coordinates, centroids the source, and measures aperture photometry.

        Returns the target measurement for a single frame.
    """
    try:
        initial_x, initial_y = world_to_pixel(frame.header, target_ra_deg, target_dec_deg)
    except Exception as exc:
        raise LightCurveError(f"Target WCS localization failed for {frame.fits_path}.") from exc
    log.info(
        "Aperture Photometry target WCS localization: "
        f"frame={frame.fits_path}, initial_pixel=({initial_x:.3f}, {initial_y:.3f})"
    )

    centroid_result = centroid(
        image=frame.image,
        x_click=initial_x,
        y_click=initial_y,
        radius=aperture_radius_px,
        r_back1=annulus_inner_radius_px,
        r_back2=annulus_outer_radius_px,
    )
    if not centroid_result.success:
        raise LightCurveError(f"Target centroiding failed for {frame.fits_path}.")
    log.info(
        "Aperture Photometry target centroid: "
        f"frame={frame.fits_path}, centroid=({centroid_result.x:.3f}, {centroid_result.y:.3f})"
    )

    photometry = measure_aperture(
        image=frame.image,
        x_center=centroid_result.x,
        y_center=centroid_result.y,
        aperture_radius_px=aperture_radius_px,
        background_model=centroid_result.background_model,
        gain=header_float(frame.header, ("GAIN", "EGAIN"), DEFAULT_GAIN),
        read_noise=header_float(frame.header, ("RDNOISE", "READNOIS", "READNOISE"), DEFAULT_READ_NOISE),
        dark=0.0,
        error_class=LightCurveError,
    )
    return TargetMeasurement(
        x=centroid_result.x,
        y=centroid_result.y,
        net_source_counts=photometry["net_source_counts"],
        source_uncertainty=photometry["source_uncertainty"],
        mean_background_per_pixel=photometry["mean_background_per_pixel"],
        peak_pixel_value=photometry["peak_pixel_value"],
        effective_source_pixels=photometry["effective_source_pixels"],
        effective_background_pixels=photometry["effective_background_pixels"],
    )


def _nearest_source_catalog_row(
    *,
    frame: FrameContext,
    ra_deg: float,
    dec_deg: float,
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for raw_row in frame.second_hdu_rows:
        try:
            rows.append(_extract_candidate_row(raw_row, frame.fits_path))
        except LightCurveError:
            continue
    if not rows:
        return None
    return min(
        rows,
            key=lambda row: angular_distance_arcsec(
            ra_deg,
            dec_deg,
            row["ra_deg"],
            row["dec_deg"],
        ),
    )


def _target_catalog_flux_proxy(
    *,
    frames: Sequence[FrameContext],
    target_measurements: Mapping[str, TargetMeasurement],
    target_ra_deg: float,
    target_dec_deg: float,
) -> float | None:
    """
        Estimates the target source catalog flux from nearby catalog matches across frames.

        Returns the median matched catalog flux, or None if no valid match is found.
    """
    target_catalog_fluxes: list[float] = []
    for frame in frames:
        target = target_measurements.get(frame.fits_path)
        if target is None:
            continue
        try:
            centroid_ra_deg, centroid_dec_deg = _pixel_to_world(frame.header, target.x, target.y)
        except Exception:
            centroid_ra_deg = target_ra_deg
            centroid_dec_deg = target_dec_deg
        target_row = _nearest_source_catalog_row(
            frame=frame,
            ra_deg=centroid_ra_deg,
            dec_deg=centroid_dec_deg,
        )
        if target_row is None:
            continue
        distance_arcsec = angular_distance_arcsec(
            centroid_ra_deg,
            centroid_dec_deg,
            target_row["ra_deg"],
            target_row["dec_deg"],
        )
        if distance_arcsec > DEFAULT_CROSSMATCH_ARCSEC:
            continue
        flux = _optional_float(target_row.get("flux"))
        if math.isfinite(flux) and flux > 0.0:
            target_catalog_fluxes.append(flux)

    if not target_catalog_fluxes:
        return None
    return float(np.median(np.asarray(target_catalog_fluxes, dtype=float)))


def _optional_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _build_field_star_catalog(
    *,
    frames: Sequence[FrameContext],
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius_px: float,
    annulus_outer_radius_px: float,
) -> list[dict[str, Any]]:
    """
        Builds comp star candidates from the source catalogs across valid frames.

        Reutrns candidates that are present in all frames and are not too close to the target or the edge of the image.
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

        target_x, target_y = target_pixels[frame.fits_path]
        target_limit_px = max(TARGET_PROXIMITY_FACTOR * aperture_radius_px, annulus_outer_radius_px)
        too_close_to_target_mask = np.hypot(x_values - target_x, y_values - target_y) <= target_limit_px
        too_close_to_edge_mask = (
            (x_values - annulus_outer_radius_px < EDGE_MARGIN_PX)
            | (y_values - annulus_outer_radius_px < EDGE_MARGIN_PX)
            | (x_values + annulus_outer_radius_px >= frame.width - EDGE_MARGIN_PX)
            | (y_values + annulus_outer_radius_px >= frame.height - EDGE_MARGIN_PX)
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
    for idx, cluster in enumerate(
        sorted(clusters, key=lambda item: (round(item["ra_deg"], 8), round(item["dec_deg"], 8)))
    ):
        if len(cluster["frame_paths"]) < len(frames):
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
                "isolation_px": isolation,
                "target_separation_px": target_sep,
            }
        )
    log.info(
        "Aperture Photometry comparison catalog summary: "
        f"clusters={len(clusters)}, rejected_insufficient_coverage={rejected_for_coverage}, "
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
    flux = _optional_float(row[SOURCE_CATALOG_FLUX_KEY])
    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg) or not math.isfinite(mag) or not math.isfinite(flux):
        raise LightCurveError(f"Second HDU row contains malformed RA/Dec/magnitude/flux values in {fits_path}.")
    return {
        "source_label": str(row.get("id", row.get("name", f"{ra_deg:.6f},{dec_deg:.6f}"))),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "mag": mag,
        "flux": flux,
    }
