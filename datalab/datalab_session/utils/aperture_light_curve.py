import logging
import math
import os
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from dateutil.parser import ParserError, parse as parse_date

from datalab.datalab_session.utils.comparison_calibration import (
    CalibrationInputs,
    ComparisonStrategy,
    SharedEnsemble,
    calibrate,
)
from datalab.datalab_session.utils.comparison_stars import (
    ComparisonMeasurement,
    ComparisonStar,
    candidate_stars_from_catalog,
    measure_candidate_on_frame,
)
from datalab.datalab_session.utils.centroiding import calculate_background_model, centroid
from datalab.datalab_session.utils.fits_metadata import (
    FrameGeometry,
    arcsec_to_pixels,
    frame_geometry,
    optional_float,
    world_to_pixel,
)
from datalab.datalab_session.utils.light_curve_errors import LightCurveError
from datalab.datalab_session.utils.target_location import TargetLocator
from datalab.datalab_session.utils.geometry import (
    angular_distance_arcsec,
    distance_pixels,
    minimum_neighbor_distances_arcsec,
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
# Every candidate is measured on every frame it lands on, so cost runs as candidates x frames.
# Bound that product rather than the pool: long series get a smaller pool, short ones are untouched.
# Sized to clear an ordinary deep field, since trimming changes which stars are on offer.
MAX_CANDIDATE_MEASUREMENTS = 200_000
MIN_COMPARISON_CANDIDATES = 50
class Phase(Enum):
    """
        The phases of an aperture photometry run, in execution order.

        The single declaration of this vocabulary, shared with the operations layer, which maps each
        phase to a progress band. Order is part of the contract: a phase fills the band running from
        the previous phase's end to its own. DOWNLOADING and SAVE bracket the pipeline and are
        reported by the operation; the pipeline itself reports the five in between.

        Deliberately Enum rather than StrEnum: CI runs Python 3.10, where StrEnum does not exist.
    """
    DOWNLOADING = "downloading"
    VALIDATE = "validate"
    CATALOG = "catalog"
    MEASURE = "measure"
    SELECT = "select"
    RENDER = "render"
    SAVE = "save"


# Receives (phase, fraction), fraction being the completed share of that phase, in [0, 1].
ProgressCallback = Callable[[Phase, float], None]



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

    @classmethod
    def from_fits(cls, fits_path: str) -> "FrameContext":
        """
            Reads and validates one frame's metadata.

            Reads only the SCI header and the CAT table, never SCI pixel data, so validation memory
            and time stay flat regardless of frame count or sensor size. Raises LightCurveError if
            the frame cannot be used.
        """
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
        return cls(
            fits_path=fits_path,
            date_obs=date_obs,
            header=header,
            second_hdu_rows=second_hdu_rows,
            width=width,
            height=height,
        )


@dataclass
class CandidateCluster:
    """
        One field star, as detected across frames: the cross-matched group of catalog rows that all
        sit at the same sky position.

        Mutable by design, unlike the frozen records elsewhere: cross-matching grows a cluster row by
        row as it walks the frames. `ra_deg`/`dec_deg` stay the position of the first detection and
        are what later rows are matched against; the emitted candidate averages the whole group.
    """
    ra_deg: float
    dec_deg: float
    rows: list[dict[str, Any]] = field(default_factory=list)
    frame_paths: set[str] = field(default_factory=set)
    source_catalog_by_frame: dict[str, dict[str, Any]] = field(default_factory=dict)
    isolation_arcsec: float = math.inf

    def matches(self, row: Mapping[str, Any]) -> bool:
        """Whether a row is this same source, seen on another frame."""
        return (
            angular_distance_arcsec(row["ra_deg"], row["dec_deg"], self.ra_deg, self.dec_deg)
            <= DEFAULT_CROSSMATCH_ARCSEC
        )

    def add(self, row: Mapping[str, Any], fits_path: str) -> None:
        self.rows.append(row)
        self.frame_paths.add(fits_path)
        self.source_catalog_by_frame[fits_path] = {
            "source_label": row["source_label"],
            "flux": row["flux"],
            "mag": row["mag"],
        }

    @classmethod
    def started_by(cls, row: Mapping[str, Any], fits_path: str) -> "CandidateCluster":
        cluster = cls(ra_deg=row["ra_deg"], dec_deg=row["dec_deg"])
        cluster.add(row, fits_path)
        return cluster


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

        pipeline_diagnostics are series-level (target localization, catalog search, calibration
        strategy) and diagnostics_by_fits_basename the per-frame comparison-star checks. The
        diagnostics property concatenates both, so the two cannot drift out of step.
    """
    frames: list[FrameResult]
    selected_comparison_stars: list[ComparisonStar]
    light_curve_rows: list[LightCurveRow]
    pipeline_diagnostics: list[str]
    diagnostics_by_fits_basename: dict[str, list[str]]
    diagnostic_image_jpegs_by_fits_basename: dict[str, bytes]

    @property
    def diagnostics(self) -> list[str]:
        """Both scopes concatenated, for callers that just want everything that was said."""
        return list(self.pipeline_diagnostics) + [
            message for messages in self.diagnostics_by_fits_basename.values() for message in messages
        ]


def generate_light_curve(
    fits_paths: list[str],
    *,
    locator: TargetLocator,
    aperture_radius: float = DEFAULT_APERTURE_RADIUS,
    annulus_inner_radius: float = DEFAULT_ANNULUS_INNER_RADIUS,
    annulus_outer_radius: float = DEFAULT_ANNULUS_OUTER_RADIUS,
    min_comparisons: int = DEFAULT_MIN_COMPARISONS,
    max_comparisons: int = DEFAULT_MAX_COMPARISONS,
    progress_callback: ProgressCallback | None = None,
    comparison: ComparisonStrategy | None = None,
) -> LightCurveResult:
    """
        Generates a calibrated target light curve from local input FITS files, using comparison
        stars from the source catalog.

        Validates frame metadata and builds the comparison-star candidate catalog from headers and
        CAT tables alone, then streams pixel data one frame at a time to measure the target and
        every candidate, selects a comparison ensemble, and produces calibrated light curve rows
        with diagnostics for the frontend. At most one frame's full-resolution pixels are in
        memory at any point, so memory does not grow with the number of input frames.

        progress_callback, if given, receives (Phase, completed fraction of that phase); the
        frame-iterating phases report once per frame.

        locator decides where the target is on each frame and comparison how the comparison
        ensemble is maintained across the series; see utils/target_location.py and
        utils/comparison_calibration.py for the kinds of each, and why both choices belong to the
        caller rather than to a mode flag here.
    """
    comparison = comparison or SharedEnsemble()
    _validate_inputs(
        fits_paths=fits_paths,
        aperture_radius=aperture_radius,
        annulus_inner_radius=annulus_inner_radius,
        annulus_outer_radius=annulus_outer_radius,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
    )

    def report_progress(phase: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(phase, min(max(fraction, 0.0), 1.0))

    diagnostics: list[str] = []
    frames = _validated_frame_contexts(
        fits_paths,
        on_frame=lambda index, total: report_progress(Phase.VALIDATE, index / total),
    )
    located = locator.locate(frames)
    target_radec_by_frame = located.by_frame
    diagnostics.extend(located.diagnostics)
    log.info(
        "Aperture Photometry pipeline starting: "
        f"fits_count={len(fits_paths)}, locator={type(locator).__name__}, "
        f"aperture_radius={aperture_radius:.3f}, "
        f"annulus_inner_radius={annulus_inner_radius:.3f}, "
        f"annulus_outer_radius={annulus_outer_radius:.3f}, "
        f"min_comparisons={min_comparisons}, max_comparisons={max_comparisons}"
    )

    diagnostics_by_fits_basename: dict[str, list[str]] = {
        os.path.basename(frame.fits_path): []
        for frame in frames
    }

    catalog, catalog_diagnostics = _build_field_star_catalog(
        frames=frames,
        target_radec_by_frame=target_radec_by_frame,
        aperture_radius=aperture_radius,
        annulus_outer_radius=annulus_outer_radius,
        min_coverage_fraction=comparison.min_frame_coverage,
        on_frame=lambda index, total: report_progress(Phase.CATALOG, index / total),
    )
    diagnostics.extend(catalog_diagnostics)
    log.info(
        "Aperture Photometry comparison catalog built: "
        f"valid_candidates={len(catalog)}"
    )
    candidate_stars = candidate_stars_from_catalog(catalog)

    target_measurements: dict[str, TargetMeasurement] = {}
    measurements_by_candidate: dict[str, dict[str, ComparisonMeasurement]] = {
        candidate.candidate_id: {} for candidate in candidate_stars
    }
    drop_failed_candidates = comparison.drops_failed_candidates
    failed_candidate_ids: set[str] = set()
    for frame_index, frame in enumerate(frames, start=1):
        frame_target_ra, frame_target_dec = target_radec_by_frame[frame.fits_path]
        target, frame_measurements, newly_failed = _measure_frame_pixels(
            frame=frame,
            candidate_stars=candidate_stars,
            skip_candidate_ids=failed_candidate_ids,
            target_ra_deg=frame_target_ra,
            target_dec_deg=frame_target_dec,
            aperture_radius=aperture_radius,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
        )
        target_measurements[frame.fits_path] = target
        if drop_failed_candidates:
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
        report_progress(Phase.MEASURE, frame_index / len(frames))

    outcome = calibrate(
        CalibrationInputs(
            frames=frames,
            candidate_stars=candidate_stars,
            measurements_by_candidate=measurements_by_candidate,
            target_measurements=target_measurements,
            min_comparisons=min_comparisons,
            max_comparisons=max_comparisons,
        ),
        strategy=comparison,
    )
    frame_calibrations = outcome.frame_calibrations
    selected_comparison_stars = outcome.used_stars
    calibration_diagnostics = outcome.diagnostics
    log.info(
        "Aperture Photometry comparison stars selected: "
        f"selected_count={len(selected_comparison_stars)}, "
        f"candidate_ids={[star.candidate_id for star in selected_comparison_stars]}, "
        f"calibration_diagnostics={len(calibration_diagnostics)}"
    )
    report_progress(Phase.SELECT, 1.0)

    diagnostics.extend(calibration_diagnostics)
    pipeline_diagnostics = diagnostics

    frame_results: list[FrameResult] = []
    light_curve_rows: list[LightCurveRow] = []
    diagnostic_image_jpegs_by_fits_basename: dict[str, bytes] = {}
    for frame_index, frame in enumerate(frames, start=1):
        target = target_measurements[frame.fits_path]
        calibration = frame_calibrations[frame.fits_path]
        if not math.isfinite(calibration.calibrated_mag) or not math.isfinite(calibration.calibrated_mag_sigma):
            log.warning(
                "Aperture Photometry non-finite light-curve row: "
                f"frame={frame.fits_path}, calibrated_mag={calibration.calibrated_mag}, "
                f"calibrated_mag_sigma={calibration.calibrated_mag_sigma}. "
                "This row is present in backend output as null after JSON serialization and the frontend plot skips it."
            )
        frame_diagnostics = comparison_star_validation_diagnostics(
            frame=frame,
            stars=calibration.stars,
            measurements=calibration.measurements,
            frame_zero_point=calibration.frame_zero_point,
        )
        diagnostics_by_fits_basename[os.path.basename(frame.fits_path)].extend(frame_diagnostics)
        diagnostic_image_jpegs_by_fits_basename[os.path.basename(frame.fits_path)] = _render_frame_overlay(
            frame=frame,
            stars=calibration.stars,
            measurements=calibration.measurements,
            target_measurement=target,
            aperture_radius=aperture_radius,
        )

        frame_results.append(
            FrameResult(
                fits_path=frame.fits_path,
                date_obs=frame.date_obs,
                target_measurement=target,
                comparison_measurements=calibration.measurements,
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
                comparison_ensemble_total_counts=calibration.ensemble_flux,
                comparison_ensemble_uncertainty=math.sqrt(calibration.ensemble_variance),
                target_differential_flux=calibration.target_rel_flux,
                target_differential_flux_uncertainty=calibration.target_rel_flux_sigma,
                target_calibrated_apparent_magnitude=calibration.calibrated_mag,
                target_calibrated_apparent_magnitude_uncertainty=calibration.calibrated_mag_sigma,
            )
        )
        report_progress(Phase.RENDER, frame_index / len(frames))

    log.info(
        "Aperture Photometry pipeline completed: "
        f"frames={len(frame_results)}, light_curve_rows={len(light_curve_rows)}, "
        f"selected_comparison_stars={len(selected_comparison_stars)}, diagnostics={len(diagnostics)}"
    )
    return LightCurveResult(
        frames=frame_results,
        selected_comparison_stars=list(selected_comparison_stars),
        light_curve_rows=light_curve_rows,
        pipeline_diagnostics=pipeline_diagnostics,
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


def _validated_frame_contexts(
    fits_paths: Sequence[str],
    on_frame: Callable[[int, int], None] | None = None,
) -> list[FrameContext]:
    """
        Reads every input path into a validated FrameContext, in observation order.

        Frames that fail validation are ignored with a warning rather than aborting the run: one bad
        file out of a submitted series should cost that file, not the light curve. on_frame, if
        given, is called as on_frame(index, total) after each path, including rejected ones.
    """
    frames: list[FrameContext] = []
    for path_index, fits_path in enumerate(fits_paths, start=1):
        log.info(f"Aperture Photometry validating FITS frame: {fits_path}")
        try:
            frames.append(FrameContext.from_fits(fits_path))
        except Exception as exc:
            log.warning(
                "Aperture Photometry ignoring input frame after validation error: "
                f"frame={fits_path}, error={exc}"
            )
        if on_frame is not None:
            on_frame(path_index, len(fits_paths))

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

        Candidates whose sky position does not land on this frame are skipped before any pixel work
        (see _candidates_in_field).

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
    in_field = _candidates_in_field(frame=frame, geometry=geometry, candidate_stars=candidate_stars)
    for candidate, is_in_field in zip(candidate_stars, in_field):
        if candidate.candidate_id in skip_candidate_ids or not is_in_field:
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


def _candidates_in_field(
    *,
    frame: FrameContext,
    geometry: FrameGeometry,
    candidate_stars: Sequence[ComparisonStar],
) -> np.ndarray:
    """
        Which candidates fall on this frame with room for their background annulus, by the same edge
        criterion _build_field_star_catalog applies when the candidates are first collected.

        A drifted non-sidereal field leaves most of the catalog off any given frame, and the evolving
        strategy never drops a candidate permanently, so without this every candidate is centroided
        on every frame. Positions that do not project return NaN, which fails every comparison below.
    """
    if not candidate_stars:
        return np.zeros(0, dtype=bool)
    ra_values = np.asarray([candidate.ra_deg for candidate in candidate_stars], dtype=float)
    dec_values = np.asarray([candidate.dec_deg for candidate in candidate_stars], dtype=float)
    x_values, y_values = geometry.wcs.world_to_pixel_values(ra_values, dec_values)
    return _within_frame_bounds(
        x_values, y_values, geometry.annulus_outer_radius_px, frame.width, frame.height
    )


def _within_frame_bounds(
    x_values: np.ndarray,
    y_values: np.ndarray,
    margin: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
        Which positions sit on the frame with room for margin on every side.

        The one definition of the frame-edge rule: the catalog builder rejects by its negation, and
        the measurement pass skips by it directly. Positions that do not project are NaN, which
        fails every comparison.
    """
    return (
        (x_values - margin >= EDGE_MARGIN_PX)
        & (y_values - margin >= EDGE_MARGIN_PX)
        & (x_values + margin < width - EDGE_MARGIN_PX)
        & (y_values + margin < height - EDGE_MARGIN_PX)
    )


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
        aperture photometry. image is the frame's pixel data, passed separately from the metadata so
        the streaming pixel pass controls how long it stays in memory. geometry carries the frame's
        cached WCS and pixel-space aperture radii.

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
    # A failed centroid, or a refinement that drifts more than TARGET_RECENTER_MAX_SHIFT_PX from
    # the WCS position, means it locked onto the host galaxy or a neighbour, so fall back to the
    # WCS position.
    recenter_shift_px = math.hypot(centroid_result.x - initial_x, centroid_result.y - initial_y)
    accept_centroid = centroid_result.success and recenter_shift_px <= TARGET_RECENTER_MAX_SHIFT_PX

    if accept_centroid:
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
        if not centroid_result.success:
            reason = "centroiding failed"
        else:
            reason = f"centroid shift {recenter_shift_px:.2f}px exceeded {TARGET_RECENTER_MAX_SHIFT_PX:.2f}px limit"
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
        gain=geometry.gain,
        read_noise=geometry.read_noise,
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


def _build_field_star_catalog(
    *,
    frames: Sequence[FrameContext],
    target_radec_by_frame: Mapping[str, tuple[float, float]],
    aperture_radius: float,
    annulus_outer_radius: float,
    min_coverage_fraction: float,
    on_frame: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
        Builds comp star candidates from the source catalogs across valid frames.

        Returns candidates detected in at least min_coverage_fraction of the frames that are not too
        close to the target or the edge of the image. The target position is per frame (it moves for
        a non-sidereal target), so the target-proximity rejection tracks the moving target and never
        lets its own catalog entry become a comparison star. on_frame, if given, is called as
        on_frame(index, total) after each frame's rows are cross-matched.
    """
    clusters: list[CandidateCluster] = []
    target_pixels = {
        frame.fits_path: world_to_pixel(frame.header, *target_radec_by_frame[frame.fits_path])
        for frame in frames
    }

    for frame_index, frame in enumerate(frames, start=1):
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
            if on_frame is not None:
                on_frame(frame_index, len(frames))
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
        too_close_to_edge_mask = ~_within_frame_bounds(
            x_values, y_values, frame_annulus_outer_radius_px, frame.width, frame.height
        )

        rejected_for_target = int(np.count_nonzero(too_close_to_target_mask))
        rejected_for_edge = int(np.count_nonzero(~too_close_to_target_mask & too_close_to_edge_mask))

        for row, target_rejected, edge_rejected in zip(rows, too_close_to_target_mask, too_close_to_edge_mask):
            if target_rejected or edge_rejected:
                continue
            matched = False
            for cluster in clusters:
                if frame.fits_path in cluster.frame_paths:
                    continue
                if cluster.matches(row):
                    cluster.add(row, frame.fits_path)
                    matched = True
                    break
            if not matched:
                clusters.append(CandidateCluster.started_by(row, frame.fits_path))
        log.info(
            "Aperture Photometry comparison candidates processed: "
            f"frame={frame.fits_path}, extracted_rows={len(rows)}, "
            f"rejected_too_close_to_target={rejected_for_target}, "
            f"rejected_too_close_to_edge={rejected_for_edge}, clusters_so_far={len(clusters)}"
        )
        if on_frame is not None:
            on_frame(frame_index, len(frames))

    return _catalog_from_clusters(
        clusters=clusters,
        target_pixels=target_pixels,
        frame_count=len(frames),
        min_coverage_fraction=min_coverage_fraction,
    )


def _catalog_from_clusters(
    *,
    clusters: Sequence[CandidateCluster],
    target_pixels: Mapping[str, tuple[float, float]],
    frame_count: int,
    min_coverage_fraction: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
        Turns cross-matched clusters into comparison candidates, dropping those seen on too few
        frames and capping the pool at what the measurement budget affords.

        Returns the candidates and its own diagnostics, rather than appending to a list the caller
        owns.
    """
    # Nearest-neighbour separation for every cluster in one pass, before the coverage filter, so the
    # loop below can just read it rather than re-scanning every other cluster from inside.
    for cluster, isolation in zip(
        clusters,
        minimum_neighbor_distances_arcsec(
            [cluster.ra_deg for cluster in clusters], [cluster.dec_deg for cluster in clusters]
        ),
    ):
        cluster.isolation_arcsec = float(isolation)

    catalog: list[dict[str, Any]] = []
    rejected_for_coverage = 0
    required_coverage = max(1, math.ceil(min_coverage_fraction * frame_count))
    for idx, cluster in enumerate(
        sorted(clusters, key=lambda item: (round(item.ra_deg, 8), round(item.dec_deg, 8)))
    ):
        if len(cluster.frame_paths) < required_coverage:
            rejected_for_coverage += 1
            continue
        catalog.append(
            {
                "candidate_id": f"cand-{idx + 1:03d}",
                "ra_deg": float(np.mean([row["ra_deg"] for row in cluster.rows])),
                "dec_deg": float(np.mean([row["dec_deg"] for row in cluster.rows])),
                "second_hdu_magnitude": float(
                    np.median(np.asarray([row["mag"] for row in cluster.rows], dtype=float))
                ),
                "source_catalog_by_frame": dict(cluster.source_catalog_by_frame),
                "frame_coverage": len(cluster.frame_paths),
                "isolation_arcsec": cluster.isolation_arcsec,
                "target_separation_px": min(
                    distance_pixels(row["pixel_x"], row["pixel_y"], *target_pixels[row["frame_path"]])
                    for row in cluster.rows
                ),
            }
        )

    # Without a coverage filter the pool is every cross-matched cluster on the field, down to
    # single-frame detections, so cap it at the best covered and most isolated. Applied whatever the
    # coverage fraction: the bound is on measurement cost, which no strategy is exempt from.
    diagnostics: list[str] = []
    capped_out = 0
    pool_limit = _candidate_pool_limit(frame_count)
    if len(catalog) > pool_limit:
        capped_out = len(catalog) - pool_limit
        ranked = sorted(catalog, key=_candidate_pool_rank)[:pool_limit]
        catalog = sorted(ranked, key=lambda candidate: candidate["candidate_id"])
        diagnostics.append(
            f"Kept the {pool_limit} best-covered comparison candidates of "
            f"{pool_limit + capped_out} found on this field, to bound measurement over "
            f"{frame_count} frames."
        )
    log.info(
        "Aperture Photometry comparison catalog summary: "
        f"clusters={len(clusters)}, required_coverage={required_coverage}/{frame_count} frames, "
        f"rejected_insufficient_coverage={rejected_for_coverage}, "
        f"rejected_over_candidate_cap={capped_out}, "
        f"valid_catalog_candidates={len(catalog)}"
    )
    return catalog, diagnostics


def _candidate_pool_limit(frame_count: int) -> int:
    """How many candidates this many frames can afford, from the candidates x frames budget."""
    return max(MIN_COMPARISON_CANDIDATES, MAX_CANDIDATE_MEASUREMENTS // max(frame_count, 1))


def _candidate_pool_rank(candidate: Mapping[str, Any]) -> tuple[float, float, str]:
    """
        Ranks candidates for the pool cap: widest frame coverage first, then most isolated.

        Coverage leads because a star present on many frames ties more of the series together, which
        is what the evolving calibration runs on. A non-finite isolation sorts last.
    """
    isolation = float(candidate["isolation_arcsec"])
    return (
        -float(candidate["frame_coverage"]),
        -isolation if math.isfinite(isolation) else math.inf,
        str(candidate["candidate_id"]),
    )


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
