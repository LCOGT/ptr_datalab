import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from dateutil.parser import ParserError, parse as parse_date

from datalab.datalab_session.utils.comparison_calibration import (
    COMPARISON_AUTO,
    COMPARISON_EVOLVING,
    COMPARISON_SHARED,
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
    frame_midpoint_mjd,
    optional_float,
    target_radec_from_header,
    world_to_pixel,
)
from datalab.datalab_session.utils.geometry import (
    angular_distance_arcsec,
    distance_pixels,
    minimum_angular_neighbor_distance_arcsec,
)
from datalab.datalab_session.utils.photometry_diagnostics import (
    candidate_overlay_jpeg_bytes,
    comparison_star_validation_diagnostics,
)
from datalab.datalab_session.utils.moving_target_search import (
    DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
    refine_positions_from_catalog,
)
from datalab.datalab_session.utils.photometry import measure_aperture
from datalab.datalab_session.utils.target_track import (
    LINEAR_TRACK_MAX_SPAN_HOURS,
    MAX_TRACK_FIT_ORDER,
    TargetTrack,
    TrackSeed,
    fit_target_track,
    track_rate_arcsec_per_minute,
    track_seeds_from_input,
)

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
# Pipeline phases reported to progress_callback, in execution order.
PROGRESS_PHASES = ("validate", "catalog", "measure", "select", "render")

# Receives (phase, fraction): phase is one of PROGRESS_PHASES and fraction is the completed
# share of that phase, in [0, 1].
ProgressCallback = Callable[[str, float], None]

# Target-position source. "fixed" is a single series-wide RA/Dec (sidereal). "header" reads a
# per-frame RA/Dec from each frame's moving-target ephemeris keywords, for frames whose mount tracked
# the object. "track" interpolates a per-frame RA/Dec from positions the user marked on a handful of
# frames, for a moving target imaged on a sidereally-tracked field, where nothing in the header says
# where the object is.
TARGET_POSITION_FIXED = "fixed"
TARGET_POSITION_HEADER = "header"
TARGET_POSITION_TRACK = "track"
# Target refinement: centroid around the predicted pixel with a recenter cap ("centroid"), or measure
# at the predicted pixel without recentering ("forced" — for extended/cometary targets whose centroid
# wanders off the ephemeris, reusing the WCS-fallback measurement path).
REFINEMENT_CENTROID = "centroid"
REFINEMENT_FORCED = "forced"


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
    target_ra_deg: float | None = None,
    target_dec_deg: float | None = None,
    aperture_radius: float = DEFAULT_APERTURE_RADIUS,
    annulus_inner_radius: float = DEFAULT_ANNULUS_INNER_RADIUS,
    annulus_outer_radius: float = DEFAULT_ANNULUS_OUTER_RADIUS,
    min_comparisons: int = DEFAULT_MIN_COMPARISONS,
    max_comparisons: int = DEFAULT_MAX_COMPARISONS,
    progress_callback: ProgressCallback | None = None,
    *,
    target_position_mode: str = TARGET_POSITION_FIXED,
    refinement_mode: str = REFINEMENT_CENTROID,
    comparison_mode: str = COMPARISON_SHARED,
    target_track_seeds: Sequence[TrackSeed] | None = None,
    track_search_radius_arcsec: float = DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
) -> LightCurveResult:
    """
        Generates a calibrated target light curve from local input FITS files, using comparison
        stars from the source catalog.

        Validates frame metadata and builds the comparison-star candidate catalog from headers and
        CAT tables alone, then streams pixel data one frame at a time to measure the target and
        every candidate, selects a comparison ensemble, and produces calibrated light curve rows
        with diagnostics for the frontend. At most one frame's full-resolution pixels are in
        memory at any point, so memory does not grow with the number of input frames.

        progress_callback, if given, receives (phase, completed fraction of that phase) with
        phases from PROGRESS_PHASES; the frame-iterating phases report once per frame.

        target_position_mode selects where the target's per-frame RA/Dec comes from: "fixed" uses the
        single series-wide target_ra_deg/target_dec_deg (sidereal); "header" reads each frame's
        moving-target keywords (non-sidereal); "track" interpolates target_track_seeds, the positions
        a user marked on two or more frames, to every frame's observation time. refinement_mode
        selects "centroid" (recenter with a cap, falling back to the predicted pixel) or "forced"
        (measure at the predicted pixel).
    """
    _validate_modes(target_position_mode, refinement_mode, comparison_mode)
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
        on_frame=lambda index, total: report_progress("validate", index / total),
    )
    target_radec_by_frame = _resolve_target_positions(
        frames=frames,
        target_position_mode=target_position_mode,
        target_ra_deg=target_ra_deg,
        target_dec_deg=target_dec_deg,
        target_track_seeds=target_track_seeds,
        track_search_radius_arcsec=track_search_radius_arcsec,
        diagnostics=diagnostics,
    )
    log.info(
        "Aperture Photometry pipeline starting: "
        f"fits_count={len(fits_paths)}, target_position_mode={target_position_mode}, "
        f"refinement_mode={refinement_mode}, "
        f"aperture_radius={aperture_radius:.3f}, "
        f"annulus_inner_radius={annulus_inner_radius:.3f}, "
        f"annulus_outer_radius={annulus_outer_radius:.3f}, "
        f"min_comparisons={min_comparisons}, max_comparisons={max_comparisons}"
    )

    diagnostics_by_fits_basename: dict[str, list[str]] = {
        os.path.basename(frame.fits_path): []
        for frame in frames
    }

    # "shared" only ever uses stars present on every frame, so pre-filtering to the coverage fraction
    # keeps its measurement cost down. "evolving"/"auto" must keep sparsely-detected stars (they carry
    # a drifted field), so they retain every cross-matched candidate and pay to measure them all.
    min_coverage_fraction = (
        COMPARISON_FRAME_COVERAGE_FRACTION if comparison_mode == COMPARISON_SHARED else 0.0
    )
    catalog = _build_field_star_catalog(
        frames=frames,
        target_radec_by_frame=target_radec_by_frame,
        aperture_radius=aperture_radius,
        annulus_outer_radius=annulus_outer_radius,
        min_coverage_fraction=min_coverage_fraction,
        on_frame=lambda index, total: report_progress("catalog", index / total),
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
    # A shared ensemble needs every star on every frame, so a candidate that fails once cannot
    # contribute and is dropped from further measurement to save work. An evolving ensemble uses each
    # frame's own in-field stars, so a star that leaves the frame (fails here) must still be measured
    # on the frames where it *is* present -- never permanently dropped.
    drop_failed_candidates = comparison_mode == COMPARISON_SHARED
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
            refinement_mode=refinement_mode,
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
        report_progress("measure", frame_index / len(frames))

    frame_calibrations, selected_comparison_stars, calibration_diagnostics = calibrate(
        comparison_mode=comparison_mode,
        frames=frames,
        candidate_stars=candidate_stars,
        measurements_by_candidate=measurements_by_candidate,
        target_measurements=target_measurements,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
        error_class=LightCurveError,
    )
    log.info(
        "Aperture Photometry comparison stars selected: "
        f"selected_count={len(selected_comparison_stars)}, "
        f"candidate_ids={[star.candidate_id for star in selected_comparison_stars]}, "
        f"calibration_diagnostics={len(calibration_diagnostics)}"
    )
    report_progress("select", 1.0)

    diagnostics.extend(calibration_diagnostics)

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
        diagnostics.extend(frame_diagnostics)
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
        report_progress("render", frame_index / len(frames))

    log.info(
        "Aperture Photometry pipeline completed: "
        f"frames={len(frame_results)}, light_curve_rows={len(light_curve_rows)}, "
        f"selected_comparison_stars={len(selected_comparison_stars)}, diagnostics={len(diagnostics)}"
    )
    return LightCurveResult(
        frames=frame_results,
        selected_comparison_stars=list(selected_comparison_stars),
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


def _validate_modes(target_position_mode: str, refinement_mode: str, comparison_mode: str) -> None:
    if target_position_mode not in (TARGET_POSITION_FIXED, TARGET_POSITION_HEADER, TARGET_POSITION_TRACK):
        raise LightCurveError(
            f"target_position_mode must be one of {TARGET_POSITION_FIXED!r}, "
            f"{TARGET_POSITION_HEADER!r}, {TARGET_POSITION_TRACK!r}."
        )
    if refinement_mode not in (REFINEMENT_CENTROID, REFINEMENT_FORCED):
        raise LightCurveError(
            f"refinement_mode must be {REFINEMENT_CENTROID!r} or {REFINEMENT_FORCED!r}."
        )
    if comparison_mode not in (COMPARISON_SHARED, COMPARISON_EVOLVING, COMPARISON_AUTO):
        raise LightCurveError(
            f"comparison_mode must be one of {COMPARISON_SHARED!r}, {COMPARISON_EVOLVING!r}, {COMPARISON_AUTO!r}."
        )


def _resolve_target_positions(
    *,
    frames: Sequence[FrameContext],
    target_position_mode: str,
    target_ra_deg: float | None,
    target_dec_deg: float | None,
    target_track_seeds: Sequence[TrackSeed] | None = None,
    track_search_radius_arcsec: float = DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
    diagnostics: list[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """
        Resolves the target RA/Dec (degrees) used on each frame.

        In "fixed" mode every frame shares the one series-wide target position. In "header" mode the
        moving target's position is read from each frame's ephemeris keywords, so the pixel it lands
        on changes frame to frame; a frame whose keywords are absent/unparseable raises, since the
        target cannot be located without them. In "track" mode a polynomial is fitted through the
        user's seed positions and evaluated at each frame's exposure midpoint.
    """
    if target_position_mode == TARGET_POSITION_FIXED:
        if target_ra_deg is None or target_dec_deg is None:
            raise LightCurveError("Fixed target position requires target_ra_deg and target_dec_deg.")
        fixed = (float(target_ra_deg), float(target_dec_deg))
        return {frame.fits_path: fixed for frame in frames}

    if target_position_mode == TARGET_POSITION_TRACK:
        return _track_target_positions(
            frames=frames,
            target_track_seeds=target_track_seeds,
            track_search_radius_arcsec=track_search_radius_arcsec,
            diagnostics=diagnostics if diagnostics is not None else [],
        )

    positions: dict[str, tuple[float, float]] = {}
    for frame in frames:
        try:
            positions[frame.fits_path] = target_radec_from_header(frame.header)
        except ValueError as exc:
            raise LightCurveError(f"Cannot read moving-target position for {frame.fits_path}: {exc}") from exc
        log.info(
            "Aperture Photometry moving-target position: "
            f"frame={frame.fits_path}, ra={positions[frame.fits_path][0]:.8f}, "
            f"dec={positions[frame.fits_path][1]:.8f}"
        )
    return positions


def _track_target_positions(
    *,
    frames: Sequence[FrameContext],
    target_track_seeds: Sequence[TrackSeed] | None,
    track_search_radius_arcsec: float = DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
    diagnostics: list[str],
) -> dict[str, tuple[float, float]]:
    """
        Locates the moving target on every frame, starting from the user's seed positions.

        The seeds are interpolated to each frame's exposure midpoint -- not its start, because that
        is the position the target's trail is centred on -- to predict where the target should be.
        That prediction is then used as a search position: the frame's own source catalog is checked
        for a detection near it that is not a field star, and the track is refitted through the
        detections that move consistently. Where that succeeds the target is measured at a real
        detected position rather than an interpolated guess; where it does not, the interpolated
        position stands, so a faint or uncatalogued target still yields a measurement.

        Frames outside the seed time span are extrapolated rather than dropped -- the fit is still
        the best information available -- but both extrapolation and a long arc carried by only two
        seeds are surfaced as diagnostics, since those are the two ways a predicted position quietly
        drifts off the object.
    """
    if not target_track_seeds:
        raise LightCurveError(
            f"Target position mode {TARGET_POSITION_TRACK!r} requires target_track_seeds: "
            "the positions the target was identified at on two or more frames."
        )
    try:
        track = fit_target_track(target_track_seeds)
    except ValueError as exc:
        raise LightCurveError(f"Cannot fit a target track from the supplied seeds: {exc}") from exc

    rate_arcsec_per_minute = track_rate_arcsec_per_minute(track)
    diagnostics.append(
        f"Target track fitted from {len(track.seeds)} seed position(s) as a degree-{track.order} "
        f"polynomial over a {track.seed_span_hours:.2f} h arc, mean apparent rate "
        f"{rate_arcsec_per_minute:.3f} arcsec/min."
    )

    frame_times: list[tuple[str, float]] = []
    extrapolated: list[str] = []
    for frame in frames:
        try:
            midpoint_mjd = frame_midpoint_mjd(frame.header, fallback_start=frame.date_obs)
        except ValueError as exc:
            raise LightCurveError(f"Cannot determine an observation time for {frame.fits_path}: {exc}") from exc
        frame_times.append((frame.fits_path, midpoint_mjd))
        if not track.covers(midpoint_mjd):
            extrapolated.append(os.path.basename(frame.fits_path))

    refinement = refine_positions_from_catalog(
        frame_times=frame_times,
        catalog_rows_by_frame={frame.fits_path: frame.second_hdu_rows for frame in frames},
        track=track,
        seeds=track.seeds,
        search_radius_arcsec=track_search_radius_arcsec,
    )
    positions = refinement.positions
    diagnostics.extend(refinement.diagnostics)
    for fits_path, midpoint_mjd in frame_times:
        log.info(
            "Aperture Photometry tracked-target position: "
            f"frame={fits_path}, midpoint_mjd={midpoint_mjd:.8f}, "
            f"ra={positions[fits_path][0]:.8f}, dec={positions[fits_path][1]:.8f}"
        )

    if extrapolated:
        diagnostics.append(
            f"{len(extrapolated)} frame(s) fall outside the seed time span and were extrapolated "
            f"rather than interpolated, so their predicted positions are the least reliable: "
            f"{', '.join(extrapolated)}."
        )
    if track.order < MAX_TRACK_FIT_ORDER and track.seed_span_hours > LINEAR_TRACK_MAX_SPAN_HOURS:
        diagnostics.append(
            f"Only {len(track.seeds)} seed positions were supplied over a "
            f"{track.seed_span_hours:.1f} h arc, so the track is a straight line. Apparent tracks "
            f"curve over spans beyond about {LINEAR_TRACK_MAX_SPAN_HOURS:.0f} h; identifying the "
            "target on a third frame near the middle of the series would fit a curve instead and "
            "keep the predicted positions on the object."
        )
    return positions


def _validated_frame_contexts(
    fits_paths: Sequence[str],
    on_frame: Callable[[int, int], None] | None = None,
) -> list[FrameContext]:
    """
        Builds validated frame metadata for each input FITS path.

        Reads only the SCI header and the CAT table -- never SCI pixel data -- so validation memory
        and time stay flat regardless of frame count or sensor size. Frames that fail validation
        are ignored with a warning. on_frame, if given, is called as on_frame(index, total) after
        each input path is processed, including rejected ones.
    """
    frames: list[FrameContext] = []
    for path_index, fits_path in enumerate(fits_paths, start=1):
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
    refinement_mode: str = REFINEMENT_CENTROID,
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
        refinement_mode=refinement_mode,
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
    refinement_mode: str = REFINEMENT_CENTROID,
) -> TargetMeasurement:
    """
        Converts the target RA and Dec to pixel coordinates, optionally centroids the source, and
        measures aperture photometry. image is the frame's pixel data, passed separately from the
        metadata so the streaming pixel pass controls how long it stays in memory. geometry carries
        the frame's cached WCS and pixel-space aperture radii.

        In "centroid" mode the target is never allowed to drop a frame: if centroiding fails or the
        refinement drifts too far from the WCS position, it measures at the authoritative WCS
        position instead. In "forced" mode it always measures at the WCS/ephemeris position without
        recentering -- for extended (cometary) targets whose light centroid wanders off the
        ephemeris, where chasing the centroid would bias the position.

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

    if refinement_mode == REFINEMENT_FORCED:
        centroid_result = None
        recenter_shift_px = 0.0
        accept_centroid = False
    else:
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
        if refinement_mode == REFINEMENT_FORCED:
            reason = "forced photometry at ephemeris position"
        elif not centroid_result.success:
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
    min_coverage_fraction: float = COMPARISON_FRAME_COVERAGE_FRACTION,
    on_frame: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """
        Builds comp star candidates from the source catalogs across valid frames.

        Returns candidates detected in at least min_coverage_fraction of the frames that are not too
        close to the target or the edge of the image. The target position is per frame (it moves for
        a non-sidereal target), so the target-proximity rejection tracks the moving target and never
        lets its own catalog entry become a comparison star. on_frame, if given, is called as
        on_frame(index, total) after each frame's rows are cross-matched.
    """
    clusters: list[dict[str, Any]] = []
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
        if on_frame is not None:
            on_frame(frame_index, len(frames))

    catalog: list[dict[str, Any]] = []
    rejected_for_coverage = 0
    required_coverage = max(1, math.ceil(min_coverage_fraction * len(frames)))
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
