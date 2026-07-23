"""
Comparison-set calibration strategies for the aperture photometry light curve.

Both strategies place every frame on the catalog magnitude system via a per-frame, catalog-anchored
zero point -- the calibration that already lets the sidereal pipeline combine frames across sites,
instruments, and telescope classes. They differ only in the *identity* of the comparison ensemble:

- "shared": one ensemble present on every frame (sidereal, and non-sidereal fields that barely drift).
- "evolving": a per-frame ensemble whose zero points are solved jointly over the sparse (star x frame)
  matrix and anchored to catalog magnitudes, so a drifting field whose comparison stars turn over
  completely still yields one coherent light curve. Frames that share no stars (a fully turned-over
  field, or a long time gap) form separate groups, each anchored to catalog magnitudes on its own and
  flagged, since they cannot be tied together steplessly.

Kept out of aperture_light_curve.py so that module stays focused on the pixel/measurement pipeline.
"""
from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from datalab.datalab_session.utils.comparison_stars import (
    ComparisonMeasurement,
    ComparisonStar,
    MAX_ACCEPTABLE_VARIABILITY,
    select_comparison_stars,
)
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag

if TYPE_CHECKING:  # avoid a runtime import cycle with aperture_light_curve
    from datalab.datalab_session.utils.aperture_light_curve import FrameContext, TargetMeasurement

log = logging.getLogger()
log.setLevel(logging.INFO)

# Comparison-set strategy: one ensemble present on every frame ("shared", sidereal); an evolving
# ensemble whose per-frame zero point is solved jointly over the sparse star x frame matrix and
# anchored to catalog magnitudes ("evolving", drifted non-sidereal fields); or "auto" -- try "shared"
# and fall back to "evolving" when no ensemble spans the series.
COMPARISON_SHARED = "shared"
COMPARISON_EVOLVING = "evolving"
COMPARISON_AUTO = "auto"
# Evolving self-calibration solve controls.
EVOLVING_SOLVE_ITERATIONS = 50
EVOLVING_SOLVE_TOLERANCE_MAG = 1e-6
# d(magnitude)/d(fractional flux error) = 2.5/ln(10); converts a flux SNR into a magnitude sigma.
MAG_PER_DEX = 2.5 / math.log(10.0)


@dataclass(frozen=True)
class FrameCalibration:
    """
        Everything the per-frame light-curve row and diagnostics need for one frame, produced by a
        calibration strategy. stars and measurements are that frame's active comparison ensemble
        (fixed across frames for "shared", per-frame for "evolving") and are aligned.
    """
    stars: tuple[ComparisonStar, ...]
    measurements: tuple[ComparisonMeasurement, ...]
    ensemble_flux: float
    ensemble_variance: float
    target_rel_flux: float
    target_rel_flux_sigma: float
    frame_zero_point: float
    calibrated_mag: float
    calibrated_mag_sigma: float


def calibrate(
    *,
    comparison_mode: str,
    frames: Sequence["FrameContext"],
    candidate_stars: Sequence[ComparisonStar],
    measurements_by_candidate: Mapping[str, Mapping[str, ComparisonMeasurement]],
    target_measurements: Mapping[str, "TargetMeasurement"],
    min_comparisons: int,
    max_comparisons: int,
    error_class: type[Exception] = ValueError,
) -> tuple[dict[str, FrameCalibration], tuple[ComparisonStar, ...], list[str]]:
    """
        Calibrates the light curve: dispatches to a comparison-set strategy, then widens each point's
        uncertainty by an empirical error floor. Returns the per-frame calibration, the comparison
        stars used across the series, and calibration-level diagnostics.
    """
    frame_calibrations, used_stars, diagnostics = _dispatch_calibration(
        comparison_mode=comparison_mode,
        frames=frames,
        candidate_stars=candidate_stars,
        measurements_by_candidate=measurements_by_candidate,
        target_measurements=target_measurements,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
        error_class=error_class,
    )

    # The formal per-point uncertainty carries only photon + zero-point-scatter terms and badly
    # underestimates real frame-to-frame reproducibility (flat-field, scintillation, catalog noise).
    # Measure that systematic floor directly from how well the comparison stars' own calibrated
    # magnitudes repeat frame to frame, and add it in quadrature to every point.
    floor = _empirical_error_floor(frame_calibrations, [frame.fits_path for frame in frames])
    if floor > 0.0:
        frame_calibrations = {
            fits_path: (
                replace(calibration, calibrated_mag_sigma=math.hypot(calibration.calibrated_mag_sigma, floor))
                if math.isfinite(calibration.calibrated_mag_sigma)
                else calibration
            )
            for fits_path, calibration in frame_calibrations.items()
        }
        diagnostics = list(diagnostics) + [
            f"Applied a {floor * 1000.0:.1f} mmag empirical error floor (comparison-star "
            "frame-to-frame reproducibility) in quadrature to each point's uncertainty."
        ]
    return frame_calibrations, used_stars, diagnostics


def _dispatch_calibration(
    *,
    comparison_mode: str,
    frames: Sequence["FrameContext"],
    candidate_stars: Sequence[ComparisonStar],
    measurements_by_candidate: Mapping[str, Mapping[str, ComparisonMeasurement]],
    target_measurements: Mapping[str, "TargetMeasurement"],
    min_comparisons: int,
    max_comparisons: int,
    error_class: type[Exception] = ValueError,
) -> tuple[dict[str, FrameCalibration], tuple[ComparisonStar, ...], list[str]]:
    """
        Dispatches to a comparison-set calibration strategy, returning a per-frame calibration, the
        comparison stars used across the series, and calibration-level diagnostics.

        "auto" tries the shared ensemble first (identical to the sidereal path) and falls back to the
        evolving per-frame calibration when no ensemble spans the series -- from the one pixel pass
        already done, so the fallback costs no re-measurement.
    """
    if comparison_mode == COMPARISON_EVOLVING:
        return _calibrate_evolving(
            frames=frames,
            candidate_stars=candidate_stars,
            measurements_by_candidate=measurements_by_candidate,
            target_measurements=target_measurements,
            min_comparisons=min_comparisons,
            error_class=error_class,
        )
    if comparison_mode == COMPARISON_SHARED:
        return _calibrate_shared(
            frames=frames,
            candidate_stars=candidate_stars,
            measurements_by_candidate=measurements_by_candidate,
            target_measurements=target_measurements,
            min_comparisons=min_comparisons,
            max_comparisons=max_comparisons,
            error_class=error_class,
        )
    try:
        return _calibrate_shared(
            frames=frames,
            candidate_stars=candidate_stars,
            measurements_by_candidate=measurements_by_candidate,
            target_measurements=target_measurements,
            min_comparisons=min_comparisons,
            max_comparisons=max_comparisons,
            error_class=error_class,
        )
    except error_class as exc:
        log.warning(
            "Aperture Photometry shared comparison ensemble unavailable; "
            f"falling back to evolving calibration: {exc}"
        )
        calibrations, used_stars, diagnostics = _calibrate_evolving(
            frames=frames,
            candidate_stars=candidate_stars,
            measurements_by_candidate=measurements_by_candidate,
            target_measurements=target_measurements,
            min_comparisons=min_comparisons,
            error_class=error_class,
        )
        return (
            calibrations,
            used_stars,
            [f"No comparison ensemble spans every frame ({exc}); used evolving per-frame calibration."]
            + list(diagnostics),
        )


def _calibrate_shared(
    *,
    frames: Sequence["FrameContext"],
    candidate_stars: Sequence[ComparisonStar],
    measurements_by_candidate: Mapping[str, Mapping[str, ComparisonMeasurement]],
    target_measurements: Mapping[str, "TargetMeasurement"],
    min_comparisons: int,
    max_comparisons: int,
    error_class: type[Exception],
) -> tuple[dict[str, FrameCalibration], tuple[ComparisonStar, ...], list[str]]:
    """
        Sidereal calibration: one comparison ensemble present on every frame, calibrated by a
        per-frame catalog-anchored zero point. Each frame's zero point is
        frame_zero_point = 2.5*log10(measured ensemble counts) + ensemble catalog magnitude, so the
        calibration self-corrects that frame's airmass/instrument/exposure. Unchanged behavior from
        before the strategy split.
    """
    target_mag_proxy = _target_magnitude_proxy(target_measurements.values(), error_class=error_class)
    log.info(f"Aperture Photometry target magnitude proxy: {target_mag_proxy:.6f}")
    selection = select_comparison_stars(
        frames=frames,
        candidates=candidate_stars,
        measurements_by_candidate=measurements_by_candidate,
        target_mag_proxy=target_mag_proxy,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
        error_class=error_class,
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
        raise error_class("Comparison-star magnitude calibration produced a non-positive ensemble reference flux.")
    ensemble_reference_mag = -2.5 * math.log10(ensemble_reference_flux)

    frame_calibrations: dict[str, FrameCalibration] = {}
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
            raise error_class(f"Ensemble comparison flux is invalid for frame {frame.fits_path}.")

        target_rel_flux, target_rel_flux_sigma = relative_flux(target, ensemble_flux, ensemble_variance)
        calibrated_flux = target_rel_flux * ensemble_reference_flux
        frame_zero_point = 2.5 * math.log10(ensemble_flux) + ensemble_reference_mag
        if calibrated_flux > 0.0:
            calibrated_mag = -2.5 * math.log10(target.net_source_counts) + frame_zero_point
            _, calibrated_mag_sigma = flux_to_mag(target_rel_flux, target_rel_flux_sigma)
        else:
            calibrated_mag = math.nan
            calibrated_mag_sigma = math.nan
        frame_calibrations[frame.fits_path] = FrameCalibration(
            stars=tuple(selection.selected_stars),
            measurements=comparison_measurements,
            ensemble_flux=ensemble_flux,
            ensemble_variance=ensemble_variance,
            target_rel_flux=target_rel_flux,
            target_rel_flux_sigma=target_rel_flux_sigma,
            frame_zero_point=frame_zero_point,
            calibrated_mag=calibrated_mag,
            calibrated_mag_sigma=calibrated_mag_sigma,
        )
    return frame_calibrations, tuple(selection.selected_stars), []


def _calibrate_evolving(
    *,
    frames: Sequence["FrameContext"],
    candidate_stars: Sequence[ComparisonStar],
    measurements_by_candidate: Mapping[str, Mapping[str, ComparisonMeasurement]],
    target_measurements: Mapping[str, "TargetMeasurement"],
    min_comparisons: int,
    error_class: type[Exception],
) -> tuple[dict[str, FrameCalibration], tuple[ComparisonStar, ...], list[str]]:
    """
        Evolving calibration for a drifting non-sidereal field, where no single ensemble spans the
        series. Solves a per-frame catalog-anchored zero point jointly over the sparse (star x frame)
        matrix: comparison stars link the frames they share, so a star entering the field is tied to
        the same internal magnitude system the incumbents define, and membership can turn over
        completely without stepping the light curve. Disconnected groups (a fully turned-over field
        or a long time gap with no shared stars) are each anchored to catalog magnitudes on their own
        and flagged, since they cannot be tied steplessly.
    """
    star_by_id = {candidate.candidate_id: candidate for candidate in candidate_stars}
    # present[candidate_id][fits_path] = (instrumental_mag, measurement) for positive measurements.
    present: dict[str, dict[str, tuple[float, ComparisonMeasurement]]] = {}
    for candidate_id, per_frame in measurements_by_candidate.items():
        candidate = star_by_id.get(candidate_id)
        if candidate is None or not math.isfinite(candidate.reference_magnitude):
            continue
        rows: dict[str, tuple[float, ComparisonMeasurement]] = {}
        for fits_path, measurement in per_frame.items():
            counts = measurement.net_source_counts
            if math.isfinite(counts) and counts > 0.0:
                rows[fits_path] = (-2.5 * math.log10(counts), measurement)
        if rows:
            present[candidate_id] = rows
    if not present:
        raise error_class("Evolving calibration found no usable comparison-star measurements.")

    frame_paths = [frame.fits_path for frame in frames]
    components = _connected_components(present)

    zero_point_by_frame: dict[str, float] = {}
    zp_sigma_by_frame: dict[str, float] = {}
    active_ids_by_frame: dict[str, list[str]] = {fits_path: [] for fits_path in frame_paths}
    used_ids: set[str] = set()
    for component in components:
        zero_points, zp_sigmas, active_ids = _solve_component_zero_points(
            star_ids=component["stars"],
            frame_paths=component["frames"],
            present=present,
            star_by_id=star_by_id,
        )
        zero_point_by_frame.update(zero_points)
        zp_sigma_by_frame.update(zp_sigmas)
        for fits_path, ids in active_ids.items():
            active_ids_by_frame[fits_path] = ids
            used_ids.update(ids)

    diagnostics: list[str] = []
    populated_components = [component for component in components if component["frames"]]
    if len(populated_components) > 1:
        diagnostics.append(
            f"Evolving calibration: the comparison field splits into {len(populated_components)} "
            "groups sharing no common stars, each anchored to catalog magnitudes independently; "
            "magnitude steps may exist between them."
        )

    frame_calibrations: dict[str, FrameCalibration] = {}
    for frame in frames:
        fits_path = frame.fits_path
        target = target_measurements[fits_path]
        active_ids = active_ids_by_frame.get(fits_path, [])
        measurements = tuple(present[candidate_id][fits_path][1] for candidate_id in active_ids)
        stars = tuple(star_by_id[candidate_id] for candidate_id in active_ids)
        counts = np.asarray([measurement.net_source_counts for measurement in measurements], dtype=float)
        uncertainties = np.asarray([measurement.source_uncertainty for measurement in measurements], dtype=float)
        ensemble_flux = float(np.sum(counts)) if counts.size else 0.0
        ensemble_variance = float(np.sum(np.square(uncertainties))) if uncertainties.size else 0.0
        target_rel_flux, target_rel_flux_sigma = relative_flux(target, ensemble_flux, ensemble_variance)

        zero_point = zero_point_by_frame.get(fits_path, math.nan)
        zp_sigma = zp_sigma_by_frame.get(fits_path, math.nan)
        if len(active_ids) >= min_comparisons and math.isfinite(zero_point) and target.net_source_counts > 0.0:
            calibrated_mag = -2.5 * math.log10(target.net_source_counts) + zero_point
            target_inst_sigma = MAG_PER_DEX * (target.source_uncertainty / target.net_source_counts)
            calibrated_mag_sigma = math.hypot(target_inst_sigma, zp_sigma if math.isfinite(zp_sigma) else 0.0)
        else:
            calibrated_mag = math.nan
            calibrated_mag_sigma = math.nan
            if len(active_ids) < min_comparisons:
                diagnostics.append(
                    f"Evolving calibration: {os.path.basename(fits_path)} has {len(active_ids)} usable "
                    f"comparison stars (< {min_comparisons}); light-curve row omitted."
                )
        frame_calibrations[fits_path] = FrameCalibration(
            stars=stars,
            measurements=measurements,
            ensemble_flux=ensemble_flux,
            ensemble_variance=ensemble_variance,
            target_rel_flux=target_rel_flux,
            target_rel_flux_sigma=target_rel_flux_sigma,
            frame_zero_point=zero_point if math.isfinite(zero_point) else 0.0,
            calibrated_mag=calibrated_mag,
            calibrated_mag_sigma=calibrated_mag_sigma,
        )

    used_stars = tuple(star_by_id[candidate_id] for candidate_id in sorted(used_ids))
    if not used_stars:
        raise error_class("Evolving calibration produced no usable comparison stars.")
    log.info(
        "Aperture Photometry evolving calibration: "
        f"components={len(populated_components)}, used_comparison_stars={len(used_stars)}"
    )
    return frame_calibrations, used_stars, diagnostics


def relative_flux(target: "TargetMeasurement", ensemble_flux: float, ensemble_variance: float) -> tuple[float, float]:
    """
        Target flux relative to the comparison ensemble, and its uncertainty from the target and
        ensemble photon errors. NaN sigma when the target has non-positive counts (measuring blank
        sky), where the fractional error is undefined.
    """
    if not math.isfinite(ensemble_flux) or ensemble_flux <= 0.0:
        return math.nan, math.nan
    target_rel_flux = target.net_source_counts / ensemble_flux
    if target.net_source_counts > 0.0:
        target_variance = target.source_uncertainty * target.source_uncertainty
        target_rel_flux_sigma = abs(target_rel_flux) * math.sqrt(
            target_variance / (target.net_source_counts * target.net_source_counts)
            + ensemble_variance / (ensemble_flux * ensemble_flux)
        )
    else:
        target_rel_flux_sigma = math.nan
    return target_rel_flux, target_rel_flux_sigma


def _empirical_error_floor(
    frame_calibrations: Mapping[str, FrameCalibration],
    frame_order: Sequence[str],
) -> float:
    """
        The systematic per-point error floor, measured from the comparison stars themselves: how well
        each star's *own* calibrated magnitude repeats frame to frame. Residuals are taken about each
        star's series mean, so a star's static catalog offset (which cancels in the differential light
        curve) does not inflate the floor; the median over stars is robust to a few variable stars.
        This captures flat-field / scintillation / catalog scatter that the formal photon + zero-point
        error omits. Needs >= 3 frames per star; returns 0.0 when the series is too sparse to estimate.
    """
    calibrated_by_star: dict[str, list[float]] = defaultdict(list)
    for fits_path in frame_order:
        calibration = frame_calibrations.get(fits_path)
        # Only frames that yielded a real light-curve row carry a trustworthy zero point.
        if calibration is None or not math.isfinite(calibration.calibrated_mag):
            continue
        for star, measurement in zip(calibration.stars, calibration.measurements):
            counts = measurement.net_source_counts
            if math.isfinite(counts) and counts > 0.0:
                calibrated_by_star[star.candidate_id].append(
                    -2.5 * math.log10(counts) + calibration.frame_zero_point
                )
    per_star_rms = [
        float(np.std(np.asarray(values, dtype=float)))
        for values in calibrated_by_star.values()
        if len(values) >= 3
    ]
    if not per_star_rms:
        return 0.0
    return float(np.median(per_star_rms))


def _target_magnitude_proxy(
    measurements: Iterable["TargetMeasurement"],
    *,
    error_class: type[Exception] = ValueError,
) -> float:
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
        raise error_class("Target photometry never produced positive source counts.")
    return float(np.median(np.asarray(instrumental_mags, dtype=float)))


def _connected_components(
    present: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, list[str]]]:
    """
        Connected components of the bipartite graph whose nodes are comparison stars and frames, with
        an edge wherever a star was measured on a frame. Each component is a set of frames that a
        common internal magnitude system can tie together; frames in different components share no
        comparison stars.
    """
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for candidate_id, rows in present.items():
        star_node = ("star", candidate_id)
        for fits_path in rows:
            frame_node = ("frame", fits_path)
            adjacency[star_node].add(frame_node)
            adjacency[frame_node].add(star_node)

    visited: set[tuple[str, str]] = set()
    components: list[dict[str, list[str]]] = []
    for node in list(adjacency):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        component_stars: set[str] = set()
        component_frames: set[str] = set()
        while stack:
            current = stack.pop()
            if current[0] == "star":
                component_stars.add(current[1])
            else:
                component_frames.add(current[1])
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append({"stars": sorted(component_stars), "frames": sorted(component_frames)})
    return components


def _solve_component_zero_points(
    *,
    star_ids: Sequence[str],
    frame_paths: Sequence[str],
    present: Mapping[str, Mapping[str, tuple[float, Any]]],
    star_by_id: Mapping[str, ComparisonStar],
) -> tuple[dict[str, float], dict[str, float], dict[str, list[str]]]:
    """
        Robustly solves per-frame zero points ZP_f and internal star magnitudes M_c for one connected
        component of the (star x frame) graph, from the model m[c,f] + ZP_f = M_c. Alternates robust
        (median) updates of ZP_f and M_c, re-anchoring the absolute scale to catalog magnitudes each
        pass so the internal system stays on the catalog scale. After an initial solve it drops
        variable stars (large residual scatter) and re-solves once.

        Returns per-frame ZP_f, per-frame ZP standard error, and the active (kept) star ids per frame.
    """
    internal_mag, zero_points = _iterate_component_solve(star_ids, frame_paths, present, star_by_id)

    kept_ids = [
        candidate_id
        for candidate_id in star_ids
        if _residual_scatter(candidate_id, frame_paths, present, internal_mag, zero_points) <= MAX_ACCEPTABLE_VARIABILITY
    ]
    if kept_ids and len(kept_ids) < len(star_ids):
        star_ids = kept_ids
        internal_mag, zero_points = _iterate_component_solve(star_ids, frame_paths, present, star_by_id)

    zp_by_frame: dict[str, float] = {}
    zp_sigma_by_frame: dict[str, float] = {}
    active_ids_by_frame: dict[str, list[str]] = {}
    for fits_path in frame_paths:
        per_star_zero_points = [
            internal_mag[candidate_id] - present[candidate_id][fits_path][0]
            for candidate_id in star_ids
            if fits_path in present[candidate_id]
        ]
        active_ids = sorted(candidate_id for candidate_id in star_ids if fits_path in present[candidate_id])
        active_ids_by_frame[fits_path] = active_ids
        if per_star_zero_points:
            zp_by_frame[fits_path] = zero_points[fits_path]
            if len(per_star_zero_points) > 1:
                zp_sigma_by_frame[fits_path] = float(
                    np.std(per_star_zero_points, ddof=1) / math.sqrt(len(per_star_zero_points))
                )
            else:
                zp_sigma_by_frame[fits_path] = math.nan
        else:
            zp_by_frame[fits_path] = math.nan
            zp_sigma_by_frame[fits_path] = math.nan
    return zp_by_frame, zp_sigma_by_frame, active_ids_by_frame


def _iterate_component_solve(
    star_ids: Sequence[str],
    frame_paths: Sequence[str],
    present: Mapping[str, Mapping[str, tuple[float, Any]]],
    star_by_id: Mapping[str, ComparisonStar],
) -> tuple[dict[str, float], dict[str, float]]:
    """
        Fixed-point robust solve of the two-way model m[c,f] + ZP_f = M_c over a component, anchored
        to catalog magnitudes. Returns internal star magnitudes M_c and per-frame zero points ZP_f.
    """
    catalog_mag = {candidate_id: float(star_by_id[candidate_id].reference_magnitude) for candidate_id in star_ids}
    internal_mag = dict(catalog_mag)
    zero_points = {fits_path: 0.0 for fits_path in frame_paths}

    for _ in range(EVOLVING_SOLVE_ITERATIONS):
        max_delta = 0.0
        for fits_path in frame_paths:
            estimates = [
                internal_mag[candidate_id] - present[candidate_id][fits_path][0]
                for candidate_id in star_ids
                if fits_path in present[candidate_id]
            ]
            if estimates:
                updated = float(np.median(estimates))
                max_delta = max(max_delta, abs(updated - zero_points[fits_path]))
                zero_points[fits_path] = updated
        for candidate_id in star_ids:
            estimates = [
                present[candidate_id][fits_path][0] + zero_points[fits_path]
                for fits_path in frame_paths
                if fits_path in present[candidate_id]
            ]
            if estimates:
                updated = float(np.median(estimates))
                max_delta = max(max_delta, abs(updated - internal_mag[candidate_id]))
                internal_mag[candidate_id] = updated
        # Anchor the absolute scale: the model has a gauge freedom (add a constant to every M_c and
        # subtract it from every ZP_f), so pin it to the catalog scale each pass.
        shift = float(np.median([internal_mag[candidate_id] - catalog_mag[candidate_id] for candidate_id in star_ids]))
        for candidate_id in star_ids:
            internal_mag[candidate_id] -= shift
        for fits_path in frame_paths:
            zero_points[fits_path] += shift
        if max_delta < EVOLVING_SOLVE_TOLERANCE_MAG:
            break
    return internal_mag, zero_points


def _residual_scatter(
    candidate_id: str,
    frame_paths: Sequence[str],
    present: Mapping[str, Mapping[str, tuple[float, Any]]],
    internal_mag: Mapping[str, float],
    zero_points: Mapping[str, float],
) -> float:
    """
        Scatter of a star's per-frame residuals (m[c,f] + ZP_f) - M_c about the solved model, i.e.
        how variable it is once the frame zero points are removed. Used to reject variable stars.
    """
    residuals = [
        (present[candidate_id][fits_path][0] + zero_points[fits_path]) - internal_mag[candidate_id]
        for fits_path in frame_paths
        if fits_path in present[candidate_id]
    ]
    if len(residuals) < 2:
        return 0.0
    return float(np.std(residuals))
