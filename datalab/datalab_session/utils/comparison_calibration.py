"""Comparison-set calibration strategies for the aperture photometry light curve."""
from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
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
from datalab.datalab_session.exceptions import LightCurveError

if TYPE_CHECKING:  # avoid a runtime import cycle with aperture_light_curve
    from datalab.datalab_session.utils.aperture_light_curve import FrameContext, TargetMeasurement

log = logging.getLogger(__name__)

# Requiring presence on every frame collapses the pool: catalog detection near the limiting
# magnitude is noisy and multi-telescope sets rarely catalog the same star throughout.
SHARED_FRAME_COVERAGE_FRACTION = 0.8
# An overshoot smaller than this is not worth reporting: it says nothing about the calibration.
TARGET_EXTRAPOLATION_TOLERANCE_MAG = 0.1
# Evolving self-calibration solve controls.
EVOLVING_SOLVE_ITERATIONS = 50
EVOLVING_SOLVE_TOLERANCE_MAG = 1e-6


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


@dataclass(frozen=True)
class CalibrationInputs:
    """What a calibration strategy works from, threaded unchanged from the pipeline."""
    frames: Sequence[FrameContext]
    candidate_stars: Sequence[ComparisonStar]
    measurements_by_candidate: Mapping[str, Mapping[str, ComparisonMeasurement]]
    target_measurements: Mapping[str, TargetMeasurement]
    min_comparisons: int
    max_comparisons: int

    @property
    def frame_paths(self) -> list[str]:
        return [frame.fits_path for frame in self.frames]


@dataclass(frozen=True)
class ComparisonMatrix:
    """
        The sparse (star x frame) instrumental-magnitude matrix the evolving solve runs on, with the
        catalog magnitudes that anchor it and the target brightness that trims it. An entry exists
        only where a star was measured with positive counts.
    """
    rows: Mapping[str, Mapping[str, tuple[float, ComparisonMeasurement]]]
    star_by_id: Mapping[str, ComparisonStar]
    target_instrumental_mag_by_frame: Mapping[str, float]
    max_comparisons: int

    @classmethod
    def build(cls, inputs: "CalibrationInputs") -> "ComparisonMatrix":
        """
            Keeps only what the solve can use: positive counts on a star that has a finite catalog
            magnitude to anchor to.
        """
        star_by_id = {candidate.candidate_id: candidate for candidate in inputs.candidate_stars}
        rows: dict[str, dict[str, tuple[float, ComparisonMeasurement]]] = {}
        for candidate_id, per_frame in inputs.measurements_by_candidate.items():
            candidate = star_by_id.get(candidate_id)
            if candidate is None or not math.isfinite(candidate.reference_magnitude):
                continue
            measured = {
                fits_path: (-2.5 * math.log10(measurement.net_source_counts), measurement)
                for fits_path, measurement in per_frame.items()
                if math.isfinite(measurement.net_source_counts) and measurement.net_source_counts > 0.0
            }
            if measured:
                rows[candidate_id] = measured
        return cls(
            rows=rows,
            star_by_id=star_by_id,
            target_instrumental_mag_by_frame={
                fits_path: (
                    -2.5 * math.log10(measurement.net_source_counts)
                    if measurement.net_source_counts > 0.0
                    else math.nan
                )
                for fits_path, measurement in inputs.target_measurements.items()
            },
            max_comparisons=inputs.max_comparisons,
        )

    def __bool__(self) -> bool:
        return bool(self.rows)

    def measured_on(self, candidate_id: str, fits_path: str) -> bool:
        return fits_path in self.rows[candidate_id]

    def instrumental_mag(self, candidate_id: str, fits_path: str) -> float:
        return self.rows[candidate_id][fits_path][0]

    def measurement(self, candidate_id: str, fits_path: str) -> ComparisonMeasurement:
        return self.rows[candidate_id][fits_path][1]

    def catalog_mag(self, candidate_id: str) -> float:
        return float(self.star_by_id[candidate_id].reference_magnitude)

    def target_instrumental_mag(self, fits_path: str) -> float:
        """The target's own instrumental magnitude, NaN where it had no positive counts."""
        return self.target_instrumental_mag_by_frame.get(fits_path, math.nan)


@dataclass(frozen=True)
class EvolvingSolution:
    """
        The evolving strategy's solve, merged across every connected component: each frame's zero
        point, its standard error, and the comparison stars that frame calibrates against.

        populated_components is how many groups the field split into; more than one means parts of
        the series share no stars and were anchored separately, so steps between them are possible.
    """
    zero_point_by_frame: dict[str, float]
    zp_sigma_by_frame: dict[str, float]
    active_ids_by_frame: dict[str, list[str]]
    populated_components: int


@dataclass(frozen=True)
class CalibrationOutcome:
    """What a comparison strategy produces: the per-frame calibration, the stars it drew on, and
    anything the user should know about how it went."""
    frame_calibrations: dict[str, FrameCalibration]
    used_stars: tuple[ComparisonStar, ...]
    diagnostics: list[str]


class ComparisonStrategy(ABC):
    """
        How the comparison ensemble is maintained across the series. A strategy also states what it
        needs from the measurement pass, since that follows from how it uses the candidates.
    """

    # Fraction of frames a candidate must be catalogued on to enter the pool at all, and whether a
    # candidate that fails to measure on one frame is abandoned for the rest.
    min_frame_coverage: float
    drops_failed_candidates: bool

    @abstractmethod
    def calibrate(self, inputs: CalibrationInputs) -> CalibrationOutcome:
        """Calibrate every frame, or raise LightCurveError if this strategy cannot."""


@dataclass(frozen=True)
class SharedEnsemble(ComparisonStrategy):
    """One ensemble present on every frame: the sidereal default."""

    min_frame_coverage = SHARED_FRAME_COVERAGE_FRACTION
    drops_failed_candidates = True

    def calibrate(self, inputs: CalibrationInputs) -> CalibrationOutcome:
        return _calibrate_shared(inputs)


@dataclass(frozen=True)
class EvolvingEnsemble(ComparisonStrategy):
    """
        A per-frame ensemble whose zero points are solved jointly over the sparse star x frame matrix
        and anchored to catalog magnitudes, so a turned-over field still yields one light curve.
    """

    min_frame_coverage = 0.0
    drops_failed_candidates = False

    def calibrate(self, inputs: CalibrationInputs) -> CalibrationOutcome:
        return _calibrate_evolving(inputs)


@dataclass(frozen=True)
class SharedThenEvolving(ComparisonStrategy):
    """
        Try the shared ensemble, fall back to the evolving one when none spans the series. Takes the
        evolving measurement requirements throughout, since the need is not known until after
        measurement.
    """

    min_frame_coverage = 0.0
    drops_failed_candidates = False

    def calibrate(self, inputs: CalibrationInputs) -> CalibrationOutcome:
        try:
            return _calibrate_shared(inputs)
        except LightCurveError as exc:
            log.warning(
                "Aperture Photometry shared comparison ensemble unavailable; "
                f"falling back to evolving calibration: {exc}"
            )
            outcome = _calibrate_evolving(inputs)
            return replace(
                outcome,
                diagnostics=[
                    f"No comparison ensemble spans every frame ({exc}); used evolving per-frame "
                    "calibration."
                ] + list(outcome.diagnostics),
            )


def calibrate(
    inputs: CalibrationInputs,
    strategy: ComparisonStrategy,
) -> CalibrationOutcome:
    """
        Calibrates the light curve with the given strategy, then widens each point's uncertainty by
        an empirical error floor, which every strategy earns the same way.
    """
    outcome = strategy.calibrate(inputs)
    frame_calibrations, diagnostics = outcome.frame_calibrations, outcome.diagnostics

    # The formal per-point uncertainty carries only photon + zero-point-scatter terms and badly
    # underestimates real frame-to-frame reproducibility (flat-field, scintillation, catalog noise).
    # Measure that systematic floor directly from how well the comparison stars' own calibrated
    # magnitudes repeat frame to frame, and add it in quadrature to every point.
    floor = _empirical_error_floor(frame_calibrations, inputs.frame_paths)
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

    extrapolation = _extrapolated_target_diagnostic(frame_calibrations)
    if extrapolation is not None:
        diagnostics = list(diagnostics) + [extrapolation]
    return replace(outcome, frame_calibrations=frame_calibrations, diagnostics=diagnostics)


def _extrapolated_target_diagnostic(frame_calibrations: Mapping[str, FrameCalibration]) -> str | None:
    """
        Says so when the target falls outside the comparison stars' magnitude range, where its
        calibrated magnitude is extrapolated from the zero point rather than interpolated within it.
    """
    target_mags = [
        calibration.calibrated_mag
        for calibration in frame_calibrations.values()
        if math.isfinite(calibration.calibrated_mag)
    ]
    reference_mags = [
        star.reference_magnitude
        for calibration in frame_calibrations.values()
        for star in calibration.stars
        if math.isfinite(star.reference_magnitude)
    ]
    if not target_mags or not reference_mags:
        return None

    target_mag = float(np.median(target_mags))
    brightest, faintest = min(reference_mags), max(reference_mags)
    if target_mag < brightest - TARGET_EXTRAPOLATION_TOLERANCE_MAG:
        offset, direction = brightest - target_mag, "brighter"
    elif target_mag > faintest + TARGET_EXTRAPOLATION_TOLERANCE_MAG:
        offset, direction = target_mag - faintest, "fainter"
    else:
        return None
    return (
        f"The target ({target_mag:.2f} mag) is {direction} than every comparison star: the ensemble "
        f"spans {brightest:.2f} to {faintest:.2f} mag, so the target's magnitude is extrapolated "
        f"{offset:.2f} mag beyond it."
    )


def _calibrate_shared(inputs: CalibrationInputs) -> CalibrationOutcome:
    """
        Sidereal calibration: one comparison ensemble present on every frame, calibrated by a
        per-frame catalog-anchored zero point. Each frame's zero point is
        frame_zero_point = 2.5*log10(measured ensemble counts) + ensemble catalog magnitude, so the
        calibration self-corrects that frame's airmass/instrument/exposure. Unchanged behavior from
        before the strategy split.
    """
    target_mag_proxy = _target_magnitude_proxy(inputs.target_measurements.values())
    log.info(f"Aperture Photometry target magnitude proxy: {target_mag_proxy:.6f}")
    selection = select_comparison_stars(
        frames=inputs.frames,
        candidates=inputs.candidate_stars,
        measurements_by_candidate=inputs.measurements_by_candidate,
        target_mag_proxy=target_mag_proxy,
        min_comparisons=inputs.min_comparisons,
        max_comparisons=inputs.max_comparisons,
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

    frame_calibrations: dict[str, FrameCalibration] = {}
    for frame in inputs.frames:
        target = inputs.target_measurements[frame.fits_path]
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
    return CalibrationOutcome(frame_calibrations, tuple(selection.selected_stars), [])


def _calibrate_evolving(inputs: CalibrationInputs) -> CalibrationOutcome:
    """
        Evolving calibration for a drifting non-sidereal field, where no single ensemble spans the
        series. Comparison stars link the frames they share, so a star entering the field is tied to
        the magnitude system the incumbents already define and membership can turn over completely
        without stepping the light curve. Disconnected groups (a fully turned-over field, or a time
        gap with no shared stars) are anchored to catalog magnitudes separately and flagged, since
        nothing ties them steplessly.

        Every linking star feeds the solve; only each frame's reported ensemble is capped.
    """
    matrix = ComparisonMatrix.build(inputs)
    if not matrix:
        raise LightCurveError("Evolving calibration found no usable comparison-star measurements.")

    solution = _solve_evolving_zero_points(matrix, inputs.frame_paths)

    diagnostics: list[str] = []
    if solution.populated_components > 1:
        diagnostics.append(
            f"Evolving calibration: the comparison field splits into {solution.populated_components} "
            "groups sharing no common stars, each anchored to catalog magnitudes independently; "
            "magnitude steps may exist between them."
        )

    frame_calibrations: dict[str, FrameCalibration] = {}
    for fits_path in inputs.frame_paths:
        calibration, shortfall = _evolving_frame_calibration(
            matrix,
            fits_path=fits_path,
            target=inputs.target_measurements[fits_path],
            active_ids=solution.active_ids_by_frame.get(fits_path, []),
            zero_point=solution.zero_point_by_frame.get(fits_path, math.nan),
            zp_sigma=solution.zp_sigma_by_frame.get(fits_path, math.nan),
            min_comparisons=inputs.min_comparisons,
        )
        frame_calibrations[fits_path] = calibration
        if shortfall is not None:
            diagnostics.append(shortfall)

    used_ids = {candidate_id for ids in solution.active_ids_by_frame.values() for candidate_id in ids}
    used_stars = tuple(matrix.star_by_id[candidate_id] for candidate_id in sorted(used_ids))
    if not used_stars:
        raise LightCurveError("Evolving calibration produced no usable comparison stars.")
    log.info(
        "Aperture Photometry evolving calibration: "
        f"components={solution.populated_components}, used_comparison_stars={len(used_stars)}"
    )
    return CalibrationOutcome(frame_calibrations, used_stars, diagnostics)


def _solve_evolving_zero_points(
    matrix: ComparisonMatrix,
    frame_paths: Sequence[str],
) -> EvolvingSolution:
    """
        Solves each connected component of the (star x frame) graph and merges the results.

        Components share no comparison stars, so nothing links their magnitude systems and each is
        solved on its own against catalog magnitudes.
    """
    components = _connected_components(matrix.rows)
    zero_point_by_frame: dict[str, float] = {}
    zp_sigma_by_frame: dict[str, float] = {}
    active_ids_by_frame: dict[str, list[str]] = {fits_path: [] for fits_path in frame_paths}
    for component in components:
        zero_points, zp_sigmas, active_ids = _solve_component_zero_points(
            matrix, star_ids=component["stars"], frame_paths=component["frames"]
        )
        zero_point_by_frame.update(zero_points)
        zp_sigma_by_frame.update(zp_sigmas)
        active_ids_by_frame.update(active_ids)
    return EvolvingSolution(
        zero_point_by_frame=zero_point_by_frame,
        zp_sigma_by_frame=zp_sigma_by_frame,
        active_ids_by_frame=active_ids_by_frame,
        populated_components=sum(1 for component in components if component["frames"]),
    )


def _evolving_frame_calibration(
    matrix: ComparisonMatrix,
    *,
    fits_path: str,
    target: TargetMeasurement,
    active_ids: Sequence[str],
    zero_point: float,
    zp_sigma: float,
    min_comparisons: int,
) -> tuple[FrameCalibration, str | None]:
    """
        One frame's calibration from its solved zero point and ensemble.

        Returns the calibration, plus a diagnostic when the frame has too few comparison stars to
        calibrate at all (its magnitude is NaN and the light-curve row is omitted downstream).
    """
    measurements = tuple(matrix.measurement(candidate_id, fits_path) for candidate_id in active_ids)
    stars = tuple(matrix.star_by_id[candidate_id] for candidate_id in active_ids)
    counts = np.asarray([measurement.net_source_counts for measurement in measurements], dtype=float)
    uncertainties = np.asarray([measurement.source_uncertainty for measurement in measurements], dtype=float)
    ensemble_flux = float(np.sum(counts)) if counts.size else 0.0
    ensemble_variance = float(np.sum(np.square(uncertainties))) if uncertainties.size else 0.0
    target_rel_flux, target_rel_flux_sigma = relative_flux(target, ensemble_flux, ensemble_variance)

    shortfall: str | None = None
    if len(active_ids) >= min_comparisons and math.isfinite(zero_point) and target.net_source_counts > 0.0:
        calibrated_mag = -2.5 * math.log10(target.net_source_counts) + zero_point
        _, target_inst_sigma = flux_to_mag(target.net_source_counts, target.source_uncertainty)
        calibrated_mag_sigma = math.hypot(target_inst_sigma, zp_sigma if math.isfinite(zp_sigma) else 0.0)
    else:
        calibrated_mag = math.nan
        calibrated_mag_sigma = math.nan
        if len(active_ids) < min_comparisons:
            shortfall = (
                f"Evolving calibration: {os.path.basename(fits_path)} has {len(active_ids)} usable "
                f"comparison stars (< {min_comparisons}); light-curve row omitted."
            )
    calibration = FrameCalibration(
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
    return calibration, shortfall


def relative_flux(target: TargetMeasurement, ensemble_flux: float, ensemble_variance: float) -> tuple[float, float]:
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


def _connected_components(
    matrix_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, list[str]]]:
    """
        Connected components of the bipartite graph whose nodes are comparison stars and frames, with
        an edge wherever a star was measured on a frame. Each component is a set of frames that a
        common internal magnitude system can tie together; frames in different components share no
        comparison stars.

        Takes the raw rows rather than the ComparisonMatrix: this is graph connectivity, and none of
        the magnitudes or the ensemble cap bear on it.
    """
    adjacency: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for candidate_id, frames_for_star in matrix_rows.items():
        star_node = ("star", candidate_id)
        for fits_path in frames_for_star:
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
    matrix: ComparisonMatrix,
    *,
    star_ids: Sequence[str],
    frame_paths: Sequence[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, list[str]]]:
    """
        Solves per-frame zero points ZP_f and internal star magnitudes M_c for one connected component
        of the (star x frame) graph, from the model m[c,f] + ZP_f = M_c, then drops variable stars
        (large residual scatter) and re-solves once.

        ZP_f is the median over the frame's capped ensemble, so the zero point and its standard error
        describe the same stars. Returns ZP_f, its standard error, and the kept star ids per frame.
    """
    internal_mag, zero_points = _iterate_component_solve(matrix, star_ids, frame_paths)

    kept_ids = [
        candidate_id
        for candidate_id in star_ids
        if _residual_scatter(matrix, candidate_id, frame_paths, internal_mag, zero_points) <= MAX_ACCEPTABLE_VARIABILITY
    ]
    if kept_ids and len(kept_ids) < len(star_ids):
        star_ids = kept_ids
        internal_mag, zero_points = _iterate_component_solve(matrix, star_ids, frame_paths)

    zp_by_frame: dict[str, float] = {}
    zp_sigma_by_frame: dict[str, float] = {}
    active_ids_by_frame: dict[str, list[str]] = {}
    for fits_path in frame_paths:
        active_ids = _capped_frame_comparisons(
            matrix,
            candidate_ids=[c for c in star_ids if matrix.measured_on(c, fits_path)],
            fits_path=fits_path,
        )
        per_star_zero_points = [
            internal_mag[candidate_id] - matrix.instrumental_mag(candidate_id, fits_path)
            for candidate_id in active_ids
        ]
        active_ids_by_frame[fits_path] = active_ids
        if per_star_zero_points:
            zp_by_frame[fits_path] = float(np.median(per_star_zero_points))
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


def _capped_frame_comparisons(
    matrix: ComparisonMatrix,
    *,
    candidate_ids: Sequence[str],
    fits_path: str,
) -> list[str]:
    """
        Trims one frame's ensemble to the user's maximum, ranked by closeness in measured brightness
        to the target: the criterion _source_catalog_sort_key uses, valid here for the same reason
        that both magnitudes are instrumental and from this pipeline. Where the target has no
        positive counts the brightest are kept instead. Returned sorted by candidate id.
    """
    if len(candidate_ids) <= matrix.max_comparisons:
        return sorted(candidate_ids)
    target_mag = matrix.target_instrumental_mag(fits_path)

    def rank(candidate_id: str) -> tuple[float, str]:
        instrumental_mag = matrix.instrumental_mag(candidate_id, fits_path)
        if math.isfinite(target_mag):
            return abs(instrumental_mag - target_mag), candidate_id
        return instrumental_mag, candidate_id

    return sorted(sorted(candidate_ids, key=rank)[:matrix.max_comparisons])


def _iterate_component_solve(
    matrix: ComparisonMatrix,
    star_ids: Sequence[str],
    frame_paths: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    """
        Fixed-point robust solve of the two-way model m[c,f] + ZP_f = M_c over a component, anchored
        to catalog magnitudes. Returns internal star magnitudes M_c and per-frame zero points ZP_f.
    """
    catalog_mag = {candidate_id: matrix.catalog_mag(candidate_id) for candidate_id in star_ids}
    internal_mag = dict(catalog_mag)
    zero_points = {fits_path: 0.0 for fits_path in frame_paths}
    # The sparsity pattern is fixed for the life of the solve, so index it once in both directions.
    # Re-deriving it per iteration walks the full star x frame product, which on a drifted field is
    # mostly absent entries: 50 iterations x 200 stars x 999 frames of membership tests.
    frames_of_star = {
        candidate_id: [
            (fits_path, matrix.instrumental_mag(candidate_id, fits_path))
            for fits_path in frame_paths
            if matrix.measured_on(candidate_id, fits_path)
        ]
        for candidate_id in star_ids
    }
    stars_on_frame: dict[str, list[tuple[str, float]]] = {fits_path: [] for fits_path in frame_paths}
    for candidate_id in star_ids:
        for fits_path, instrumental_mag in frames_of_star[candidate_id]:
            stars_on_frame[fits_path].append((candidate_id, instrumental_mag))

    for _ in range(EVOLVING_SOLVE_ITERATIONS):
        max_delta = 0.0
        for fits_path in frame_paths:
            estimates = [
                internal_mag[candidate_id] - instrumental_mag
                for candidate_id, instrumental_mag in stars_on_frame[fits_path]
            ]
            if estimates:
                updated = float(np.median(estimates))
                max_delta = max(max_delta, abs(updated - zero_points[fits_path]))
                zero_points[fits_path] = updated
        for candidate_id in star_ids:
            estimates = [
                instrumental_mag + zero_points[fits_path]
                for fits_path, instrumental_mag in frames_of_star[candidate_id]
            ]
            if estimates:
                updated = float(np.median(estimates))
                max_delta = max(max_delta, abs(updated - internal_mag[candidate_id]))
                internal_mag[candidate_id] = updated
        # Anchor the absolute scale: the model m + ZP = M has a gauge freedom (add a constant to
        # every M_c and to every ZP_f), so pin it to the catalog scale each pass. Both shift by the
        # same signed amount, since ZP_f = M_c - m[c,f].
        shift = float(np.median([internal_mag[candidate_id] - catalog_mag[candidate_id] for candidate_id in star_ids]))
        for candidate_id in star_ids:
            internal_mag[candidate_id] -= shift
        for fits_path in frame_paths:
            zero_points[fits_path] -= shift
        if max_delta < EVOLVING_SOLVE_TOLERANCE_MAG:
            break
    return internal_mag, zero_points


def _residual_scatter(
    matrix: ComparisonMatrix,
    candidate_id: str,
    frame_paths: Sequence[str],
    internal_mag: Mapping[str, float],
    zero_points: Mapping[str, float],
) -> float:
    """
        Scatter of a star's per-frame residuals (m[c,f] + ZP_f) - M_c about the solved model, i.e.
        how variable it is once the frame zero points are removed. Used to reject variable stars.
    """
    residuals = [
        (matrix.instrumental_mag(candidate_id, fits_path) + zero_points[fits_path]) - internal_mag[candidate_id]
        for fits_path in frame_paths
        if matrix.measured_on(candidate_id, fits_path)
    ]
    if len(residuals) < 2:
        return 0.0
    return float(np.std(residuals))
