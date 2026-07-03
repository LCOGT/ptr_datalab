from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from datalab.datalab_session.utils.centroiding import centroid
from datalab.datalab_session.utils.fits_metadata import aperture_unit_scale, header_float, world_to_pixel
from datalab.datalab_session.utils.photometry import measure_aperture


DEFAULT_GAIN = 1.0
DEFAULT_READ_NOISE = 0.0
# A candidate whose frame-to-frame instrumental magnitude scatter exceeds this (mag) is variable
# and unfit as a photometric reference, so it is excluded from the comparison ensemble.
MAX_ACCEPTABLE_VARIABILITY = 1.0
# A good comparison star's (catalog_mag - measured_instrumental_mag) equals the frame ensemble's
# zero point, common to all such stars. A candidate whose residual departs from the ensemble median
# by more than this many magnitudes has an untrustworthy catalog magnitude (usually a blended or
# mismatched cross-match) and would bias the zero-point calibration, so it is dropped.
MAX_ZERO_POINT_RESIDUAL_MAG = 0.5


@dataclass(frozen=True)
class ComparisonMeasurement:
    candidate_id: str
    fits_path: str
    x: float
    y: float
    net_source_counts: float
    source_uncertainty: float
    mean_background_per_pixel: float
    peak_pixel_value: float
    effective_source_pixels: float
    effective_background_pixels: float


@dataclass(frozen=True)
class ComparisonStar:
    candidate_id: str
    ra_deg: float
    dec_deg: float
    reference_magnitude: float
    reference_magnitude_source: str
    source_catalog_by_frame: Mapping[str, Mapping[str, Any]]
    variability_score: float
    isolation_px: float
    target_separation_px: float
    measured_instrumental_magnitude: float = math.inf


@dataclass(frozen=True)
class ComparisonSelectionResult:
    selected_stars: tuple[ComparisonStar, ...]
    diagnostics: tuple[str, ...]


def select_comparison_stars(
    *,
    frames: Sequence[Any],
    catalog: Sequence[dict[str, Any]],
    target_mag_proxy: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    min_comparisons: int,
    max_comparisons: int,
    aperture_unit: str = "px",
    error_class: type[Exception] = ValueError,
) -> ComparisonSelectionResult:
    """
        Selects comparison stars from source catalog candidates.

        Measures every candidate, drops variable stars and zero-point-inconsistent (blended or
        mismatched) catalog matches, then ranks the rest by how close their measured brightness is
        to the target's. Returns the comparison ensemble for calibration.
    """
    enriched = _measure_and_rank_candidates(
        frames=frames,
        catalog=catalog,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        aperture_unit=aperture_unit,
        error_class=error_class,
    )
    stable = [candidate for candidate in enriched if candidate.variability_score <= MAX_ACCEPTABLE_VARIABILITY]
    consistent = _reject_zero_point_outliers(stable)
    ranked = sorted(
        consistent,
        key=lambda candidate: _source_catalog_sort_key(candidate, target_mag_proxy),
    )

    if len(ranked) < min_comparisons:
        raise error_class("Source-catalog variability strategy failed to yield the minimum comparison ensemble.")

    return ComparisonSelectionResult(
        selected_stars=tuple(ranked[:max_comparisons]),
        diagnostics=tuple(),
    )


def _reject_zero_point_outliers(candidates: Sequence[ComparisonStar]) -> list[ComparisonStar]:
    """
        Drops candidates whose (catalog_mag - measured_instrumental_mag) residual departs from the
        ensemble median by more than MAX_ZERO_POINT_RESIDUAL_MAG -- these carry an untrustworthy
        catalog magnitude (typically a blended or mismatched cross-match) that would bias the
        zero-point calibration. Needs a few stars for a robust median; below that, keeps all, and
        never lets the guard empty the pool.
    """
    if len(candidates) < 3:
        return list(candidates)
    residuals = np.asarray(
        [candidate.reference_magnitude - candidate.measured_instrumental_magnitude for candidate in candidates],
        dtype=float,
    )
    median_residual = float(np.median(residuals))
    kept = [
        candidate
        for candidate, residual in zip(candidates, residuals)
        if abs(residual - median_residual) <= MAX_ZERO_POINT_RESIDUAL_MAG
    ]
    return kept if kept else list(candidates)


def _source_catalog_sort_key(candidate: ComparisonStar, target_mag_proxy: float) -> tuple[float, float, str]:
    """
        Ranks by closeness in brightness to the target. Both sides are instrumental magnitudes
        measured by this pipeline (median over frames), so they share a zero point; comparing the
        catalog's calibrated reference_magnitude against the instrumental target_mag_proxy would
        instead compare across a ~zero-point offset and collapse to "pick the brightest catalog star".
    """
    return (
        abs(candidate.measured_instrumental_magnitude - target_mag_proxy),
        -candidate.isolation_px,
        candidate.candidate_id,
    )

def measure_candidate_on_frame(
    *,
    frame: Any,
    candidate: ComparisonStar,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    aperture_unit: str = "px",
    error_class: type[Exception] = ValueError,
) -> ComparisonMeasurement:
    """
        Measures aperture photometry for one comparison-star candidate on a single FITS frame.

        Converts the candidate's RA/Dec to pixel coordinates via the frame WCS, centroids around
        that position to refine it (correcting small WCS or catalog inaccuracies), then measures
        aperture photometry at the refined position -- estimating the background, summing the
        aperture flux, and computing the net source counts and their uncertainty.

        Returns the comparison-star measurement for this frame.
    """
    scale = aperture_unit_scale(frame.header, aperture_unit)
    aperture_radius_px = aperture_radius_px / scale
    annulus_inner_radius_px = annulus_inner_radius_px / scale
    annulus_outer_radius_px = annulus_outer_radius_px / scale
    x, y = world_to_pixel(frame.header, candidate.ra_deg, candidate.dec_deg)
    centroid_result = centroid(
        image=frame.image,
        x_click=x,
        y_click=y,
        radius=aperture_radius_px,
        r_back1=annulus_inner_radius_px,
        r_back2=annulus_outer_radius_px,
    )
    if not centroid_result.success:
        raise error_class(f"Selected comparison-star centroiding failed for {frame.fits_path}, {candidate.candidate_id}.")
    photometry = measure_aperture(
        image=frame.image,
        x_center=centroid_result.x,
        y_center=centroid_result.y,
        aperture_radius_px=aperture_radius_px,
        background_model=centroid_result.background_model,
        gain=header_float(frame.header, ("GAIN", "EGAIN"), DEFAULT_GAIN),
        read_noise=header_float(frame.header, ("RDNOISE", "READNOIS", "READNOISE"), DEFAULT_READ_NOISE),
        dark=0.0,
        error_class=error_class,
    )
    return ComparisonMeasurement(
        candidate_id=candidate.candidate_id,
        fits_path=frame.fits_path,
        x=centroid_result.x,
        y=centroid_result.y,
        net_source_counts=photometry["net_source_counts"],
        source_uncertainty=photometry["source_uncertainty"],
        mean_background_per_pixel=photometry["mean_background_per_pixel"],
        peak_pixel_value=photometry["peak_pixel_value"],
        effective_source_pixels=photometry["effective_source_pixels"],
        effective_background_pixels=photometry["effective_background_pixels"],
    )


def _measure_and_rank_candidates(
    *,
    frames: Sequence[Any],
    catalog: Sequence[dict[str, Any]],
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    aperture_unit: str = "px",
    error_class: type[Exception],
) -> list[ComparisonStar]:
    """
        Measures each candidate across all frames and calcaltes variability scores.

        Returns comp star candidates that have valid positive measurements across all frames.
    """
    measured_candidates: list[tuple[dict[str, Any], ComparisonStar, np.ndarray]] = []
    for candidate in sorted(catalog, key=lambda row: row["candidate_id"]):
        reference_magnitude = float(candidate.get("reference_magnitude", candidate["second_hdu_magnitude"]))
        reference_magnitude_source = str(candidate.get("reference_magnitude_source", "second_hdu"))
        source_catalog_by_frame = candidate.get("source_catalog_by_frame", {})
        candidate_star = ComparisonStar(
            candidate_id=candidate["candidate_id"],
            ra_deg=candidate["ra_deg"],
            dec_deg=candidate["dec_deg"],
            reference_magnitude=reference_magnitude,
            reference_magnitude_source=reference_magnitude_source,
            source_catalog_by_frame=source_catalog_by_frame,
            variability_score=math.inf,
            isolation_px=candidate["isolation_px"],
            target_separation_px=candidate["target_separation_px"],
        )
        try:
            per_frame = [
                measure_candidate_on_frame(
                    frame=frame,
                    candidate=candidate_star,
                    aperture_radius_px=aperture_radius_px,
                    annulus_inner_radius_px=annulus_inner_radius_px,
                    annulus_outer_radius_px=annulus_outer_radius_px,
                    aperture_unit=aperture_unit,
                    error_class=error_class,
                )
                for frame in frames
            ]
        except error_class:
            continue
        counts = np.asarray([measurement.net_source_counts for measurement in per_frame], dtype=float)
        if np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
            continue
        instrumental_mags = -2.5 * np.log10(counts)
        measured_candidates.append((candidate, candidate_star, instrumental_mags))

    if not measured_candidates:
        return []

    instrumental_mag_matrix = np.vstack([row[2] for row in measured_candidates])
    if len(measured_candidates) > 1:
        frame_offsets = np.median(instrumental_mag_matrix, axis=0)
        variability_mag_matrix = instrumental_mag_matrix - frame_offsets
    else:
        variability_mag_matrix = instrumental_mag_matrix

    selected: list[ComparisonStar] = []
    for (candidate, candidate_star, instrumental_mags), variability_mags in zip(
        measured_candidates,
        variability_mag_matrix,
    ):
        selected.append(
            ComparisonStar(
                candidate_id=candidate["candidate_id"],
                ra_deg=candidate["ra_deg"],
                dec_deg=candidate["dec_deg"],
                reference_magnitude=candidate_star.reference_magnitude,
                reference_magnitude_source=candidate_star.reference_magnitude_source,
                source_catalog_by_frame=candidate_star.source_catalog_by_frame,
                variability_score=float(np.std(variability_mags)),
                isolation_px=candidate_star.isolation_px,
                target_separation_px=candidate_star.target_separation_px,
                measured_instrumental_magnitude=float(np.median(instrumental_mags)),
            )
        )
    return selected
