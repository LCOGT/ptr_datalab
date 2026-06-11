from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from datalab.datalab_session.utils.centroiding import centroid
from datalab.datalab_session.utils.fits_metadata import header_float, world_to_pixel
from datalab.datalab_session.utils.photometry import measure_aperture


DEFAULT_GAIN = 1.0
DEFAULT_READ_NOISE = 0.0
AIJ_COMP_BRIGHTNESS_TO_DISTANCE_WEIGHT = 50.0
AIJ_COMP_UPPER_BRIGHTNESS_PERCENT = 150.0
AIJ_COMP_LOWER_BRIGHTNESS_PERCENT = 50.0


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


@dataclass(frozen=True)
class ComparisonSelectionResult:
    selected_stars: tuple[ComparisonStar, ...]
    diagnostics: tuple[str, ...]


def select_comparison_stars(
    *,
    frames: Sequence[Any],
    catalog: Sequence[dict[str, Any]],
    target_catalog_flux: float | None,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    min_comparisons: int,
    max_comparisons: int,
    error_class: type[Exception] = ValueError,
) -> ComparisonSelectionResult:
    selected = _select_by_source_catalog(
        frames=frames,
        catalog=catalog,
        target_catalog_flux=target_catalog_flux,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        max_comparisons=max_comparisons,
        error_class=error_class,
    )

    if len(selected) < min_comparisons:
        raise error_class("Source-catalog variability strategy failed to yield the minimum comparison ensemble.")

    return ComparisonSelectionResult(
        selected_stars=tuple(selected[:max_comparisons]),
        diagnostics=tuple(),
    )

## measure_candidate_on_frame does the following:
# 1. It converts the candidate's celestial coordinates (RA, Dec) to pixel coordinates in the image using the World Coordinate System (WCS) information from the image header.
# 2. It performs centroiding around the initial pixel coordinates to refine the star's position, which helps to account for any inaccuracies in the WCS or the candidate's coordinates.
# 3. It calls the measure_aperture function to perform aperture photometry at the refined position, which includes estimating the background level, calculating the total flux within the aperture, and computing the net source counts and associated uncertainties.

def measure_candidate_on_frame(
    *,
    frame: Any,
    candidate: ComparisonStar,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    error_class: type[Exception] = ValueError,
) -> ComparisonMeasurement:
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


def astroimagej_comparison_weight(
    candidate: ComparisonStar,
    target_catalog_flux: float | None,
    image_diagonal_px: float,
) -> float:
    brightness_weight = AIJ_COMP_BRIGHTNESS_TO_DISTANCE_WEIGHT / 100.0
    distance_weight = 1.0 - brightness_weight
    norm_brightness = _astroimagej_normalized_brightness(candidate, target_catalog_flux)
    norm_distance = 1.0
    if math.isfinite(image_diagonal_px) and image_diagonal_px > 0.0:
        norm_distance = 1.0 - (candidate.target_separation_px / image_diagonal_px)
    return brightness_weight * norm_brightness + distance_weight * norm_distance


_astroimagej_comparison_weight = astroimagej_comparison_weight


def _select_by_source_catalog(
    *,
    frames: Sequence[Any],
    catalog: Sequence[dict[str, Any]],
    target_catalog_flux: float | None,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    max_comparisons: int,
    error_class: type[Exception],
) -> list[ComparisonStar]:
    enriched = _measure_and_rank_candidates(
        frames=frames,
        catalog=catalog,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        error_class=error_class,
    )
    return sorted(
        enriched,
        key=lambda candidate: _source_catalog_sort_key(candidate, target_catalog_flux, frames),
    )[:max_comparisons]


def _source_catalog_sort_key(
    candidate: ComparisonStar,
    target_catalog_flux: float | None,
    frames: Sequence[Any],
) -> tuple[float, str]:
    return (
        -astroimagej_comparison_weight(candidate, target_catalog_flux, _image_diagonal_px(frames)),
        candidate.candidate_id,
    )


def _astroimagej_normalized_brightness(candidate: ComparisonStar, target_catalog_flux: float | None) -> float:
    candidate_flux = _candidate_catalog_flux(candidate)
    if (
        target_catalog_flux is None
        or not math.isfinite(target_catalog_flux)
        or target_catalog_flux <= 0.0
        or not math.isfinite(candidate_flux)
        or candidate_flux <= 0.0
    ):
        return 0.0

    if candidate_flux <= target_catalog_flux:
        return 1.0 - (
            (target_catalog_flux - candidate_flux)
            / (target_catalog_flux * (1.0 - (AIJ_COMP_LOWER_BRIGHTNESS_PERCENT / 100.0)))
        )
    return 1.0 - (
        (candidate_flux - target_catalog_flux)
        / (target_catalog_flux * ((AIJ_COMP_UPPER_BRIGHTNESS_PERCENT / 100.0) - 1.0))
    )


def _candidate_catalog_flux(candidate: ComparisonStar) -> float:
    candidate_fluxes = [
        _optional_float(row.get("flux"))
        for row in candidate.source_catalog_by_frame.values()
    ]
    candidate_fluxes = [flux for flux in candidate_fluxes if math.isfinite(flux) and flux > 0.0]
    if not candidate_fluxes:
        return math.nan
    return float(np.median(np.asarray(candidate_fluxes, dtype=float)))


def _image_diagonal_px(frames: Sequence[Any]) -> float:
    if not frames:
        return math.nan
    frame = frames[0]
    return math.sqrt((frame.width * frame.width) + (frame.height * frame.height))


def _measure_and_rank_candidates(
    *,
    frames: Sequence[Any],
    catalog: Sequence[dict[str, Any]],
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    error_class: type[Exception],
) -> list[ComparisonStar]:
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
    for (candidate, candidate_star, _instrumental_mags), variability_mags in zip(
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
            )
        )
    return selected


def _optional_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan
