from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


RA_KEYS = ("ra_deg", "ra", "RA_DEG", "RA")
DEC_KEYS = ("dec_deg", "dec", "DEC_DEG", "DEC")
MAG_KEYS = ("mag", "magnitude", "mag_estimate", "MAG", "MAG_EST")
SATURATION_KEYS = ("saturated", "is_saturated", "SATURATED")
USABLE_KEYS = ("usable", "is_usable")
BAD_FLAG_KEYS = ("flagged_bad", "FLAGGED_BAD")
EDGE_MARGIN_PX = 2.0
TARGET_PROXIMITY_FACTOR = 2.0
NEIGHBOR_EXCLUSION_FACTOR = 1.2
DEFAULT_CROSSMATCH_ARCSEC = 0.1
DEFAULT_AAVSO_MATCH_ARCSEC = 0.5
CENTROID_TOLERANCE_PX = 0.01
DEFAULT_GAIN = 1.0
DEFAULT_READ_NOISE = 0.0
MAX_ACCEPTABLE_VARIABILITY = 0.05


class LightCurveError(ValueError):
    pass


@dataclass(frozen=True)
class FrameContext:
    fits_path: str
    date_obs: datetime
    header: Mapping[str, Any]
    image: np.ndarray
    second_hdu_rows: tuple[Mapping[str, Any], ...]
    width: int
    height: int


@dataclass(frozen=True)
class TargetMeasurement:
    x: float
    y: float
    net_source_counts: float
    source_uncertainty: float
    mean_background_per_pixel: float
    peak_pixel_value: float
    effective_source_pixels: float
    effective_background_pixels: float


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
class ComparisonCandidate:
    candidate_id: str
    ra_deg: float
    dec_deg: float
    second_hdu_magnitude: float
    frame_coverage: int
    isolation_px: float
    target_separation_px: float
    variability_score: float | None = None
    mean_instrumental_magnitude: float | None = None
    matched_aavso: bool = False


@dataclass(frozen=True)
class ComparisonStar:
    candidate_id: str
    ra_deg: float
    dec_deg: float
    reference_magnitude: float
    variability_score: float
    isolation_px: float
    target_separation_px: float
    matched_aavso: bool


@dataclass(frozen=True)
class ComparisonSelectionResult:
    strategy_used: str
    fallback_used: bool
    selected_stars: tuple[ComparisonStar, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class FrameResult:
    fits_path: str
    date_obs: datetime
    target_measurement: TargetMeasurement
    comparison_measurements: tuple[ComparisonMeasurement, ...]


@dataclass(frozen=True)
class LightCurveRow:
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
    frames: list[FrameResult]
    selected_comparison_stars: list[ComparisonStar]
    light_curve_rows: list[LightCurveRow]
    diagnostics: list[str]


@dataclass(frozen=True)
class BackendDependencies:
    load_image_data: Callable[[str], np.ndarray]
    load_primary_header: Callable[[str], Mapping[str, Any]]
    load_second_hdu_rows: Callable[[str], Sequence[Mapping[str, Any]]]
    world_to_pixel: Callable[[Mapping[str, Any], float, float], tuple[float, float]]
    pixel_to_world: Callable[[Mapping[str, Any], float, float], tuple[float, float]]
    get_dark_contribution: Callable[[str, Mapping[str, Any]], float]
    get_gain: Callable[[str, Mapping[str, Any]], float]
    get_read_noise: Callable[[str, Mapping[str, Any]], float]
    query_aavso: Callable[[float, float, float], Sequence[Mapping[str, Any]]] | None = None


def _missing_dependency(name: str):
    def _raiser(*_args: Any, **_kwargs: Any) -> Any:
        raise LightCurveError(f"Backend dependency '{name}' is not configured.")

    return _raiser


_BACKEND_DEPENDENCIES = BackendDependencies(
    load_image_data=lambda path: _load_fits_image_data(path),
    load_primary_header=lambda path: _load_fits_primary_header(path),
    load_second_hdu_rows=lambda path: _load_fits_cat_rows(path),
    world_to_pixel=lambda header, ra_deg, dec_deg: _world_to_pixel(header, ra_deg, dec_deg),
    pixel_to_world=lambda header, x, y: _pixel_to_world(header, x, y),
    get_dark_contribution=lambda _path, _header: 0.0,
    get_gain=lambda _path, header: _header_float(header, ("GAIN", "EGAIN"), DEFAULT_GAIN),
    get_read_noise=lambda _path, header: _header_float(header, ("RDNOISE", "READNOIS", "READNOISE"), DEFAULT_READ_NOISE),
    query_aavso=None,
)


def configure_backend_dependencies(dependencies: BackendDependencies) -> None:
    global _BACKEND_DEPENDENCIES
    _BACKEND_DEPENDENCIES = dependencies


def reset_backend_dependencies() -> None:
    configure_backend_dependencies(
        BackendDependencies(
            load_image_data=lambda path: _load_fits_image_data(path),
            load_primary_header=lambda path: _load_fits_primary_header(path),
            load_second_hdu_rows=lambda path: _load_fits_cat_rows(path),
            world_to_pixel=lambda header, ra_deg, dec_deg: _world_to_pixel(header, ra_deg, dec_deg),
            pixel_to_world=lambda header, x, y: _pixel_to_world(header, x, y),
            get_dark_contribution=lambda _path, _header: 0.0,
            get_gain=lambda _path, header: _header_float(header, ("GAIN", "EGAIN"), DEFAULT_GAIN),
            get_read_noise=lambda _path, header: _header_float(header, ("RDNOISE", "READNOIS", "READNOISE"), DEFAULT_READ_NOISE),
            query_aavso=None,
        )
    )


def _load_fits_image_data(fits_path: str) -> np.ndarray:
    with fits.open(fits_path) as hdul:
        try:
            data = hdul["SCI"].data
        except KeyError as exc:
            raise LightCurveError(f"SCI image HDU is missing for {fits_path}.") from exc
        if data is None:
            raise LightCurveError(f"SCI image HDU is empty for {fits_path}.")
        return np.asarray(data, dtype=float)


def _load_fits_primary_header(fits_path: str) -> Mapping[str, Any]:
    with fits.open(fits_path) as hdul:
        try:
            sci_header = dict(hdul["SCI"].header)
        except KeyError as exc:
            raise LightCurveError(f"SCI image HDU is missing for {fits_path}.") from exc
        primary_header = dict(hdul[0].header)
    return {**primary_header, **sci_header}


def _load_fits_cat_rows(fits_path: str) -> Sequence[Mapping[str, Any]]:
    with fits.open(fits_path) as hdul:
        try:
            data = hdul["CAT"].data
        except KeyError as exc:
            raise LightCurveError(f"CAT HDU is missing for {fits_path}.") from exc
        if data is None:
            raise LightCurveError(f"CAT HDU is empty for {fits_path}.")
        names = list(data.names or [])
        return [
            {name: data[name][index].item() if hasattr(data[name][index], "item") else data[name][index] for name in names}
            for index in range(len(data))
        ]


def _world_to_pixel(header: Mapping[str, Any], ra_deg: float, dec_deg: float) -> tuple[float, float]:
    wcs = _build_celestial_wcs(header)
    x, y = wcs.world_to_pixel_values(float(ra_deg), float(dec_deg))
    if not math.isfinite(float(x)) or not math.isfinite(float(y)):
        raise LightCurveError("WCS world-to-pixel conversion produced non-finite coordinates.")
    return float(x), float(y)


def _pixel_to_world(header: Mapping[str, Any], x: float, y: float) -> tuple[float, float]:
    wcs = _build_celestial_wcs(header)
    ra, dec = wcs.pixel_to_world_values(float(x), float(y))
    if not math.isfinite(float(ra)) or not math.isfinite(float(dec)):
        raise LightCurveError("WCS pixel-to-world conversion produced non-finite coordinates.")
    return float(ra), float(dec)


def _build_celestial_wcs(header: Mapping[str, Any]) -> WCS:
    wcs = WCS(dict(header))
    if not wcs.has_celestial:
        raise LightCurveError("Header does not contain a usable celestial WCS.")
    return wcs


def _header_float(header: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    for key in keys:
        if key in header:
            try:
                value = float(header[key])
            except (TypeError, ValueError):
                return default
            if math.isfinite(value):
                return value
    return default


def generate_light_curve(
    fits_paths: list[str],
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    comparison_strategy: str = "auto",
    min_comparisons: int = 5,
    max_comparisons: int = 10,
) -> LightCurveResult:
    _validate_inputs(
        fits_paths=fits_paths,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        comparison_strategy=comparison_strategy,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
    )

    diagnostics: list[str] = []
    frames = _load_and_validate_frames(fits_paths)
    frames = sorted(frames, key=lambda frame: frame.date_obs)

    target_measurements = {
        frame.fits_path: _measure_target(
            frame=frame,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
            aperture_radius_px=aperture_radius_px,
            annulus_inner_radius_px=annulus_inner_radius_px,
            annulus_outer_radius_px=annulus_outer_radius_px,
            diagnostics=diagnostics,
        )
        for frame in frames
    }

    target_mag_proxy = _target_magnitude_proxy(target_measurements.values())
    catalog, catalog_diagnostics = _build_field_star_catalog(
        frames=frames,
        target_ra_deg=target_ra_deg,
        target_dec_deg=target_dec_deg,
        aperture_radius_px=aperture_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
    )
    diagnostics.extend(catalog_diagnostics)

    selection = _select_comparison_stars(
        frames=frames,
        catalog=catalog,
        target_ra_deg=target_ra_deg,
        target_dec_deg=target_dec_deg,
        target_mag_proxy=target_mag_proxy,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        comparison_strategy=comparison_strategy,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
    )
    diagnostics.extend(selection.diagnostics)

    frame_results: list[FrameResult] = []
    light_curve_rows: list[LightCurveRow] = []

    ensemble_reference_flux = sum(
        10 ** (-0.4 * star.reference_magnitude)
        for star in selection.selected_stars
    )
    if ensemble_reference_flux <= 0.0:
        raise LightCurveError("Comparison-star magnitude calibration produced a non-positive ensemble reference flux.")

    for frame in frames:
        target = target_measurements[frame.fits_path]
        comparison_measurements = tuple(
            _measure_candidate_on_frame(
                frame=frame,
                candidate=star,
                aperture_radius_px=aperture_radius_px,
                annulus_inner_radius_px=annulus_inner_radius_px,
                annulus_outer_radius_px=annulus_outer_radius_px,
            )
            for star in selection.selected_stars
        )
        ensemble_flux = sum(m.net_source_counts for m in comparison_measurements)
        ensemble_variance = sum(m.source_uncertainty * m.source_uncertainty for m in comparison_measurements)
        if not math.isfinite(ensemble_flux) or ensemble_flux <= 0.0:
            raise LightCurveError(f"Ensemble comparison flux is invalid for frame {frame.fits_path}.")

        target_variance = target.source_uncertainty * target.source_uncertainty
        target_rel_flux = target.net_source_counts / ensemble_flux
        target_rel_flux_sigma = abs(target_rel_flux) * math.sqrt(
            _safe_fraction(target_variance, target.net_source_counts * target.net_source_counts)
            + _safe_fraction(ensemble_variance, ensemble_flux * ensemble_flux)
        )
        calibrated_mag = -2.5 * math.log10(target_rel_flux * ensemble_reference_flux)
        calibrated_mag_sigma = (2.5 / math.log(10.0)) * _safe_fraction(target_rel_flux_sigma, abs(target_rel_flux))

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

    return LightCurveResult(
        frames=frame_results,
        selected_comparison_stars=list(selection.selected_stars),
        light_curve_rows=light_curve_rows,
        diagnostics=diagnostics,
    )


def _validate_inputs(
    *,
    fits_paths: Sequence[str],
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    comparison_strategy: str,
    min_comparisons: int,
    max_comparisons: int,
) -> None:
    if not fits_paths:
        raise LightCurveError("fits_paths must be a non-empty list.")
    if aperture_radius_px <= 0:
        raise LightCurveError("aperture_radius_px must be > 0.")
    if annulus_inner_radius_px <= aperture_radius_px:
        raise LightCurveError("annulus_inner_radius_px must be greater than aperture_radius_px.")
    if annulus_outer_radius_px <= annulus_inner_radius_px:
        raise LightCurveError("annulus_outer_radius_px must be greater than annulus_inner_radius_px.")
    if min_comparisons <= 0 or max_comparisons <= 0 or min_comparisons > max_comparisons:
        raise LightCurveError("min_comparisons and max_comparisons must be positive and min_comparisons <= max_comparisons.")
    if comparison_strategy not in {"auto", "aavso_first", "variability_first"}:
        raise LightCurveError(f"Unsupported comparison_strategy '{comparison_strategy}'.")


def _load_and_validate_frames(fits_paths: Sequence[str]) -> list[FrameContext]:
    frames: list[FrameContext] = []
    for fits_path in fits_paths:
        image = np.asarray(_BACKEND_DEPENDENCIES.load_image_data(fits_path), dtype=float)
        if image.ndim != 2:
            raise LightCurveError(f"Primary image for {fits_path} is not a 2D array.")
        header = _BACKEND_DEPENDENCIES.load_primary_header(fits_path)
        date_obs = _parse_date_obs(header.get("DATE-OBS"), fits_path)
        second_hdu_rows = tuple(_normalize_rows(_BACKEND_DEPENDENCIES.load_second_hdu_rows(fits_path)))
        if not second_hdu_rows:
            raise LightCurveError(f"Second HDU is missing or empty for {fits_path}.")
        _validate_wcs(header, fits_path, image.shape)
        _validate_second_hdu(second_hdu_rows, fits_path)
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
    return frames


def _normalize_rows(rows: Sequence[Mapping[str, Any]] | Any) -> list[Mapping[str, Any]]:
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        return [dict(row) for row in rows]
    raise LightCurveError("Second HDU rows must be a sequence of mappings.")


def _parse_date_obs(value: Any, fits_path: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise LightCurveError(f"Missing DATE-OBS in {fits_path}.")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise LightCurveError(f"Malformed DATE-OBS in {fits_path}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _validate_wcs(header: Mapping[str, Any], fits_path: str, shape: tuple[int, int]) -> None:
    try:
        _BACKEND_DEPENDENCIES.world_to_pixel(header, float(header.get("CRVAL1", 0.0)), float(header.get("CRVAL2", 0.0)))
        _BACKEND_DEPENDENCIES.pixel_to_world(header, shape[1] / 2.0, shape[0] / 2.0)
    except Exception as exc:  # pragma: no cover - error path covered by tests
        raise LightCurveError(f"Missing or unusable WCS in {fits_path}.") from exc


def _validate_second_hdu(rows: Sequence[Mapping[str, Any]], fits_path: str) -> None:
    row = rows[0]
    for family, label in ((RA_KEYS, "RA"), (DEC_KEYS, "Dec"), (MAG_KEYS, "magnitude")):
        if _find_key(row, family) is None:
            raise LightCurveError(f"Second HDU in {fits_path} is missing required {label} column.")


def _measure_target(
    *,
    frame: FrameContext,
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    diagnostics: list[str],
) -> TargetMeasurement:
    try:
        initial_x, initial_y = _BACKEND_DEPENDENCIES.world_to_pixel(frame.header, target_ra_deg, target_dec_deg)
    except Exception as exc:
        diagnostics.append(f"metadata validation failure: WCS localization failed for target in {frame.fits_path}")
        raise LightCurveError(f"Target WCS localization failed for {frame.fits_path}.") from exc

    centroid = _iterative_centroid(
        image=frame.image,
        x_start=initial_x,
        y_start=initial_y,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
    )
    if centroid is None:
        diagnostics.append(f"centroid failure: target in {frame.fits_path}")
        raise LightCurveError(f"Target centroiding failed for {frame.fits_path}.")

    photometry = _measure_aperture(
        image=frame.image,
        x_center=centroid[0],
        y_center=centroid[1],
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        gain=_BACKEND_DEPENDENCIES.get_gain(frame.fits_path, frame.header),
        read_noise=_BACKEND_DEPENDENCIES.get_read_noise(frame.fits_path, frame.header),
        dark=_BACKEND_DEPENDENCIES.get_dark_contribution(frame.fits_path, frame.header),
    )
    return TargetMeasurement(
        x=centroid[0],
        y=centroid[1],
        net_source_counts=photometry["net_source_counts"],
        source_uncertainty=photometry["source_uncertainty"],
        mean_background_per_pixel=photometry["mean_background_per_pixel"],
        peak_pixel_value=photometry["peak_pixel_value"],
        effective_source_pixels=photometry["effective_source_pixels"],
        effective_background_pixels=photometry["effective_background_pixels"],
    )


def _target_magnitude_proxy(measurements: Iterable[TargetMeasurement]) -> float:
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
    aperture_radius_px: float,
    annulus_outer_radius_px: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    diagnostics: list[str] = []
    clusters: list[dict[str, Any]] = []
    target_pixels = {
        frame.fits_path: _BACKEND_DEPENDENCIES.world_to_pixel(frame.header, target_ra_deg, target_dec_deg)
        for frame in frames
    }

    for frame in frames:
        rows = [_extract_candidate_row(row, frame.fits_path) for row in frame.second_hdu_rows]
        for row in rows:
            x, y = _BACKEND_DEPENDENCIES.world_to_pixel(frame.header, row["ra_deg"], row["dec_deg"])
            row["frame_path"] = frame.fits_path
            row["pixel_x"] = x
            row["pixel_y"] = y

        for row in rows:
            x = row["pixel_x"]
            y = row["pixel_y"]
            if _too_close_to_target(
                candidate_xy=(x, y),
                target_xy=target_pixels[frame.fits_path],
                annulus_outer_radius_px=annulus_outer_radius_px,
                aperture_radius_px=aperture_radius_px,
            ):
                diagnostics.append(f"rejected comparison candidate {row['source_label']} in {frame.fits_path}: too close to target")
                continue
            if _too_close_to_edges(x, y, frame.width, frame.height, annulus_outer_radius_px):
                diagnostics.append(f"rejected comparison candidate {row['source_label']} in {frame.fits_path}: too close to image edge")
                continue
            if row["saturated"] or not row["usable"]:
                diagnostics.append(f"rejected comparison candidate {row['source_label']} in {frame.fits_path}: flagged unusable")
                continue
            if _has_close_neighbor(rows, row, annulus_outer_radius_px):
                diagnostics.append(f"rejected comparison candidate {row['source_label']} in {frame.fits_path}: nearby neighbor contamination")
                continue

            matched = False
            for cluster in clusters:
                if _angular_distance_arcsec(row["ra_deg"], row["dec_deg"], cluster["ra_deg"], cluster["dec_deg"]) <= DEFAULT_CROSSMATCH_ARCSEC:
                    cluster["rows"].append(row)
                    cluster["frame_paths"].add(frame.fits_path)
                    cluster["mags"].append(row["mag"])
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
                        "xys": [(x, y)],
                    }
                )

    catalog: list[dict[str, Any]] = []
    for idx, cluster in enumerate(
        sorted(clusters, key=lambda item: (round(item["ra_deg"], 8), round(item["dec_deg"], 8)))
    ):
        if len(cluster["frame_paths"]) < len(frames):
            diagnostics.append(
                f"rejected comparison candidate cluster near RA={cluster['ra_deg']:.6f}, Dec={cluster['dec_deg']:.6f}: insufficient frame coverage"
            )
            continue
        isolation = _minimum_neighbor_distance(cluster, clusters)
        target_sep = min(
            _distance_pixels(row["pixel_x"], row["pixel_y"], *target_pixels[row["frame_path"]])
            for row in cluster["rows"]
        )
        catalog.append(
            {
                "candidate_id": f"cand-{idx + 1:03d}",
                "ra_deg": float(np.mean([row["ra_deg"] for row in cluster["rows"]])),
                "dec_deg": float(np.mean([row["dec_deg"] for row in cluster["rows"]])),
                "second_hdu_magnitude": float(np.median(np.asarray(cluster["mags"], dtype=float))),
                "frame_coverage": len(cluster["frame_paths"]),
                "isolation_px": isolation,
                "target_separation_px": target_sep,
            }
        )
    return catalog, diagnostics


def _extract_candidate_row(row: Mapping[str, Any], fits_path: str) -> dict[str, Any]:
    ra_key = _find_key(row, RA_KEYS)
    dec_key = _find_key(row, DEC_KEYS)
    mag_key = _find_key(row, MAG_KEYS)
    if ra_key is None or dec_key is None or mag_key is None:
        raise LightCurveError(f"Second HDU rows cannot support RA/Dec matching in {fits_path}.")
    ra_deg = float(row[ra_key])
    dec_deg = float(row[dec_key])
    mag = float(row[mag_key])
    if not math.isfinite(ra_deg) or not math.isfinite(dec_deg) or not math.isfinite(mag):
        raise LightCurveError(f"Second HDU row contains malformed RA/Dec/magnitude values in {fits_path}.")
    return {
        "source_label": str(row.get("id", row.get("name", f"{ra_deg:.6f},{dec_deg:.6f}"))),
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "mag": mag,
        "saturated": any(bool(row.get(key, False)) for key in SATURATION_KEYS),
        "usable": _candidate_row_is_usable(row),
    }


def _candidate_row_is_usable(row: Mapping[str, Any]) -> bool:
    usable_key = _find_key(row, USABLE_KEYS)
    explicit_usable = bool(row[usable_key]) if usable_key is not None else True
    flagged_bad = any(bool(row.get(key, False)) for key in BAD_FLAG_KEYS)
    return explicit_usable and not flagged_bad


def _select_comparison_stars(
    *,
    frames: Sequence[FrameContext],
    catalog: Sequence[dict[str, Any]],
    target_ra_deg: float,
    target_dec_deg: float,
    target_mag_proxy: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    comparison_strategy: str,
    min_comparisons: int,
    max_comparisons: int,
) -> ComparisonSelectionResult:
    diagnostics: list[str] = []
    strategies = {
        "aavso_first": ("aavso", "variability"),
        "variability_first": ("variability", "aavso"),
        "auto": ("aavso", "variability"),
    }[comparison_strategy]

    fallback_used = False
    final_selected: tuple[ComparisonStar, ...] | None = None
    final_strategy = strategies[0]

    for index, strategy in enumerate(strategies):
        if strategy == "aavso":
            selected, strategy_diagnostics = _select_by_aavso(
                frames=frames,
                catalog=catalog,
                target_ra_deg=target_ra_deg,
                target_dec_deg=target_dec_deg,
                target_mag_proxy=target_mag_proxy,
                aperture_radius_px=aperture_radius_px,
                annulus_inner_radius_px=annulus_inner_radius_px,
                annulus_outer_radius_px=annulus_outer_radius_px,
                max_comparisons=max_comparisons,
            )
        else:
            selected, strategy_diagnostics = _select_by_variability(
                frames=frames,
                catalog=catalog,
                target_mag_proxy=target_mag_proxy,
                aperture_radius_px=aperture_radius_px,
                annulus_inner_radius_px=annulus_inner_radius_px,
                annulus_outer_radius_px=annulus_outer_radius_px,
                max_comparisons=max_comparisons,
            )
        diagnostics.extend(strategy_diagnostics)
        if len(selected) >= min_comparisons:
            final_selected = tuple(selected[:max_comparisons])
            final_strategy = strategy
            break
        if index == 0:
            fallback_used = True
            diagnostics.append(f"fallback path used: {strategy} -> {strategies[1]}")

    if final_selected is None or len(final_selected) < min_comparisons:
        raise LightCurveError("AAVSO and variability-based strategies both failed to yield the minimum comparison ensemble.")

    return ComparisonSelectionResult(
        strategy_used=final_strategy,
        fallback_used=fallback_used,
        selected_stars=final_selected,
        diagnostics=tuple(diagnostics),
    )


def _select_by_aavso(
    *,
    frames: Sequence[FrameContext],
    catalog: Sequence[dict[str, Any]],
    target_ra_deg: float,
    target_dec_deg: float,
    target_mag_proxy: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    max_comparisons: int,
) -> tuple[list[ComparisonStar], list[str]]:
    diagnostics: list[str] = []
    if _BACKEND_DEPENDENCIES.query_aavso is None:
        diagnostics.append("rejected AAVSO strategy: AAVSO client unavailable")
        return [], diagnostics

    search_radius_deg = _effective_search_radius_deg(frames[0], target_ra_deg, target_dec_deg)
    aavso_rows = _BACKEND_DEPENDENCIES.query_aavso(target_ra_deg, target_dec_deg, search_radius_deg)
    matched_ids = {
        candidate["candidate_id"]
        for candidate in catalog
        if any(
            _angular_distance_arcsec(candidate["ra_deg"], candidate["dec_deg"], float(row["ra_deg"]), float(row["dec_deg"]))
            <= DEFAULT_AAVSO_MATCH_ARCSEC
            for row in aavso_rows
        )
    }
    if not matched_ids:
        diagnostics.append("rejected AAVSO strategy: no AAVSO matches found in the second HDU catalog")
        return [], diagnostics

    enriched = _measure_and_rank_candidates(
        frames=frames,
        catalog=[candidate for candidate in catalog if candidate["candidate_id"] in matched_ids],
        target_mag_proxy=target_mag_proxy,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        matched_aavso=True,
    )
    ranked = sorted(
        enriched,
        key=lambda candidate: (
            candidate.variability_score,
            abs(candidate.reference_magnitude - target_mag_proxy),
            -candidate.isolation_px,
            candidate.candidate_id,
        ),
    )
    return ranked[:max_comparisons], diagnostics


def _select_by_variability(
    *,
    frames: Sequence[FrameContext],
    catalog: Sequence[dict[str, Any]],
    target_mag_proxy: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    max_comparisons: int,
) -> tuple[list[ComparisonStar], list[str]]:
    enriched = _measure_and_rank_candidates(
        frames=frames,
        catalog=catalog,
        target_mag_proxy=target_mag_proxy,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        matched_aavso=False,
    )
    ranked = sorted(
        [candidate for candidate in enriched if candidate.variability_score <= MAX_ACCEPTABLE_VARIABILITY],
        key=lambda candidate: (
            candidate.variability_score,
            abs(candidate.reference_magnitude - target_mag_proxy),
            -candidate.isolation_px,
            candidate.candidate_id,
        ),
    )
    return ranked[:max_comparisons], []


def _measure_and_rank_candidates(
    *,
    frames: Sequence[FrameContext],
    catalog: Sequence[dict[str, Any]],
    target_mag_proxy: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    matched_aavso: bool,
) -> list[ComparisonStar]:
    selected: list[ComparisonStar] = []
    for candidate in sorted(catalog, key=lambda row: row["candidate_id"]):
        candidate_star = ComparisonStar(
            candidate_id=candidate["candidate_id"],
            ra_deg=candidate["ra_deg"],
            dec_deg=candidate["dec_deg"],
            reference_magnitude=candidate["second_hdu_magnitude"],
            variability_score=math.inf,
            isolation_px=candidate["isolation_px"],
            target_separation_px=candidate["target_separation_px"],
            matched_aavso=matched_aavso,
        )
        try:
            per_frame = [
                _measure_candidate_on_frame(
                    frame=frame,
                    candidate=candidate_star,
                    aperture_radius_px=aperture_radius_px,
                    annulus_inner_radius_px=annulus_inner_radius_px,
                    annulus_outer_radius_px=annulus_outer_radius_px,
                )
                for frame in frames
            ]
        except LightCurveError:
            continue
        counts = np.asarray([measurement.net_source_counts for measurement in per_frame], dtype=float)
        if np.any(~np.isfinite(counts)) or np.any(counts <= 0.0):
            continue
        instrumental_mags = -2.5 * np.log10(counts)
        variability_score = float(np.std(instrumental_mags))
        selected.append(
            ComparisonStar(
                candidate_id=candidate["candidate_id"],
                ra_deg=candidate["ra_deg"],
                dec_deg=candidate["dec_deg"],
                reference_magnitude=candidate["second_hdu_magnitude"],
                variability_score=variability_score,
                isolation_px=candidate["isolation_px"],
                target_separation_px=candidate["target_separation_px"],
                matched_aavso=matched_aavso,
            )
        )
    return selected


def _measure_candidate_on_frame(
    *,
    frame: FrameContext,
    candidate: ComparisonStar,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
) -> ComparisonMeasurement:
    x, y = _BACKEND_DEPENDENCIES.world_to_pixel(frame.header, candidate.ra_deg, candidate.dec_deg)
    centroid = _iterative_centroid(
        image=frame.image,
        x_start=x,
        y_start=y,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
    )
    if centroid is None:
        raise LightCurveError(f"Selected comparison-star centroiding failed for {frame.fits_path}, {candidate.candidate_id}.")
    photometry = _measure_aperture(
        image=frame.image,
        x_center=centroid[0],
        y_center=centroid[1],
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        gain=_BACKEND_DEPENDENCIES.get_gain(frame.fits_path, frame.header),
        read_noise=_BACKEND_DEPENDENCIES.get_read_noise(frame.fits_path, frame.header),
        dark=_BACKEND_DEPENDENCIES.get_dark_contribution(frame.fits_path, frame.header),
    )
    return ComparisonMeasurement(
        candidate_id=candidate.candidate_id,
        fits_path=frame.fits_path,
        x=centroid[0],
        y=centroid[1],
        net_source_counts=photometry["net_source_counts"],
        source_uncertainty=photometry["source_uncertainty"],
        mean_background_per_pixel=photometry["mean_background_per_pixel"],
        peak_pixel_value=photometry["peak_pixel_value"],
        effective_source_pixels=photometry["effective_source_pixels"],
        effective_background_pixels=photometry["effective_background_pixels"],
    )


def _iterative_centroid(
    *,
    image: np.ndarray,
    x_start: float,
    y_start: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    max_iterations: int = 25,
    tolerance_px: float = CENTROID_TOLERANCE_PX,
) -> tuple[float, float] | None:
    height, width = image.shape
    x_center = float(x_start)
    y_center = float(y_start)
    if not (0.0 <= x_center < width and 0.0 <= y_center < height):
        return None

    for _ in range(max_iterations):
        background = _estimate_background(
            image=image,
            x_center=x_center,
            y_center=y_center,
            annulus_inner_radius_px=annulus_inner_radius_px,
            annulus_outer_radius_px=annulus_outer_radius_px,
        )
        if background is None:
            return None

        weighted_sum = 0.0
        x_weight = 0.0
        y_weight = 0.0
        min_x = max(int(math.floor(x_center - aperture_radius_px - 1)), 0)
        max_x = min(int(math.ceil(x_center + aperture_radius_px + 1)), width - 1)
        min_y = max(int(math.floor(y_center - aperture_radius_px - 1)), 0)
        max_y = min(int(math.ceil(y_center + aperture_radius_px + 1)), height - 1)

        for j in range(min_y, max_y + 1):
            for i in range(min_x, max_x + 1):
                fraction = _fractional_pixel_overlap(i, j, x_center, y_center, aperture_radius_px)
                if fraction <= 0.0:
                    continue
                signal = max((float(image[j, i]) - background) * fraction, 0.0)
                if signal <= 0.0:
                    continue
                pixel_x = i + 0.5
                pixel_y = j + 0.5
                weighted_sum += signal
                x_weight += signal * pixel_x
                y_weight += signal * pixel_y

        if weighted_sum <= 0.0:
            return None

        new_x = x_weight / weighted_sum
        new_y = y_weight / weighted_sum
        if not (0.0 <= new_x < width and 0.0 <= new_y < height):
            return None
        shift = math.hypot(new_x - x_center, new_y - y_center)
        x_center = new_x
        y_center = new_y
        if shift <= tolerance_px:
            return x_center, y_center

    return x_center, y_center


def _estimate_background(
    *,
    image: np.ndarray,
    x_center: float,
    y_center: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
) -> float | None:
    height, width = image.shape
    annulus_inner_r2 = annulus_inner_radius_px * annulus_inner_radius_px
    annulus_outer_r2 = annulus_outer_radius_px * annulus_outer_radius_px
    min_x = max(int(math.floor(x_center - annulus_outer_radius_px - 1)), 0)
    max_x = min(int(math.ceil(x_center + annulus_outer_radius_px + 1)), width - 1)
    min_y = max(int(math.floor(y_center - annulus_outer_radius_px - 1)), 0)
    max_y = min(int(math.ceil(y_center + annulus_outer_radius_px + 1)), height - 1)

    annulus_pixels: list[float] = []
    for j in range(min_y, max_y + 1):
        dy = j - y_center + 0.5
        for i in range(min_x, max_x + 1):
            dx = i - x_center + 0.5
            r2 = dx * dx + dy * dy
            if annulus_inner_r2 <= r2 <= annulus_outer_r2:
                value = float(image[j, i])
                if math.isfinite(value):
                    annulus_pixels.append(value)

    if not annulus_pixels:
        return None

    clipped = np.asarray(annulus_pixels, dtype=float)
    back_mean = 0.0
    back2_mean = 0.0
    previous_back_mean = 0.0
    for iteration in range(9):
        back_stdev = math.sqrt(max(0.0, back2_mean - back_mean * back_mean))
        lower = back_mean - 2.0 * back_stdev
        upper = back_mean + 2.0 * back_stdev
        used = clipped if iteration == 0 else clipped[(clipped >= lower) & (clipped <= upper)]
        if used.size:
            back_mean = float(np.mean(used))
            back2_mean = float(np.mean(used * used))
        if abs(previous_back_mean - back_mean) < 0.1:
            return back_mean
        previous_back_mean = back_mean
        clipped = used
    return back_mean


def _measure_aperture(
    *,
    image: np.ndarray,
    x_center: float,
    y_center: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    gain: float,
    read_noise: float,
    dark: float,
) -> dict[str, float]:
    height, width = image.shape
    source_radius = aperture_radius_px
    source_r2 = source_radius * source_radius
    annulus_inner_r2 = annulus_inner_radius_px * annulus_inner_radius_px
    annulus_outer_r2 = annulus_outer_radius_px * annulus_outer_radius_px

    min_x = max(int(math.floor(x_center - annulus_outer_radius_px - 1)), 0)
    max_x = min(int(math.ceil(x_center + annulus_outer_radius_px + 1)), width - 1)
    min_y = max(int(math.floor(y_center - annulus_outer_radius_px - 1)), 0)
    max_y = min(int(math.ceil(y_center + annulus_outer_radius_px + 1)), height - 1)

    annulus_pixels: list[float] = []
    for j in range(min_y, max_y + 1):
        dy = j - y_center + 0.5
        for i in range(min_x, max_x + 1):
            dx = i - x_center + 0.5
            r2 = dx * dx + dy * dy
            if annulus_inner_r2 <= r2 <= annulus_outer_r2:
                value = float(image[j, i])
                if math.isfinite(value):
                    annulus_pixels.append(value)

    if not annulus_pixels:
        raise LightCurveError("Background annulus does not contain any valid pixels.")

    clipped = np.asarray(annulus_pixels, dtype=float)
    back_mean = 0.0
    back2_mean = 0.0
    previous_back_mean = 0.0
    for iteration in range(9):
        back_stdev = math.sqrt(max(0.0, back2_mean - back_mean * back_mean))
        lower = back_mean - 2.0 * back_stdev
        upper = back_mean + 2.0 * back_stdev
        used = clipped if iteration == 0 else clipped[(clipped >= lower) & (clipped <= upper)]
        if used.size:
            back_mean = float(np.mean(used))
            back2_mean = float(np.mean(used * used))
        if abs(previous_back_mean - back_mean) < 0.1:
            clipped = used
            break
        previous_back_mean = back_mean
        clipped = used

    mean_background_per_pixel = max(back_mean, 0.0)
    peak = -math.inf
    source_sum = 0.0
    source_area = 0.0

    source_min_x = max(int(math.floor(x_center - source_radius - 1)), 0)
    source_max_x = min(int(math.ceil(x_center + source_radius + 1)), width - 1)
    source_min_y = max(int(math.floor(y_center - source_radius - 1)), 0)
    source_max_y = min(int(math.ceil(y_center + source_radius + 1)), height - 1)
    for j in range(source_min_y, source_max_y + 1):
        for i in range(source_min_x, source_max_x + 1):
            value = float(image[j, i])
            if not math.isfinite(value):
                continue
            fraction = _fractional_pixel_overlap(i, j, x_center, y_center, source_radius)
            if fraction <= 0.0:
                continue
            source_sum += value * fraction
            source_area += fraction
            if value > peak:
                peak = value

    if source_area <= 0.0:
        raise LightCurveError("Source aperture does not contain any valid pixels.")

    background_total = mean_background_per_pixel * source_area
    net_source = source_sum - background_total
    src = max(net_source, 0.0)
    bck = mean_background_per_pixel
    s_cnt = max(source_area, 0.0)
    src_cnt = source_area if clipped.size > 0 else 0.0
    bck_cnt = float(max(int(clipped.size), 1))
    source_uncertainty = math.sqrt(
        (src * gain)
        + s_cnt * (1.0 + src_cnt / bck_cnt) * (bck * gain + dark + read_noise * read_noise + gain * gain * 0.083521)
    ) / gain

    return {
        "net_source_counts": net_source,
        "source_uncertainty": source_uncertainty,
        "mean_background_per_pixel": mean_background_per_pixel,
        "peak_pixel_value": peak,
        "effective_source_pixels": source_area,
        "effective_background_pixels": bck_cnt,
    }


def _fractional_pixel_overlap(i: int, j: int, x_center: float, y_center: float, radius: float, substeps: int = 5) -> float:
    inside = 0
    total = substeps * substeps
    for sy in range(substeps):
        y = j + (sy + 0.5) / substeps
        for sx in range(substeps):
            x = i + (sx + 0.5) / substeps
            dx = x - x_center
            dy = y - y_center
            if dx * dx + dy * dy <= radius * radius:
                inside += 1
    return inside / total


def _find_key(mapping: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in mapping:
            return key
    lowered = {str(key).lower(): key for key in mapping}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def _distance_pixels(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _too_close_to_target(
    *,
    candidate_xy: tuple[float, float],
    target_xy: tuple[float, float],
    annulus_outer_radius_px: float,
    aperture_radius_px: float,
) -> bool:
    return _distance_pixels(*candidate_xy, *target_xy) <= max(
        TARGET_PROXIMITY_FACTOR * aperture_radius_px,
        annulus_outer_radius_px,
    )


def _too_close_to_edges(x: float, y: float, width: int, height: int, annulus_outer_radius_px: float) -> bool:
    return (
        x - annulus_outer_radius_px < EDGE_MARGIN_PX
        or y - annulus_outer_radius_px < EDGE_MARGIN_PX
        or x + annulus_outer_radius_px >= width - EDGE_MARGIN_PX
        or y + annulus_outer_radius_px >= height - EDGE_MARGIN_PX
    )


def _has_close_neighbor(rows: Sequence[dict[str, Any]], current: dict[str, Any], annulus_outer_radius_px: float) -> bool:
    for other in rows:
        if other is current:
            continue
        separation = _distance_pixels(current["pixel_x"], current["pixel_y"], other["pixel_x"], other["pixel_y"])
        if separation < annulus_outer_radius_px * NEIGHBOR_EXCLUSION_FACTOR:
            return True
    return False


def _minimum_neighbor_distance(cluster: Mapping[str, Any], clusters: Sequence[Mapping[str, Any]]) -> float:
    distances = []
    for other in clusters:
        if other is cluster:
            continue
        distances.append(
            _angular_distance_arcsec(cluster["ra_deg"], cluster["dec_deg"], other["ra_deg"], other["dec_deg"])
        )
    return min(distances) if distances else math.inf


def _angular_distance_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    cos_angle = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    cos_angle = min(1.0, max(-1.0, cos_angle))
    return math.degrees(math.acos(cos_angle)) * 3600.0


def _effective_search_radius_deg(frame: FrameContext, target_ra_deg: float, target_dec_deg: float) -> float:
    corners = [
        _BACKEND_DEPENDENCIES.pixel_to_world(frame.header, 0.0, 0.0),
        _BACKEND_DEPENDENCIES.pixel_to_world(frame.header, frame.width - 1.0, 0.0),
        _BACKEND_DEPENDENCIES.pixel_to_world(frame.header, 0.0, frame.height - 1.0),
        _BACKEND_DEPENDENCIES.pixel_to_world(frame.header, frame.width - 1.0, frame.height - 1.0),
    ]
    radii = [
        _angular_distance_arcsec(target_ra_deg, target_dec_deg, corner_ra, corner_dec) / 3600.0
        for corner_ra, corner_dec in corners
    ]
    return 0.9 * max(radii)


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise LightCurveError("Encountered a non-positive denominator in uncertainty propagation.")
    return numerator / denominator
