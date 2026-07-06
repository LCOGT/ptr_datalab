from __future__ import annotations

import base64
import logging
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageDraw, ImageFont

from datalab.datalab_session.analysis.centroiding_core import (
    centroid,
    sigma_clipped_annulus_background,
)
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.file_utils import get_hdu
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag
from datalab.datalab_session.utils.format import Format

try:
    from django.apps import apps as django_apps
    from django.conf import settings as django_settings
except ImportError:
    django_apps = None
    django_settings = None

if (
    django_settings is not None
    and django_settings.configured
    and django_apps is not None
    and django_apps.ready
):
    from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
    from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
else:
    class BaseDataOperation:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("AperturePhotometry requires configured Django settings.")

    InputDataHandler = None

log = logging.getLogger()
log.setLevel(logging.INFO)

SOURCE_CATALOG_RA_KEY = "ra"
SOURCE_CATALOG_DEC_KEY = "dec"
SOURCE_CATALOG_MAG_KEY = "mag"
SOURCE_CATALOG_FLUX_KEY = "flux"
EDGE_MARGIN_PX = 2.0
TARGET_PROXIMITY_FACTOR = 2.0
# A target recenter is accepted only if the centroid moves less than this many
# pixels from the WCS-predicted position. Larger shifts mean the centroid was
# pulled onto a neighbour or host-galaxy structure (common for a supernova
# embedded in its host), so we keep the authoritative WCS position. The cap is
# absolute (not FWHM-relative): across a 169-frame NGC 7331 supernova set, clean
# recenters stayed <=5.3px while galaxy pulls clustered >=7.8px regardless of
# seeing, because the pull is a fixed pixel offset toward the host while the
# centroid box light is dominated by the galaxy in faint bands.
TARGET_RECENTER_MAX_SHIFT_PX = 6.0
DEFAULT_CROSSMATCH_ARCSEC = 1.0
DEFAULT_GAIN = 1.0
DEFAULT_READ_NOISE = 0.0
# AstroImageJ's photometer background convergence (Photometer.java): more
# iterations and a tighter tolerance than the centroid background, which the
# AIJ source notes is needed because 0.1 "did not work for low background levels".
APERTURE_BACKGROUND_MAX_ITERATIONS = 100
APERTURE_BACKGROUND_TOLERANCE = 1e-4
MAX_ACCEPTABLE_VARIABILITY = 1.0
# A comparison candidate's catalog magnitude and this pipeline's measured instrumental
# magnitude should differ only by the frame ensemble's zero point, which is common to
# all well-behaved stars. A candidate whose (catalog_mag - instrumental_mag) departs
# from the ensemble median by more than this many magnitudes has an unreliable catalog
# magnitude (usually a blended/mismatched cross-match) and is rejected: it would be
# selected on its bright measured flux but then bias the catalog-magnitude-weighted
# zero-point calibration.
MAX_ZERO_POINT_RESIDUAL_MAG = 0.5
DEFAULT_APERTURE_RADIUS_PX = 7.64
DEFAULT_ANNULUS_INNER_RADIUS_PX = 12.73
DEFAULT_ANNULUS_OUTER_RADIUS_PX = 19.10
DEFAULT_MIN_COMPARISONS = 5
DEFAULT_MAX_COMPARISONS = 10
# A comparison candidate is established if its cross-matched cluster is detected in
# at least this fraction of frames. Catalog detection near the limiting magnitude is
# noisy, so requiring presence in *every* frame discards good stars over a single
# missed detection; selected stars are still measured (via WCS) on all frames, so the
# comparison ensemble stays consistent frame-to-frame regardless of this threshold.
COMPARISON_FRAME_COVERAGE_FRACTION = 0.8
DIAGNOSTIC_COMPARISON_STAR_COLOR = (0, 173, 239)
DIAGNOSTIC_TARGET_COLOR = (243, 131, 33)


class AperturePhotometry(BaseDataOperation):
    MINIMUM_NUMBER_OF_INPUTS = 1
    MAXIMUM_NUMBER_OF_INPUTS = 999
    PROGRESS_STEPS = {
        'INPUT_PROCESSING_PERCENTAGE_COMPLETION': 0.2,
        'APERTURE_PHOTOMETRY_PERCENTAGE_COMPLETION': 0.9,
        'OUTPUT_PERCENTAGE_COMPLETION': 1.0
    }

    @staticmethod
    def name():
        return 'Aperture Photometry'

    @staticmethod
    def description():
        return """The aperture photometry operation measures a target source across input images and calibrates the light curve with comparison stars selected from the source catalog."""

    @staticmethod
    def wizard_description():
        return {
            'name': AperturePhotometry.name(),
            'description': AperturePhotometry.description(),
            'category': 'image',
            'inputs': {
                'source': {
                    'name': 'Source Star',
                    'type': Format.SOURCE,
                    'description': 'The source star to measure',
                    'name_lookup': True
                },
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input FITS files with SCI and CAT extensions',
                    'type': Format.FITS,
                    'single_filter': True,
                    'filter_options': ['rp', 'ip', 'gp', 'zs'],
                    'requires_filter': True,
                    'minimum': AperturePhotometry.MINIMUM_NUMBER_OF_INPUTS,
                    'maximum': AperturePhotometry.MAXIMUM_NUMBER_OF_INPUTS,
                },
                'aperture_radius_px': {
                    'name': 'Aperture Radius',
                    'description': 'Source aperture radius in pixels',
                    'type': Format.FLOAT,
                    'required': True,
                    'default': DEFAULT_APERTURE_RADIUS_PX,
                },
                'annulus_inner_radius_px': {
                    'name': 'Annulus Inner Radius',
                    'description': 'Background annulus inner radius in pixels',
                    'type': Format.FLOAT,
                    'required': True,
                    'default': DEFAULT_ANNULUS_INNER_RADIUS_PX,
                },
                'annulus_outer_radius_px': {
                    'name': 'Annulus Outer Radius',
                    'description': 'Background annulus outer radius in pixels',
                    'type': Format.FLOAT,
                    'required': True,
                    'default': DEFAULT_ANNULUS_OUTER_RADIUS_PX,
                },
                'min_comparisons': {
                    'name': 'Minimum Comparison Stars',
                    'description': 'Minimum number of comparison stars required for calibration',
                    'type': Format.INT,
                    'default': DEFAULT_MIN_COMPARISONS,
                },
                'max_comparisons': {
                    'name': 'Maximum Comparison Stars',
                    'description': 'Maximum number of comparison stars used for calibration',
                    'type': Format.INT,
                    'default': DEFAULT_MAX_COMPARISONS,
                },
            }
        }

    def operate(self, submitter: Any):
        source = self.input_data.get('source')
        if not source:
            raise ClientAlertException(f'Operation {self.name()} requires a source.')

        input_files = self._validate_inputs(
            input_key='input_files',
            minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS
        )
        filter_value = _required_single_filter(input_files, self.name())
        input_basenames = [input_file.get('basename', '<missing basename>') for input_file in input_files]
        log.info(
            "Aperture Photometry operation starting: "
            f"cache_key={getattr(self, 'cache_key', '<no cache key>')}, "
            f"input_count={len(input_files)}, filter={filter_value}, basenames={input_basenames}"
        )

        try:
            target_ra = float(source.get('ra'))
            target_dec = float(source.get('dec'))
        except (TypeError, ValueError) as exc:
            raise ClientAlertException(f'Operation {self.name()} requires numeric source coordinates.') from exc

        aperture_radius_px = _required_float_input(self.input_data, 'aperture_radius_px', self.name())
        annulus_inner_radius_px = _required_float_input(self.input_data, 'annulus_inner_radius_px', self.name())
        annulus_outer_radius_px = _required_float_input(self.input_data, 'annulus_outer_radius_px', self.name())
        min_comparisons = int(self.input_data.get('min_comparisons', DEFAULT_MIN_COMPARISONS))
        max_comparisons = int(self.input_data.get('max_comparisons', DEFAULT_MAX_COMPARISONS))
        log.info(
            "Aperture Photometry parameters: "
            f"target_ra={target_ra:.8f}, target_dec={target_dec:.8f}, "
            f"aperture_radius_px={aperture_radius_px:.3f}, "
            f"annulus_inner_radius_px={annulus_inner_radius_px:.3f}, "
            f"annulus_outer_radius_px={annulus_outer_radius_px:.3f}, "
            f"min_comparisons={min_comparisons}, max_comparisons={max_comparisons}"
        )
        self.set_operation_progress(AperturePhotometry.PROGRESS_STEPS['INPUT_PROCESSING_PERCENTAGE_COMPLETION'])

        fits_paths = []
        for input_file in input_files:
            fits_path = InputDataHandler(submitter, input_file['basename'], input_file.get('source')).fits_file
            fits_paths.append(fits_path)
            log.info(
                "Aperture Photometry input resolved: "
                f"basename={input_file.get('basename')}, source={input_file.get('source')}, fits_path={fits_path}"
            )

        try:
            result = generate_light_curve(
                fits_paths=fits_paths,
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
                aperture_radius_px=aperture_radius_px,
                annulus_inner_radius_px=annulus_inner_radius_px,
                annulus_outer_radius_px=annulus_outer_radius_px,
                min_comparisons=min_comparisons,
                max_comparisons=max_comparisons,
            )
        except LightCurveError as exc:
            log.warning(f"Aperture Photometry failed: {exc}")
            raise ClientAlertException(str(exc)) from exc

        self.set_operation_progress(AperturePhotometry.PROGRESS_STEPS['APERTURE_PHOTOMETRY_PERCENTAGE_COMPLETION'])
        non_finite_rows = [
            row.fits_path
            for row in result.light_curve_rows
            if (
                not math.isfinite(row.target_calibrated_apparent_magnitude)
                or not math.isfinite(row.target_calibrated_apparent_magnitude_uncertainty)
            )
        ]
        log.info(
            "Aperture Photometry result summary: "
            f"light_curve_rows={len(result.light_curve_rows)}, "
            f"selected_comparison_stars={len(result.selected_comparison_stars)}, "
            f"diagnostics={len(result.diagnostics)}, non_finite_light_curve_rows={len(non_finite_rows)}"
        )
        if non_finite_rows:
            log.warning(
                "Aperture Photometry generated non-finite calibrated magnitude values. "
                "The frontend light-curve plot will skip these rows: "
                f"{non_finite_rows}"
            )
        output = {
            'output_data': [
                {
                    'source': source,
                    'filter': filter_value,
                    'light_curve': [_dataclass_to_plain_dict(row) for row in result.light_curve_rows],
                    'selected_comparison_stars': [
                        _dataclass_to_plain_dict(star) for star in result.selected_comparison_stars
                    ],
                    'diagnostics': _diagnostics_by_fits_basename(result),
                    'diagnostic_images': _diagnostic_images_by_fits_basename(result),
                }
            ]
        }
        self.set_output(output, is_raw=True)
        self.set_operation_progress(AperturePhotometry.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')
        log.info(
            "Aperture Photometry operation completed: "
            f"cache_key={getattr(self, 'cache_key', '<no cache key>')}"
        )


class LightCurveError(ValueError):
    pass


def _required_float_input(input_data: Mapping[str, Any], key: str, operation_name: str) -> float:
    if key not in input_data or input_data.get(key) in (None, ''):
        raise ClientAlertException(f'Operation {operation_name} requires {key}.')
    try:
        return float(input_data[key])
    except (TypeError, ValueError) as exc:
        raise ClientAlertException(f'Operation {operation_name} requires numeric {key}.') from exc


def _required_single_filter(input_files: Sequence[Mapping[str, Any]], operation_name: str) -> str:
    filters = []
    for input_file in input_files:
        filter_value = input_file.get('filter') or input_file.get('primary_optical_element')
        if not filter_value:
            basename = input_file.get('basename', 'input file')
            raise ClientAlertException(f'Operation {operation_name} requires a filter for {basename}.')
        filters.append(str(filter_value))

    unique_filters = set(filters)
    if len(unique_filters) > 1:
        raise ClientAlertException(f'Operation {operation_name} requires all input files to use the same filter.')
    return filters[0]


def _dataclass_to_plain_dict(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        return _json_safe_value(dict(value))
    plain = asdict(value)
    return _json_safe_value(plain)


def _diagnostics_by_fits_basename(result: Any) -> dict[str, list[str]]:
    grouped = getattr(result, "diagnostics_by_fits_basename", None)
    if grouped is not None:
        return {
            str(basename): list(diagnostics)
            for basename, diagnostics in grouped.items()
        }

    basenames = [
        os.path.basename(row.fits_path)
        for row in getattr(result, "light_curve_rows", [])
    ]
    diagnostics = list(getattr(result, "diagnostics", []))
    return {basename: list(diagnostics) for basename in basenames}


def _diagnostic_images_by_fits_basename(result: Any) -> dict[str, str]:
    grouped = getattr(result, "diagnostic_images_by_fits_basename", None)
    if grouped is None:
        return {}
    return {
        str(basename): str(image_base64)
        for basename, image_base64 in grouped.items()
        if image_base64
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value

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
    # Median instrumental magnitude (-2.5*log10(net counts)) measured across all
    # frames with this pipeline's aperture. On the same zero point as the target's
    # magnitude proxy, so the two are directly comparable for brightness matching.
    measured_instrumental_magnitude: float = math.inf


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
    selected_comparison_stars: list[ComparisonStar]
    light_curve_rows: list[LightCurveRow]
    diagnostics: list[str]
    diagnostics_by_fits_basename: dict[str, list[str]]
    diagnostic_images_by_fits_basename: dict[str, str]


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


def _default_backend_dependencies() -> BackendDependencies:
    return BackendDependencies(
        load_image_data=lambda path: _load_fits_image_data(path),
        load_primary_header=lambda path: _load_fits_primary_header(path),
        load_second_hdu_rows=lambda path: _load_fits_cat_rows(path),
        world_to_pixel=lambda header, ra_deg, dec_deg: _world_to_pixel(header, ra_deg, dec_deg),
        pixel_to_world=lambda header, x, y: _pixel_to_world(header, x, y),
        get_dark_contribution=lambda _path, _header: 0.0,
        get_gain=lambda _path, header: _header_float(header, ("GAIN", "EGAIN"), DEFAULT_GAIN),
        get_read_noise=lambda _path, header: _header_float(
            header,
            ("RDNOISE", "READNOIS", "READNOISE"),
            DEFAULT_READ_NOISE,
        ),
    )


_BACKEND_DEPENDENCIES = _default_backend_dependencies()


def configure_backend_dependencies(dependencies: BackendDependencies) -> None:
    global _BACKEND_DEPENDENCIES
    _BACKEND_DEPENDENCIES = dependencies


def reset_backend_dependencies() -> None:
    configure_backend_dependencies(_default_backend_dependencies())


def _load_fits_image_data(fits_path: str) -> np.ndarray:
    try:
        data = get_hdu(fits_path, extension="SCI").data
    except ClientAlertException as exc:
        raise LightCurveError(f"SCI image HDU is missing for {fits_path}.") from exc
    if data is None:
        raise LightCurveError(f"SCI image HDU is empty for {fits_path}.")
    return np.asarray(data, dtype=float)


def _load_fits_primary_header(fits_path: str) -> Mapping[str, Any]:
    try:
        sci_header = dict(get_hdu(fits_path, extension="SCI").header)
    except ClientAlertException as exc:
        raise LightCurveError(f"SCI image HDU is missing for {fits_path}.") from exc
    with fits.open(fits_path) as hdul:
        primary_header = dict(hdul[0].header)
    return {**primary_header, **sci_header}


def _load_fits_cat_rows(fits_path: str) -> Sequence[Mapping[str, Any]]:
    try:
        data = get_hdu(fits_path, extension="CAT").data
    except ClientAlertException as exc:
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
    min_comparisons: int = 5,
    max_comparisons: int = 10,
    aperture_unit: str = "px",
) -> LightCurveResult:
    log.info(
        "Aperture Photometry pipeline starting: "
        f"fits_count={len(fits_paths)}, target_ra={target_ra_deg:.8f}, target_dec={target_dec_deg:.8f}, "
        f"aperture_radius_px={aperture_radius_px:.3f}, "
        f"annulus_inner_radius_px={annulus_inner_radius_px:.3f}, "
        f"annulus_outer_radius_px={annulus_outer_radius_px:.3f}, "
        f"min_comparisons={min_comparisons}, max_comparisons={max_comparisons}"
    )
    _validate_inputs(
        fits_paths=fits_paths,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
    )

    diagnostics: list[str] = []
    frames, skipped_frames = _load_and_validate_frames(fits_paths)
    if not frames:
        raise LightCurveError(
            "No input frames had a usable source catalog (RA/Dec/magnitude columns)."
        )
    frames = sorted(frames, key=lambda frame: frame.date_obs)
    diagnostics_by_fits_basename: dict[str, list[str]] = {
        os.path.basename(frame.fits_path): []
        for frame in frames
    }
    for basename, reason in skipped_frames:
        message = f"Skipped {basename}: {reason}."
        diagnostics.append(message)
        diagnostics_by_fits_basename.setdefault(basename, []).append(message)
    log.info(
        "Aperture Photometry frames loaded and sorted: "
        f"frame_count={len(frames)}, skipped={len(skipped_frames)}, "
        f"ordered_paths={[frame.fits_path for frame in frames]}"
    )

    target_measurements = {
        frame.fits_path: _measure_target(
            frame=frame,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
            aperture_radius_px=aperture_radius_px,
            annulus_inner_radius_px=annulus_inner_radius_px,
            annulus_outer_radius_px=annulus_outer_radius_px,
            diagnostics=diagnostics_by_fits_basename[os.path.basename(frame.fits_path)],
            aperture_unit=aperture_unit,
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

    target_mag_proxy = _target_magnitude_proxy(target_measurements.values())
    log.info(f"Aperture Photometry target magnitude proxy: {target_mag_proxy:.6f}")
    catalog = _build_field_star_catalog(
        frames=frames,
        target_ra_deg=target_ra_deg,
        target_dec_deg=target_dec_deg,
        aperture_radius_px=aperture_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        aperture_unit=aperture_unit,
    )
    log.info(f"Aperture Photometry comparison catalog built: valid_candidates={len(catalog)}")

    selected_stars = _select_comparison_stars(
        frames=frames,
        catalog=catalog,
        target_mag_proxy=target_mag_proxy,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        min_comparisons=min_comparisons,
        max_comparisons=max_comparisons,
        aperture_unit=aperture_unit,
    )
    log.info(
        "Aperture Photometry comparison stars selected: "
        f"selected_count={len(selected_stars)}, "
        f"candidate_ids={[star.candidate_id for star in selected_stars]}"
    )

    ensemble_reference_flux = sum(
        10 ** (-0.4 * star.reference_magnitude)
        for star in selected_stars
    )
    if ensemble_reference_flux <= 0.0:
        raise LightCurveError("Comparison-star magnitude calibration produced a non-positive ensemble reference flux.")
    ensemble_reference_mag = -2.5 * math.log10(ensemble_reference_flux)
    log.info(
        "Aperture Photometry calibration reference: "
        f"ensemble_reference_flux={ensemble_reference_flux:.12e}, "
        f"ensemble_reference_mag={ensemble_reference_mag:.6f}"
    )

    light_curve_rows: list[LightCurveRow] = []
    diagnostic_images_by_fits_basename: dict[str, str] = {}
    for frame in frames:
        target = target_measurements[frame.fits_path]
        comparison_measurements = tuple(
            _measure_candidate_on_frame(
                frame=frame,
                candidate=star,
                aperture_radius_px=aperture_radius_px,
                annulus_inner_radius_px=annulus_inner_radius_px,
                annulus_outer_radius_px=annulus_outer_radius_px,
                aperture_unit=aperture_unit,
            )
            for star in selected_stars
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
        # Zero point that calibrates instrumental magnitudes against the comparison
        # ensemble; also reused below for the comparison-star validation diagnostics.
        frame_zero_point = 2.5 * math.log10(ensemble_flux) + ensemble_reference_mag
        if target.net_source_counts > 0.0:
            calibrated_mag = -2.5 * math.log10(target.net_source_counts) + frame_zero_point
            _, calibrated_mag_sigma = flux_to_mag(target_rel_flux, target_rel_flux_sigma)
        else:
            calibrated_mag = math.nan
            calibrated_mag_sigma = math.nan
        log.info(
            "Aperture Photometry frame calibration: "
            f"frame={frame.fits_path}, target_counts={target.net_source_counts:.6f}, "
            f"comparison_ensemble_counts={ensemble_flux:.6f}, "
            f"target_rel_flux={target_rel_flux:.12e}, target_rel_flux_sigma={target_rel_flux_sigma:.12e}, "
            f"frame_zero_point={frame_zero_point:.6f}, "
            f"calibrated_mag={_format_float(calibrated_mag, precision=6)}, "
            f"calibrated_mag_sigma={_format_float(calibrated_mag_sigma, precision=6)}"
        )
        if not math.isfinite(calibrated_mag) or not math.isfinite(calibrated_mag_sigma):
            log.warning(
                "Aperture Photometry non-finite light-curve row: "
                f"frame={frame.fits_path}, calibrated_mag={calibrated_mag}, "
                f"calibrated_mag_sigma={calibrated_mag_sigma}. "
                "This row is present in backend output as null after JSON serialization and the frontend plot skips it."
            )
        frame_diagnostics = _comparison_star_validation_diagnostics(
            frame=frame,
            stars=selected_stars,
            measurements=comparison_measurements,
            target_measurement=target,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
            frame_zero_point=frame_zero_point,
        )
        diagnostics.extend(frame_diagnostics)
        diagnostics_by_fits_basename[os.path.basename(frame.fits_path)].extend(frame_diagnostics)
        frame_aperture_radius_px = aperture_radius_px / _aperture_px_scale(frame, aperture_unit)
        diagnostic_images_by_fits_basename[os.path.basename(frame.fits_path)] = _candidate_overlay_jpeg_base64(
            frame=frame,
            stars=selected_stars,
            measurements=comparison_measurements,
            target_measurement=target,
            aperture_radius_px=frame_aperture_radius_px,
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
        f"light_curve_rows={len(light_curve_rows)}, "
        f"selected_comparison_stars={len(selected_stars)}, diagnostics={len(diagnostics)}"
    )
    return LightCurveResult(
        selected_comparison_stars=list(selected_stars),
        light_curve_rows=light_curve_rows,
        diagnostics=diagnostics,
        diagnostics_by_fits_basename=diagnostics_by_fits_basename,
        diagnostic_images_by_fits_basename=diagnostic_images_by_fits_basename,
    )


def _validate_inputs(
    *,
    fits_paths: Sequence[str],
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
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


def _load_and_validate_frames(
    fits_paths: Sequence[str],
) -> tuple[list[FrameContext], list[tuple[str, str]]]:
    """Load frames, skipping any without a usable source catalog.

    Returns the usable frames plus a list of ``(basename, reason)`` for frames
    skipped because their catalog (second HDU) could not provide the RA/Dec/
    magnitude/flux columns the comparison-star matching needs. Skipping (rather
    than aborting) keeps a single bad frame from killing the whole light curve;
    poor-conditions frames often have an uncalibrated catalog.
    """
    frames: list[FrameContext] = []
    skipped: list[tuple[str, str]] = []
    for fits_path in fits_paths:
        basename = os.path.basename(fits_path)
        log.info(f"Aperture Photometry loading FITS frame: {fits_path}")
        image = np.asarray(_BACKEND_DEPENDENCIES.load_image_data(fits_path), dtype=float)
        if image.ndim != 2:
            raise LightCurveError(f"Primary image for {fits_path} is not a 2D array.")
        header = _BACKEND_DEPENDENCIES.load_primary_header(fits_path)
        date_obs = _parse_date_obs(header.get("DATE-OBS"), fits_path)
        _validate_wcs(header, fits_path, image.shape)

        try:
            second_hdu_rows = tuple(_normalize_rows(_BACKEND_DEPENDENCIES.load_second_hdu_rows(fits_path)))
        except LightCurveError as exc:
            skipped.append((basename, f"source catalog could not be read ({exc})"))
            log.warning(f"Aperture Photometry skipping {basename}: source catalog could not be read ({exc})")
            continue
        if not second_hdu_rows:
            skipped.append((basename, "source catalog (second HDU) is empty"))
            log.warning(f"Aperture Photometry skipping {basename}: source catalog is empty")
            continue
        missing_column = _missing_catalog_column(second_hdu_rows)
        if missing_column is not None:
            skipped.append((basename, f"source catalog is missing the {missing_column} column"))
            log.warning(f"Aperture Photometry skipping {basename}: source catalog is missing the {missing_column} column")
            continue

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
    return frames, skipped


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


def _missing_catalog_column(rows: Sequence[Mapping[str, Any]]) -> str | None:
    row = rows[0]
    for key, label in (
        (SOURCE_CATALOG_RA_KEY, "RA"),
        (SOURCE_CATALOG_DEC_KEY, "Dec"),
        (SOURCE_CATALOG_MAG_KEY, "magnitude"),
        (SOURCE_CATALOG_FLUX_KEY, "flux"),
    ):
        if key not in row:
            return label
    return None


def _frame_pixel_scale_arcsec(header: Any) -> float:
    """Arcsec-per-pixel for a frame, from its WCS astrometric solution (preferred) or
    the nominal PIXSCALE header, so an angular aperture can be sized in pixels."""
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scale = float(np.mean(proj_plane_pixel_scales(WCS(header)) * 3600.0))
        if math.isfinite(scale) and scale > 0.0:
            return scale
    except Exception:
        pass
    pixscale = header.get("PIXSCALE") if hasattr(header, "get") else None
    if pixscale is not None and float(pixscale) > 0.0:
        return float(pixscale)
    raise LightCurveError("Cannot determine a pixel scale for an arcsec-unit aperture.")


def _aperture_px_scale(frame: FrameContext, aperture_unit: str) -> float:
    """Divisor that converts aperture radii to pixels for a frame. 1.0 when the radii
    are already pixels; the frame's arcsec/px plate scale when they are an angular size
    (so the aperture covers the same sky on every telescope regardless of pixel scale)."""
    if aperture_unit == "px":
        return 1.0
    if aperture_unit != "arcsec":
        raise LightCurveError(f"Unsupported aperture_unit {aperture_unit!r}; expected 'px' or 'arcsec'.")
    return _frame_pixel_scale_arcsec(frame.header)


def _measure_target(
    *,
    frame: FrameContext,
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    diagnostics: list[str],
    aperture_unit: str = "px",
) -> TargetMeasurement:
    scale = _aperture_px_scale(frame, aperture_unit)
    aperture_radius_px = aperture_radius_px / scale
    annulus_inner_radius_px = annulus_inner_radius_px / scale
    annulus_outer_radius_px = annulus_outer_radius_px / scale
    try:
        initial_x, initial_y = _BACKEND_DEPENDENCIES.world_to_pixel(frame.header, target_ra_deg, target_dec_deg)
    except Exception as exc:
        raise LightCurveError(f"Target WCS localization failed for {frame.fits_path}.") from exc
    log.info(
        "Aperture Photometry target WCS localization: "
        f"frame={frame.fits_path}, initial_pixel=({initial_x:.3f}, {initial_y:.3f})"
    )

    centroid = _iterative_centroid(
        image=frame.image,
        x_start=initial_x,
        y_start=initial_y,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
    )

    # Accept the recenter only if it stays within TARGET_RECENTER_MAX_SHIFT_PX of the
    # WCS-predicted position; a larger shift means the centroid was pulled onto a
    # neighbour or host-galaxy structure rather than the target, so we fall back to
    # the authoritative WCS position.
    basename = os.path.basename(frame.fits_path)

    if centroid is None:
        x_center, y_center = initial_x, initial_y
        message = (
            f"Target recenter skipped on {basename}: centroiding failed; "
            "measured at the WCS position instead. Frame retained."
        )
        diagnostics.append(message)
        log.warning(f"Aperture Photometry {message}")
    else:
        recenter_shift_px = math.hypot(centroid[0] - initial_x, centroid[1] - initial_y)
        if recenter_shift_px > TARGET_RECENTER_MAX_SHIFT_PX:
            x_center, y_center = initial_x, initial_y
            message = (
                f"Target recenter skipped on {basename}: centroid shift "
                f"{recenter_shift_px:.2f}px exceeded the {TARGET_RECENTER_MAX_SHIFT_PX:.2f}px "
                "limit; measured at the WCS position instead. Frame retained."
            )
            diagnostics.append(message)
            log.warning(f"Aperture Photometry {message}")
        else:
            x_center, y_center = centroid
    log.info(
        "Aperture Photometry target centroid: "
        f"frame={frame.fits_path}, position=({x_center:.3f}, {y_center:.3f})"
    )

    photometry = _measure_aperture(
        image=frame.image,
        x_center=x_center,
        y_center=y_center,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        gain=_BACKEND_DEPENDENCIES.get_gain(frame.fits_path, frame.header),
        read_noise=_BACKEND_DEPENDENCIES.get_read_noise(frame.fits_path, frame.header),
        dark=_BACKEND_DEPENDENCIES.get_dark_contribution(frame.fits_path, frame.header),
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


def _candidate_overlay_jpeg_base64(
    *,
    frame: FrameContext,
    stars: Sequence[ComparisonStar],
    measurements: Sequence[ComparisonMeasurement],
    target_measurement: TargetMeasurement,
    aperture_radius_px: float,
) -> str:
    image = _normalize_image_for_jpeg(frame.image)
    draw = ImageDraw.Draw(image)
    font = _diagnostic_overlay_font(frame.width, frame.height)
    stars_by_id = {star.candidate_id: star for star in stars}
    min_dimension = max(min(frame.width, frame.height), 1)
    radius = max(float(aperture_radius_px), min_dimension * 0.018, 14.0)
    line_width = max(3, int(round(min_dimension * 0.004)))
    label_padding = max(3, int(round(min_dimension * 0.004)))

    for measurement in measurements:
        if measurement.candidate_id not in stars_by_id:
            continue
        x = float(measurement.x)
        y = _diagnostic_display_y(float(measurement.y), frame.height)
        if not math.isfinite(x) or not math.isfinite(y):
            continue

        label = measurement.candidate_id
        halo_bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(
            halo_bbox,
            outline=(0, 0, 0),
            width=line_width + 2,
        )
        draw.ellipse(
            halo_bbox,
            outline=DIAGNOSTIC_COMPARISON_STAR_COLOR,
            width=line_width,
        )

        label_x = x + radius + label_padding
        label_y = y - radius - label_padding
        label_bbox = draw.textbbox((label_x, label_y), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        if label_x + label_width + label_padding > frame.width:
            label_x = max(x - radius - label_width - label_padding, 0)
        if label_y < 0:
            label_y = min(y + radius + label_padding, max(frame.height - label_height - label_padding, 0))
        draw.text((label_x, label_y), label, fill=DIAGNOSTIC_COMPARISON_STAR_COLOR, font=font)

    target_x = float(target_measurement.x)
    target_y = _diagnostic_display_y(float(target_measurement.y), frame.height)
    if math.isfinite(target_x) and math.isfinite(target_y):
        target_bbox = (
            target_x - radius,
            target_y - radius,
            target_x + radius,
            target_y + radius,
        )
        draw.ellipse(
            target_bbox,
            outline=(0, 0, 0),
            width=line_width + 2,
        )
        draw.ellipse(
            target_bbox,
            outline=DIAGNOSTIC_TARGET_COLOR,
            width=line_width,
        )

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _normalize_image_for_jpeg(image_data: np.ndarray) -> Image.Image:
    finite = np.asarray(image_data, dtype=float)
    finite_values = finite[np.isfinite(finite)]
    if finite_values.size:
        zmin, zmax = np.percentile(finite_values, (1, 99.7))
    else:
        zmin, zmax = 0.0, 1.0
    if not math.isfinite(float(zmin)) or not math.isfinite(float(zmax)) or zmax <= zmin:
        zmin = float(np.nanmin(finite_values)) if finite_values.size else 0.0
        zmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
    if zmax <= zmin:
        zmax = zmin + 1.0

    scaled = np.clip((finite - zmin) / (zmax - zmin), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    gray = np.flip((scaled * 255.0).astype(np.uint8), axis=0)
    return Image.fromarray(gray).convert("RGB")


def _diagnostic_display_y(y: float, height: int) -> float:
    return float(height - 1) - y


def _diagnostic_overlay_font(width: int, height: int) -> ImageFont.ImageFont:
    font_size = max(32, min(160, int(round(min(width, height) * 0.05))))
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except OSError:
        return ImageFont.load_default()


def _target_magnitude_proxy(measurements: Iterable[TargetMeasurement]) -> float:
    instrumental_mags = [
        -2.5 * math.log10(measurement.net_source_counts)
        for measurement in measurements
        if measurement.net_source_counts > 0
    ]
    if not instrumental_mags:
        raise LightCurveError("Target photometry never produced positive source counts.")
    return float(np.median(np.asarray(instrumental_mags, dtype=float)))


def _comparison_star_validation_diagnostics(
    *,
    frame: FrameContext,
    stars: Sequence[ComparisonStar],
    measurements: Sequence[ComparisonMeasurement],
    target_measurement: TargetMeasurement,
    target_ra_deg: float,
    target_dec_deg: float,
    frame_zero_point: float,
) -> list[str]:
    diagnostics = [
        (
            "comparison star identifier | RA | Dec | calculated flux | FITS source catalog flux | "
            "calculated magnitude | FITS source catalog magnitude"
        ),
    ]
    stars_by_id = {star.candidate_id: star for star in stars}
    for measurement in measurements:
        star = stars_by_id[measurement.candidate_id]
        catalog_row = star.source_catalog_by_frame.get(frame.fits_path, {})
        calculated_magnitude = _flux_to_magnitude(measurement.net_source_counts, frame_zero_point)
        fits_catalog_flux = _optional_float(catalog_row.get("flux"))
        fits_catalog_mag = _optional_float(catalog_row.get("mag"))
        diagnostics.append(
            "comparison-star validation row: "
            f"{star.candidate_id} | "
            f"{_format_float(star.ra_deg, precision=3)} | "
            f"{_format_float(star.dec_deg, precision=3)} | "
            f"{_format_float(measurement.net_source_counts, precision=0)} | "
            f"{_format_float(fits_catalog_flux, precision=0)} | "
            f"{_format_float(calculated_magnitude, precision=3)} | "
            f"{_format_float(fits_catalog_mag, precision=3)}"
        )
    return diagnostics


def _flux_to_magnitude(flux: float, zero_point: float) -> float:
    if not math.isfinite(flux) or flux <= 0.0 or not math.isfinite(zero_point):
        return math.nan
    magnitude, _ = flux_to_mag(flux, 0.0)
    return float(magnitude) + zero_point


def _optional_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _format_float(value: float, *, precision: int) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


def _build_field_star_catalog(
    *,
    frames: Sequence[FrameContext],
    target_ra_deg: float,
    target_dec_deg: float,
    aperture_radius_px: float,
    annulus_outer_radius_px: float,
    aperture_unit: str = "px",
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    target_pixels = {
        frame.fits_path: _BACKEND_DEPENDENCIES.world_to_pixel(frame.header, target_ra_deg, target_dec_deg)
        for frame in frames
    }

    for frame in frames:
        frame_scale = _aperture_px_scale(frame, aperture_unit)
        frame_aperture_radius_px = aperture_radius_px / frame_scale
        frame_annulus_outer_radius_px = annulus_outer_radius_px / frame_scale
        rows: list[dict[str, Any]] = []
        for raw_row in frame.second_hdu_rows:
            try:
                rows.append(_extract_candidate_row(raw_row, frame.fits_path))
            except LightCurveError as exc:
                log.warning(f"rejected comparison candidate in {frame.fits_path}: {exc}")
        rejected_for_target = 0
        rejected_for_edge = 0
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
                annulus_outer_radius_px=frame_annulus_outer_radius_px,
                aperture_radius_px=frame_aperture_radius_px,
            ):
                rejected_for_target += 1
                continue
            if _too_close_to_edges(x, y, frame.width, frame.height, frame_annulus_outer_radius_px):
                rejected_for_edge += 1
                continue

            matched = False
            for cluster in clusters:
                if frame.fits_path in cluster["frame_paths"]:
                    continue
                if _angular_distance_arcsec(row["ra_deg"], row["dec_deg"], cluster["ra_deg"], cluster["dec_deg"]) <= DEFAULT_CROSSMATCH_ARCSEC:
                    cluster["rows"].append(row)
                    cluster["frame_paths"].add(frame.fits_path)
                    cluster["mags"].append(row["mag"])
                    cluster["source_catalog_by_frame"][frame.fits_path] = {
                        "source_label": row["source_label"],
                        "flux": row["flux"],
                        "mag": row["mag"],
                    }
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
                "source_catalog_by_frame": dict(cluster["source_catalog_by_frame"]),
                "isolation_px": isolation,
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


def _select_comparison_stars(
    *,
    frames: Sequence[FrameContext],
    catalog: Sequence[dict[str, Any]],
    target_mag_proxy: float,
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    min_comparisons: int,
    max_comparisons: int,
    aperture_unit: str = "px",
) -> tuple[ComparisonStar, ...]:
    enriched = _measure_and_rank_candidates(
        frames=frames,
        catalog=catalog,
        aperture_radius_px=aperture_radius_px,
        annulus_inner_radius_px=annulus_inner_radius_px,
        annulus_outer_radius_px=annulus_outer_radius_px,
        aperture_unit=aperture_unit,
    )
    stable = [candidate for candidate in enriched if candidate.variability_score <= MAX_ACCEPTABLE_VARIABILITY]
    consistent = _reject_zero_point_outliers(stable)
    ranked = sorted(
        consistent,
        key=lambda candidate: _source_catalog_sort_key(candidate, target_mag_proxy),
    )

    if len(ranked) < min_comparisons:
        raise LightCurveError("Source-catalog variability strategy failed to yield the minimum comparison ensemble.")

    return tuple(ranked[:max_comparisons])


def _reject_zero_point_outliers(candidates: Sequence[ComparisonStar]) -> list[ComparisonStar]:
    # A good comparison star's (catalog_mag - measured_instrumental_mag) equals the
    # frame ensemble's zero point, common to all such stars. Candidates whose residual
    # departs from the median by more than MAX_ZERO_POINT_RESIDUAL_MAG have an
    # untrustworthy catalog magnitude (typically a blended/mismatched cross-match) and
    # would bias the catalog-magnitude-weighted zero-point calibration, so drop them.
    # Needs a few stars to estimate a robust median; below that, keep all.
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
    rejected = [candidate for candidate in candidates if candidate not in kept]
    if rejected:
        log.info(
            "Aperture Photometry rejected zero-point-inconsistent comparison candidates: "
            f"median_residual={median_residual:.4f}, "
            f"rejected={[(c.candidate_id, round(c.reference_magnitude - c.measured_instrumental_magnitude, 4)) for c in rejected]}"
        )
    # Never let the consistency guard empty the pool; if it would, fall back to the
    # unfiltered set and let the min_comparisons check decide.
    return kept if len(kept) >= 1 else list(candidates)


def _source_catalog_sort_key(candidate: ComparisonStar, target_mag_proxy: float) -> tuple[float, float, str]:
    # Rank by closeness in brightness to the target. Both sides are instrumental
    # magnitudes measured by this pipeline (median over frames), so they share a
    # zero point; comparing the catalog's calibrated reference_magnitude against
    # the instrumental target_mag_proxy would instead compare across a ~zero-point
    # offset and collapse to "pick the brightest catalog star".
    return (
        abs(candidate.measured_instrumental_magnitude - target_mag_proxy),
        -candidate.isolation_px,
        candidate.candidate_id,
    )


def _measure_and_rank_candidates(
    *,
    frames: Sequence[FrameContext],
    catalog: Sequence[dict[str, Any]],
    aperture_radius_px: float,
    annulus_inner_radius_px: float,
    annulus_outer_radius_px: float,
    aperture_unit: str = "px",
) -> list[ComparisonStar]:
    measured_candidates: list[tuple[dict[str, Any], ComparisonStar, np.ndarray]] = []
    for candidate in sorted(catalog, key=lambda row: row["candidate_id"]):
        reference_magnitude = float(candidate["second_hdu_magnitude"])
        reference_magnitude_source = "second_hdu"
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
                _measure_candidate_on_frame(
                    frame=frame,
                    candidate=candidate_star,
                    aperture_radius_px=aperture_radius_px,
                    annulus_inner_radius_px=annulus_inner_radius_px,
                    annulus_outer_radius_px=annulus_outer_radius_px,
                    aperture_unit=aperture_unit,
                )
                for frame in frames
            ]
        except LightCurveError:
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
        variability_score = float(np.std(variability_mags))
        selected.append(
            ComparisonStar(
                candidate_id=candidate["candidate_id"],
                ra_deg=candidate["ra_deg"],
                dec_deg=candidate["dec_deg"],
                reference_magnitude=candidate_star.reference_magnitude,
                reference_magnitude_source=candidate_star.reference_magnitude_source,
                source_catalog_by_frame=candidate_star.source_catalog_by_frame,
                variability_score=variability_score,
                isolation_px=candidate_star.isolation_px,
                target_separation_px=candidate_star.target_separation_px,
                measured_instrumental_magnitude=float(np.median(instrumental_mags)),
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
    aperture_unit: str = "px",
) -> ComparisonMeasurement:
    scale = _aperture_px_scale(frame, aperture_unit)
    aperture_radius_px = aperture_radius_px / scale
    annulus_inner_radius_px = annulus_inner_radius_px / scale
    annulus_outer_radius_px = annulus_outer_radius_px / scale
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
) -> tuple[float, float] | None:
    # Delegate to the shared AstroImageJ-style centroid (Howell marginal-sum with
    # an iterative sigma-clipped background) that also backs the interactive
    # centroiding endpoint, instead of maintaining a separate centroid here. The
    # aperture geometry maps directly onto the centroid's source radius and
    # background annulus.
    result = centroid(
        image,
        float(x_start),
        float(y_start),
        aperture_radius_px,
        annulus_inner_radius_px,
        annulus_outer_radius_px,
        find_centroid=True,
        remove_background_stars=True,
        use_plane_background=False,
    )
    if not result.success:
        return None
    return result.x, result.y


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

    background_result = sigma_clipped_annulus_background(
        image,
        x_center,
        y_center,
        annulus_inner_radius_px,
        annulus_outer_radius_px,
        remove_stars=True,
        max_iterations=APERTURE_BACKGROUND_MAX_ITERATIONS,
        tolerance=APERTURE_BACKGROUND_TOLERANCE,
    )
    if background_result is None:
        raise LightCurveError("Background annulus does not contain any valid pixels.")
    background_mean, kept_background_pixels = background_result
    background_pixel_count = len(kept_background_pixels)

    mean_background_per_pixel = max(background_mean, 0.0)
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
    src_cnt = source_area if background_pixel_count > 0 else 0.0
    bck_cnt = float(max(background_pixel_count, 1))
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



def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        raise LightCurveError("Encountered a non-positive denominator in uncertainty propagation.")
    return numerator / denominator
