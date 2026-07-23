import logging
from dataclasses import asdict
from typing import Any

from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.aperture_light_curve import (
    DEFAULT_ANNULUS_INNER_RADIUS,
    DEFAULT_ANNULUS_OUTER_RADIUS,
    DEFAULT_APERTURE_RADIUS,
    DEFAULT_MAX_COMPARISONS,
    DEFAULT_MIN_COMPARISONS,
    LightCurveError,
    generate_light_curve,
)
from datalab.datalab_session.utils.comparison_calibration import COMPARISON_AUTO
from datalab.datalab_session.utils.diagnostic_images import save_diagnostic_images_to_s3
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.format import Format


log = logging.getLogger()
log.setLevel(logging.INFO)


# Shared by both moving-target operations. Deliberately module-level functions rather than a common
# base class: available_operations() registers every BaseDataOperation subclass it can import, so an
# intermediate base would register itself as an operation with no name.
MINIMUM_NUMBER_OF_INPUTS = 1
MAXIMUM_NUMBER_OF_INPUTS = 999
PROGRESS_STEPS = {
    'INPUT_PROCESSING_PERCENTAGE_COMPLETION': 0.2,
    'APERTURE_PHOTOMETRY_PERCENTAGE_COMPLETION': 0.9,
    'OUTPUT_PERCENTAGE_COMPLETION': 1.0
}


def shared_wizard_inputs() -> dict[str, Any]:
    """The input files and aperture parameters both moving-target operations take."""
    return {
        'input_files': {
            'name': 'Input Files',
            'description': 'The input FITS files with SCI and CAT extensions, of a single moving target in one filter',
            'type': Format.FITS,
            'single_filter': True,
            'filter_options': ['rp', 'ip', 'gp', 'zs'],
            'requires_filter': True,
            'minimum': MINIMUM_NUMBER_OF_INPUTS,
            'maximum': MAXIMUM_NUMBER_OF_INPUTS,
        },
        'aperture_radius': {
            'name': 'Aperture Radius',
            'description': 'Source aperture radius, in arcseconds (use a larger value for extended cometary targets)',
            'type': Format.FLOAT,
            'required': True,
            'default': DEFAULT_APERTURE_RADIUS,
        },
        'annulus_inner_radius': {
            'name': 'Annulus Inner Radius',
            'description': 'Background annulus inner radius, in arcseconds',
            'type': Format.FLOAT,
            'required': True,
            'default': DEFAULT_ANNULUS_INNER_RADIUS,
        },
        'annulus_outer_radius': {
            'name': 'Annulus Outer Radius',
            'description': 'Background annulus outer radius, in arcseconds',
            'type': Format.FLOAT,
            'required': True,
            'default': DEFAULT_ANNULUS_OUTER_RADIUS,
        },
        'min_comparisons': {
            'name': 'Minimum Comparison Stars',
            'description': 'Minimum number of comparison stars required per frame for calibration',
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


def run_light_curve(
    operation: BaseDataOperation,
    submitter: User,
    *,
    target_position_mode: str,
    light_curve_kwargs: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
    log_summary: str = '',
) -> None:
    """
        Runs the photometry pipeline end to end for a moving-target operation and publishes its output.

        The two operations differ only in how the target is located on each frame, so everything after
        that -- input validation, aperture parameters, file-cache resolution, calibration, diagnostic
        upload and output shape -- is shared here. target_position_mode selects the localization mode;
        light_curve_kwargs adds any mode-specific arguments for generate_light_curve, output_data any
        mode-specific keys to echo back, and log_summary is appended to the completion log.
    """
    input_files = operation._validate_inputs(
        input_key='input_files',
        minimum_inputs=MINIMUM_NUMBER_OF_INPUTS
    )
    log.info(f"{operation.name()} operation on {', '.join([image['basename'] for image in input_files])}")

    try:
        aperture_radius = float(operation.input_data['aperture_radius'])
        annulus_inner_radius = float(operation.input_data['annulus_inner_radius'])
        annulus_outer_radius = float(operation.input_data['annulus_outer_radius'])
        min_comparisons = int(operation.input_data.get('min_comparisons', DEFAULT_MIN_COMPARISONS))
        max_comparisons = int(operation.input_data.get('max_comparisons', DEFAULT_MAX_COMPARISONS))
        operation.set_operation_progress(PROGRESS_STEPS['INPUT_PROCESSING_PERCENTAGE_COMPLETION'])
        # Resolve inputs to local file-cache paths only. Pixel data is loaded (and released)
        # frame by frame inside generate_light_curve, never held for all inputs at once.
        file_cache = FileCache()
        fits_paths = [
            file_cache.get_fits(input_file['basename'], input_file.get('source'), submitter)
            for input_file in input_files
        ]
        result = generate_light_curve(
            fits_paths=fits_paths,
            aperture_radius=aperture_radius,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
            min_comparisons=min_comparisons,
            max_comparisons=max_comparisons,
            target_position_mode=target_position_mode,
            comparison_mode=COMPARISON_AUTO,
            **(light_curve_kwargs or {}),
        )
    except LightCurveError as exc:
        log.warning(f"{operation.name()} failed: {exc}")
        raise ClientAlertException(str(exc)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ClientAlertException(f'Operation {operation.name()} received invalid input.') from exc

    operation.set_operation_progress(PROGRESS_STEPS['APERTURE_PHOTOMETRY_PERCENTAGE_COMPLETION'])
    diagnostic_image_urls = save_diagnostic_images_to_s3(
        cache_key=operation.cache_key,
        temp_dir=operation.temp,
        diagnostic_image_jpegs_by_fits_basename=result.diagnostic_image_jpegs_by_fits_basename,
    )
    filter_value = input_files[0].get('filter', input_files[0].get('primary_optical_element', 'None'))
    output = {
        'output_data': [
            {
                'aperture_radius': aperture_radius,
                'annulus_inner_radius': annulus_inner_radius,
                'annulus_outer_radius': annulus_outer_radius,
                'filter': filter_value,
                'light_curve': [asdict(row) for row in result.light_curve_rows],
                'selected_comparison_stars': [
                    asdict(star) for star in result.selected_comparison_stars
                ],
                'diagnostics': result.diagnostics_by_fits_basename,
                'diagnostic_images': diagnostic_image_urls,
                **(output_data or {}),
            }
        ]
    }
    operation.set_output(output, is_raw=True)
    operation.set_operation_progress(PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
    operation.set_status('COMPLETED')
    log.info(
        f"{operation.name()} output: filter={filter_value}, "
        f"light_curve_rows={len(result.light_curve_rows)}, "
        f"selected_comparison_stars={len(result.selected_comparison_stars)}, "
        f"diagnostic_images={len(diagnostic_image_urls)}{log_summary}"
    )
