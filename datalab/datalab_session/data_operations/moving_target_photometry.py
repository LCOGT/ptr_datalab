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
    TARGET_POSITION_HEADER,
    TARGET_POSITION_TRACK,
    generate_light_curve,
)
from datalab.datalab_session.utils.comparison_calibration import COMPARISON_AUTO
from datalab.datalab_session.utils.diagnostic_images import save_diagnostic_images_to_s3
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.moving_target_search import DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC
from datalab.datalab_session.utils.period_analysis import period_output_from_light_curve_rows
from datalab.datalab_session.utils.target_track import MINIMUM_TRACK_SAMPLES, track_samples_from_input


log = logging.getLogger()
log.setLevel(logging.INFO)


# The two operations below differ only in how the target is located on each frame, so everything
# else lives in the module-level helpers they share. Deliberately not a common base class:
# available_operations() registers every BaseDataOperation subclass it can import, so an
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
            'description': 'Source aperture radius, in arcseconds (use a larger value in poor seeing, or if the target trails within an exposure)',
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
    compute_period: bool = True,
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

        compute_period adds a Lomb-Scargle period analysis of the finished light curve (for folding a
        rotation curve), emitting the same output keys as the VariableStar operation; it is skipped
        when the light curve has too few measured points to be meaningful.
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
    period_output = period_output_from_light_curve_rows(result.light_curve_rows) if compute_period else None
    if compute_period and period_output is None:
        log.info(f"{operation.name()}: too few measured points for a period search; skipped.")
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
                'pipeline_diagnostics': result.pipeline_diagnostics,
                'diagnostic_images': diagnostic_image_urls,
                **(period_output or {}),
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


class NonSiderealAperturePhotometry(BaseDataOperation):
    """
        Builds a calibrated aperture photometry light curve for a non-sidereal (moving) target, a
        minor planet, comet, or NEO, across input images, for measuring rotation from brightness
        modulation.

        The target has no fixed sky position: it is read per frame from the moving-target ephemeris
        header keywords (CAT-RA/CAT-DEC), so no source is supplied. Because the star field drifts as
        the target moves, calibration falls back automatically from a single shared comparison
        ensemble to an evolving, catalog-anchored per-frame zero point when no ensemble spans the
        series. Returns light curve rows and diagnostic data for the frontend.
    """
    @staticmethod
    def name():
        return 'Non-Sidereal Aperture Photometry'

    @staticmethod
    def description():
        return """The non-sidereal aperture photometry operation measures a moving solar-system target across input images, locating it per frame from the ephemeris header keywords, and calibrates the light curve against comparison stars from the source catalog -- carrying the calibration across a drifting star field."""

    @staticmethod
    def wizard_description():
        return {
            'name': NonSiderealAperturePhotometry.name(),
            'description': NonSiderealAperturePhotometry.description(),
            'category': 'image',
            'inputs': shared_wizard_inputs(),
        }

    def operate(self, submitter: User):
        """
            Runs non-sidereal aperture photometry for the submitted input FITS files.

            The moving target's position is read per frame from the ephemeris header keywords, so no
            source is required. Returns a calibrated light curve and diagnostic data for the frontend.
        """
        run_light_curve(self, submitter, target_position_mode=TARGET_POSITION_HEADER)


class MovingTargetAperturePhotometry(BaseDataOperation):
    """
        Builds a calibrated aperture photometry light curve for a moving solar-system target imaged
        on sidereally-tracked frames, where no header keyword records where the object is.

        The user identifies the target on two or more frames and submits those samples as
        {mjd, ra, dec} samples. A polynomial track is fitted through them, a line from two samples, a
        curve from three or more, and evaluated at each frame's exposure midpoint to predict where
        the target is. That prediction is then used to search the frame's own source catalog for the
        target, so the aperture lands on a detected source rather than an interpolated guess.

        This is the counterpart to NonSiderealAperturePhotometry: there the mount tracked the object
        and its position came from the ephemeris headers; here the mount tracked the stars, so the
        object's position has to be interpolated from the user's own samples.
    """
    @staticmethod
    def name():
        return 'Moving Target Aperture Photometry'

    @staticmethod
    def description():
        return """The moving target aperture photometry operation measures a solar-system object across sidereally-tracked images, where the object moves through a fixed star field. Identify the target on at least two frames -- ideally the first, the last, and one in the middle -- and the operation interpolates its position on every other frame, locates it in each frame's source catalog, and calibrates the light curve against comparison stars."""

    @staticmethod
    def wizard_description():
        return {
            'name': MovingTargetAperturePhotometry.name(),
            'description': MovingTargetAperturePhotometry.description(),
            'category': 'image',
            'inputs': {
                **shared_wizard_inputs(),
                'target_track': {
                    'name': 'Target Samples',
                    'description': (
                        'Where the target is on two or more frames, as {mjd, ra, dec} in decimal degrees, '
                        'with mjd the UTC exposure midpoint. Two samples interpolate along a straight '
                        'line, which holds for a night; add a third near the middle for a series spanning '
                        'more than about half a day, since apparent tracks curve.'
                    ),
                    # The shared target-position contract across all aperture photometry operations:
                    # a list of {mjd, ra, dec}. The fixed and header operations take one or zero
                    # positions; this one takes MINIMUM_TRACK_SAMPLES or more to fit a track through.
                    'type': Format.SOURCE,
                    'multiple': True,
                    'required': True,
                    'minimum': MINIMUM_TRACK_SAMPLES,
                },
                'track_search_radius': {
                    'name': 'Target Search Radius',
                    'description': (
                        'How far from the interpolated position to search each frame for the target, in '
                        'arcseconds. Widen it if the samples are sparse or the object is fast; a wider '
                        'search admits more field stars to be confused with the target.'
                    ),
                    'type': Format.FLOAT,
                    'default': DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
                },
            }
        }

    def operate(self, submitter: User):
        """
            Runs moving-target aperture photometry for the submitted input FITS files.

            The target's position on each frame is interpolated from the samples the user supplied
            and then refined against the frame's source catalog, so no ephemeris header keywords are
            needed. Returns a calibrated light curve and diagnostic data for the frontend.
        """
        raw_track = self.input_data.get('target_track')
        if not raw_track:
            raise ClientAlertException(
                f'Operation {self.name()} requires the target to be identified on at least '
                f'{MINIMUM_TRACK_SAMPLES} frames.'
            )
        try:
            track_samples = track_samples_from_input(raw_track)
            track_search_radius = float(
                self.input_data.get('track_search_radius', DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC)
            )
        except (TypeError, ValueError) as exc:
            raise ClientAlertException(f'Invalid target samples: {exc}') from exc

        run_light_curve(
            self,
            submitter,
            target_position_mode=TARGET_POSITION_TRACK,
            light_curve_kwargs={
                'target_track_samples': track_samples,
                'track_search_radius_arcsec': track_search_radius,
            },
            output_data={
                'track_search_radius': track_search_radius,
                'target_track': [asdict(sample) for sample in track_samples],
            },
            log_summary=f", track_samples={len(track_samples)}",
        )
