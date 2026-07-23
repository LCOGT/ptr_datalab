import logging
from dataclasses import asdict

from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.aperture_light_curve import (
    DEFAULT_ANNULUS_INNER_RADIUS,
    DEFAULT_ANNULUS_OUTER_RADIUS,
    DEFAULT_APERTURE_RADIUS,
    DEFAULT_MAX_COMPARISONS,
    DEFAULT_MIN_COMPARISONS,
    TARGET_POSITION_TRACK,
    LightCurveError,
    generate_light_curve,
)
from datalab.datalab_session.utils.comparison_calibration import COMPARISON_AUTO
from datalab.datalab_session.utils.moving_target_search import DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC
from datalab.datalab_session.utils.target_track import MINIMUM_TRACK_SEEDS, track_seeds_from_input


log = logging.getLogger()
log.setLevel(logging.INFO)


class MovingTargetAperturePhotometry(BaseDataOperation):
    """
        Builds a calibrated aperture photometry light curve for a moving solar-system target imaged
        on sidereally-tracked frames, where no header keyword records where the object is.

        The user identifies the target on two or more frames and submits those sightings as
        {mjd, ra, dec} seeds. A polynomial track is fitted through them -- a line from two seeds, a
        curve from three or more -- and evaluated at each frame's exposure midpoint to place the
        aperture. Calibration falls back automatically from a shared comparison ensemble to an
        evolving one, so a series whose field drifts still yields one magnitude system.

        This is the counterpart to NonSiderealAperturePhotometry: there the mount tracked the object
        and its position came from the ephemeris headers; here the mount tracked the stars, so the
        object's position has to be interpolated from the user's own sightings.
    """
    MINIMUM_NUMBER_OF_INPUTS = 1
    MAXIMUM_NUMBER_OF_INPUTS = 999
    PROGRESS_STEPS = {
        'INPUT_PROCESSING_PERCENTAGE_COMPLETION': 0.2,
        'APERTURE_PHOTOMETRY_PERCENTAGE_COMPLETION': 0.9,
        'OUTPUT_PERCENTAGE_COMPLETION': 1.0
    }

    @staticmethod
    def name():
        return 'Moving Target Aperture Photometry'

    @staticmethod
    def description():
        return """The moving target aperture photometry operation measures a solar-system object across sidereally-tracked images, where the object moves through a fixed star field. Identify the target on at least two frames -- ideally the first, the last, and one in the middle -- and the operation interpolates its position on every other frame and calibrates the light curve against comparison stars from the source catalog."""

    @staticmethod
    def wizard_description():
        return {
            'name': MovingTargetAperturePhotometry.name(),
            'description': MovingTargetAperturePhotometry.description(),
            'category': 'image',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input FITS files with SCI and CAT extensions, of a single moving target in one filter',
                    'type': Format.FITS,
                    'single_filter': True,
                    'filter_options': ['rp', 'ip', 'gp', 'zs'],
                    'requires_filter': True,
                    'minimum': MovingTargetAperturePhotometry.MINIMUM_NUMBER_OF_INPUTS,
                    'maximum': MovingTargetAperturePhotometry.MAXIMUM_NUMBER_OF_INPUTS,
                },
                'target_track': {
                    'name': 'Target Sightings',
                    'description': (
                        'Where the target is on two or more frames, as {mjd, ra, dec} in decimal degrees, '
                        'with mjd the UTC exposure midpoint. Two sightings interpolate along a straight '
                        'line, which holds for a night; add a third near the middle for a series spanning '
                        'more than about half a day, since apparent tracks curve.'
                    ),
                    'type': Format.SOURCE,
                    'multiple': True,
                    'required': True,
                    'minimum': MINIMUM_TRACK_SEEDS,
                },
                'track_search_radius': {
                    'name': 'Target Search Radius',
                    'description': (
                        'How far from the interpolated position to search each frame for the target, in '
                        'arcseconds. Widen it if the sightings are sparse or the object is fast; a wider '
                        'search admits more field stars to be confused with the target.'
                    ),
                    'type': Format.FLOAT,
                    'default': DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC,
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
        }

    def operate(self, submitter: User):
        """
            Runs moving-target aperture photometry for the submitted input FITS files.

            The target's position on each frame is interpolated from the sightings the user supplied,
            so no ephemeris header keywords are needed. Returns a calibrated light curve and
            diagnostic data for the frontend.
        """
        input_files = self._validate_inputs(
            input_key='input_files',
            minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS
        )
        log.info(f"Moving Target Aperture Photometry operation on {', '.join([image['basename'] for image in input_files])}")

        raw_track = self.input_data.get('target_track')
        if not raw_track:
            raise ClientAlertException(
                f'Operation {self.name()} requires the target to be identified on at least '
                f'{MINIMUM_TRACK_SEEDS} frames.'
            )
        try:
            track_seeds = track_seeds_from_input(raw_track)
        except ValueError as exc:
            raise ClientAlertException(f'Invalid target sightings: {exc}') from exc

        try:
            aperture_radius = float(self.input_data['aperture_radius'])
            annulus_inner_radius = float(self.input_data['annulus_inner_radius'])
            annulus_outer_radius = float(self.input_data['annulus_outer_radius'])
            min_comparisons = int(self.input_data.get('min_comparisons', DEFAULT_MIN_COMPARISONS))
            max_comparisons = int(self.input_data.get('max_comparisons', DEFAULT_MAX_COMPARISONS))
            track_search_radius = float(
                self.input_data.get('track_search_radius', DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC)
            )
            self.set_operation_progress(MovingTargetAperturePhotometry.PROGRESS_STEPS['INPUT_PROCESSING_PERCENTAGE_COMPLETION'])
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
                target_position_mode=TARGET_POSITION_TRACK,
                comparison_mode=COMPARISON_AUTO,
                target_track_seeds=track_seeds,
                track_search_radius_arcsec=track_search_radius,
            )
        except LightCurveError as exc:
            log.warning(f"Moving Target Aperture Photometry failed: {exc}")
            raise ClientAlertException(str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientAlertException(f'Operation {self.name()} received invalid input.') from exc

        self.set_operation_progress(MovingTargetAperturePhotometry.PROGRESS_STEPS['APERTURE_PHOTOMETRY_PERCENTAGE_COMPLETION'])
        filter_value = input_files[0].get('filter', input_files[0].get('primary_optical_element', 'None'))
        output = {
            'output_data': [
                {
                    'aperture_radius': aperture_radius,
                    'annulus_inner_radius': annulus_inner_radius,
                    'annulus_outer_radius': annulus_outer_radius,
                    'track_search_radius': track_search_radius,
                    'filter': filter_value,
                    'target_track': [asdict(seed) for seed in track_seeds],
                    'light_curve': [asdict(row) for row in result.light_curve_rows],
                    'selected_comparison_stars': [
                        asdict(star) for star in result.selected_comparison_stars
                    ],
                    'diagnostics': result.diagnostics_by_fits_basename,
                    'diagnostic_images': result.diagnostic_images_by_fits_basename,
                }
            ]
        }
        self.set_output(output, is_raw=True)
        self.set_operation_progress(MovingTargetAperturePhotometry.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')
        log.info(
            "Moving Target Aperture Photometry output: "
            f"filter={filter_value}, track_seeds={len(track_seeds)}, "
            f"light_curve_rows={len(result.light_curve_rows)}, "
            f"selected_comparison_stars={len(result.selected_comparison_stars)}, "
            f"diagnostic_images={len(result.diagnostic_images_by_fits_basename)}"
        )
