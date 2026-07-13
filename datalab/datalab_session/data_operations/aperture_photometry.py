import logging
from dataclasses import asdict

from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.input_data_handler import InputDataHandler
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.aperture_light_curve import (
    DEFAULT_ANNULUS_INNER_RADIUS_ARCSEC,
    DEFAULT_ANNULUS_OUTER_RADIUS_ARCSEC,
    DEFAULT_APERTURE_RADIUS_ARCSEC,
    DEFAULT_MAX_COMPARISONS,
    DEFAULT_MIN_COMPARISONS,
    LightCurveError,
    generate_light_curve,
)


log = logging.getLogger()
log.setLevel(logging.INFO)


class AperturePhotometry(BaseDataOperation):
    """
        Builds a calibrated aperture photometry light curve for a target source across input images, using comparison stars from the source catalog.

        Returns light curve rows and diagnostic data for the frontend.
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
                'aperture_radius_arcsec': {
                    'name': 'Aperture Radius',
                    'description': 'Source aperture radius, in arcseconds',
                    'type': Format.FLOAT,
                    'required': True,
                    'default': DEFAULT_APERTURE_RADIUS_ARCSEC,
                },
                'annulus_inner_radius_arcsec': {
                    'name': 'Annulus Inner Radius',
                    'description': 'Background annulus inner radius, in arcseconds',
                    'type': Format.FLOAT,
                    'required': True,
                    'default': DEFAULT_ANNULUS_INNER_RADIUS_ARCSEC,
                },
                'annulus_outer_radius_arcsec': {
                    'name': 'Annulus Outer Radius',
                    'description': 'Background annulus outer radius, in arcseconds',
                    'type': Format.FLOAT,
                    'required': True,
                    'default': DEFAULT_ANNULUS_OUTER_RADIUS_ARCSEC,
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

    def operate(self, submitter: User):
        """
            Runs aperture photometry for the submitted source and input FITS files.
            
            Returns a calibrated light curve and diagnostic data for the frontend.
        """
        source = self.input_data.get('source')
        if not source:
            raise ClientAlertException(f'Operation {self.name()} requires a source.')

        input_files = self._validate_inputs(
            input_key='input_files',
            minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS
        )
        log.info(f"Aperture Photometry operation on {', '.join([image['basename'] for image in input_files])}")

        try:
            target_ra = float(source.get('ra'))
            target_dec = float(source.get('dec'))
            aperture_radius_arcsec = float(self.input_data['aperture_radius_arcsec'])
            annulus_inner_radius_arcsec = float(self.input_data['annulus_inner_radius_arcsec'])
            annulus_outer_radius_arcsec = float(self.input_data['annulus_outer_radius_arcsec'])
            min_comparisons = int(self.input_data.get('min_comparisons', DEFAULT_MIN_COMPARISONS))
            max_comparisons = int(self.input_data.get('max_comparisons', DEFAULT_MAX_COMPARISONS))
            self.set_operation_progress(AperturePhotometry.PROGRESS_STEPS['INPUT_PROCESSING_PERCENTAGE_COMPLETION'])
            input_handlers = [
                InputDataHandler(submitter, input_file['basename'], input_file.get('source'))
                for input_file in input_files
            ]
            result = generate_light_curve(
                input_handlers=input_handlers,
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
                aperture_radius_arcsec=aperture_radius_arcsec,
                annulus_inner_radius_arcsec=annulus_inner_radius_arcsec,
                annulus_outer_radius_arcsec=annulus_outer_radius_arcsec,
                min_comparisons=min_comparisons,
                max_comparisons=max_comparisons,
            )
        except LightCurveError as exc:
            log.warning(f"Aperture Photometry failed: {exc}")
            raise ClientAlertException(str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientAlertException(f'Operation {self.name()} received invalid input.') from exc

        self.set_operation_progress(AperturePhotometry.PROGRESS_STEPS['APERTURE_PHOTOMETRY_PERCENTAGE_COMPLETION'])
        filter_value = input_files[0].get('filter', input_files[0].get('primary_optical_element', 'None'))
        output = {
            'output_data': [
                {
                    'source': source,
                    'aperture_radius_arcsec': aperture_radius_arcsec,
                    'annulus_inner_radius_arcsec': annulus_inner_radius_arcsec,
                    'annulus_outer_radius_arcsec': annulus_outer_radius_arcsec,
                    'filter': filter_value,
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
        self.set_operation_progress(AperturePhotometry.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')
        log.info(
            "Aperture Photometry output: "
            f"filter={filter_value}, light_curve_rows={len(result.light_curve_rows)}, "
            f"selected_comparison_stars={len(result.selected_comparison_stars)}, "
            f"diagnostic_images={len(result.diagnostic_images_by_fits_basename)}"
        )
