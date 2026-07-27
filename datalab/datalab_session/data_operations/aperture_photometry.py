import logging
from dataclasses import asdict

from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation, ProgressStep
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.diagnostic_images import save_diagnostic_images_to_s3
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.aperture_light_curve import (
    DEFAULT_ANNULUS_INNER_RADIUS,
    DEFAULT_ANNULUS_OUTER_RADIUS,
    DEFAULT_APERTURE_RADIUS,
    DEFAULT_MAX_COMPARISONS,
    DEFAULT_MIN_COMPARISONS,
    LightCurveError,
    generate_light_curve,
)
from datalab.datalab_session.utils.period_analysis import period_output_from_light_curve_rows
from datalab.datalab_session.utils.target_track import track_samples_from_input


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
        'downloading': ProgressStep('Downloading input frames', 0.25),
        'validate': ProgressStep('Validating input frames', 0.3),
        'catalog': ProgressStep('Building comparison star catalog', 0.45),
        'measure': ProgressStep('Measuring source and comparison stars', 0.6),
        'select': ProgressStep('Selecting comparison stars', 0.75),
        'render': ProgressStep('Creating diagnostic images', 0.9),
        'save': ProgressStep('Saving output images', 1.0)
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
                'target_track': {
                    'name': 'Target',
                    'type': Format.SOURCE,
                    'description': 'The target to measure, as one {ra, dec} position in decimal degrees. An mjd may be supplied for consistency with the moving-target operations but is unused: a fixed target does not move.',
                    'name_lookup': True,
                    'multiple': True,
                    'minimum': 1,
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
                'aperture_radius': {
                    'name': 'Aperture Radius',
                    'description': 'Source aperture radius, in arcseconds',
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

    def _resolve_fixed_target(self) -> dict:
        """
            The fixed target, from the unified target_track input or a legacy source.

            All aperture photometry operations now receive the target position as a list of
            {ra, dec}, optionally with an mjd; a fixed target is a single-element list, and its mjd
            (if any) is ignored, since the position does not change with time. The legacy source input
            ({ra, dec}) is still accepted so existing API clients and saved sessions keep working,
            but is no longer advertised in the wizard.

            Returns the source dict echoed back in the output, so any name the wizard resolved
            survives alongside the coordinates.
        """
        raw_track = self.input_data.get('target_track')
        if raw_track:
            try:
                samples = track_samples_from_input(raw_track, minimum=1, require_mjd=False)
            except ValueError as exc:
                raise ClientAlertException(f'Invalid target position: {exc}') from exc
            first = raw_track[0] if isinstance(raw_track[0], dict) else {}
            source = {'ra': samples[0].ra_deg, 'dec': samples[0].dec_deg}
            if first.get('name'):
                source['name'] = first['name']
            return source

        source = self.input_data.get('source')
        if source:
            try:
                return {**source, 'ra': float(source['ra']), 'dec': float(source['dec'])}
            except (KeyError, TypeError, ValueError) as exc:
                raise ClientAlertException(f'Invalid source coordinates: {exc}') from exc

        raise ClientAlertException(f'Operation {self.name()} requires a target position.')

    def operate(self, submitter: User):
        """
            Runs aperture photometry for the submitted source and input FITS files.
            
            Returns a calibrated light curve and diagnostic data for the frontend.
        """
        target_source = self._resolve_fixed_target()
        target_ra, target_dec = target_source['ra'], target_source['dec']

        input_files = self._validate_inputs(
            input_key='input_files',
            minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS
        )
        log.info(f"Aperture Photometry operation on {', '.join([image['basename'] for image in input_files])}")

        try:
            aperture_radius = float(self.input_data['aperture_radius'])
            annulus_inner_radius = float(self.input_data['annulus_inner_radius'])
            annulus_outer_radius = float(self.input_data['annulus_outer_radius'])
            min_comparisons = int(self.input_data.get('min_comparisons', DEFAULT_MIN_COMPARISONS))
            max_comparisons = int(self.input_data.get('max_comparisons', DEFAULT_MAX_COMPARISONS))
            # Resolve inputs to local file-cache paths only. Pixel data is loaded (and released)
            # frame by frame inside generate_light_curve, never held for all inputs at once.
            file_cache = FileCache()
            fits_paths = []
            for index, input_file in enumerate(input_files, start=1):
                fits_paths.append(file_cache.get_fits(input_file['basename'], input_file.get('source'), submitter))
                self._report_pipeline_progress('downloading', index / len(input_files))

            result = generate_light_curve(
                fits_paths=fits_paths,
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
                aperture_radius=aperture_radius,
                annulus_inner_radius=annulus_inner_radius,
                annulus_outer_radius=annulus_outer_radius,
                min_comparisons=min_comparisons,
                max_comparisons=max_comparisons,
                progress_callback=self._report_pipeline_progress,
            )
        except LightCurveError as exc:
            log.warning(f"Aperture Photometry failed: {exc}")
            raise ClientAlertException(str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientAlertException(f'Operation {self.name()} received invalid input.') from exc

        diagnostic_image_urls = save_diagnostic_images_to_s3(
            cache_key=self.cache_key,
            temp_dir=self.temp,
            diagnostic_image_jpegs_by_fits_basename=result.diagnostic_image_jpegs_by_fits_basename,
            on_progress=lambda fraction: self._report_pipeline_progress('save', fraction),
        )
        # Lomb-Scargle period analysis of the finished light curve, for folding a periodic (e.g.
        # variable-star) target; emits the same keys as the VariableStar operation, or nothing when
        # the light curve has too few measured points to be meaningful.
        period_output = period_output_from_light_curve_rows(result.light_curve_rows)
        filter_value = input_files[0].get('filter', input_files[0].get('primary_optical_element', 'None'))
        output = {
            'output_data': [
                {
                    'source': target_source,
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
                }
            ]
        }
        self.set_output(output, is_raw=True)
        self.set_operation_progress(1.0)
        self.set_message("")
        self.set_status('COMPLETED')
        log.info(
            "Aperture Photometry output: "
            f"filter={filter_value}, light_curve_rows={len(result.light_curve_rows)}, "
            f"selected_comparison_stars={len(result.selected_comparison_stars)}, "
            f"diagnostic_images={len(diagnostic_image_urls)}"
        )

    def _report_pipeline_progress(self, phase: str, fraction: float):
        """
            Advances this operation's overall progress and status message through the PROGRESS_STEPS
            band identified by phase. The band runs from the previous step's progress to this step's,
            filling as fraction goes 0 -> 1.
        """
        steps = list(AperturePhotometry.PROGRESS_STEPS.values())
        band = list(AperturePhotometry.PROGRESS_STEPS).index(phase)
        band_start = steps[band - 1].progress if band > 0 else 0.0
        step = steps[band]
        self.set_operation_progress(band_start + (step.progress - band_start) * fraction)
        self.set_message(f"{step.message}: {fraction * 100:.0f}%")
