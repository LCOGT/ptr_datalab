"""
Aperture photometry of a target across a series of images.

One pipeline, three operations. They differ only in how the target is located on each frame and how
the comparison ensemble is maintained as the field moves under it, both of which are objects the
operation constructs and hands to the runner:

- Aperture Photometry: a fixed sky position, one comparison ensemble on every frame (sidereal).
- Non-Sidereal Aperture Photometry: the ephemeris headers of frames whose mount tracked the object.
- Moving Target Aperture Photometry: a track fitted through positions the user marked, for an
  object that moved through a sidereally-tracked field.

Three operations rather than one with a mode, because each advertises different wizard inputs, and
deliberately not a common base class: available_operations() registers every BaseDataOperation
subclass it can import, so an intermediate base would register itself as an operation with no name.
"""
import logging
from dataclasses import asdict
from typing import Any

from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation, ProgressStep
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.aperture_light_curve import (
    DEFAULT_ANNULUS_INNER_RADIUS,
    DEFAULT_ANNULUS_OUTER_RADIUS,
    DEFAULT_APERTURE_RADIUS,
    DEFAULT_MAX_COMPARISONS,
    DEFAULT_MIN_COMPARISONS,
    LightCurveError,
    Phase,
    generate_light_curve,
)
from datalab.datalab_session.utils.comparison_calibration import (
    ComparisonStrategy,
    SharedEnsemble,
    SharedThenEvolving,
)
from datalab.datalab_session.utils.diagnostic_images import save_diagnostic_images_to_s3
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.moving_target_search import DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC
from datalab.datalab_session.utils.period_analysis import period_output_from_light_curve_rows
from datalab.datalab_session.utils.target_location import (
    EphemerisHeaders,
    FittedTrack,
    FixedPosition,
    TargetLocator,
)
from datalab.datalab_session.utils.target_track import MINIMUM_TRACK_SAMPLES, track_samples_from_input


log = logging.getLogger()
log.setLevel(logging.INFO)


MINIMUM_NUMBER_OF_INPUTS = 1
MAXIMUM_NUMBER_OF_INPUTS = 999
# What the user is told during each Phase, and the progress each one ends at. Each band runs from
# the previous phase's end to its own, filling as that phase completes.
PROGRESS_STEPS = {
    Phase.DOWNLOADING: ProgressStep('Downloading input frames', 0.25),
    Phase.VALIDATE: ProgressStep('Validating input frames', 0.3),
    Phase.CATALOG: ProgressStep('Building comparison star catalog', 0.45),
    Phase.MEASURE: ProgressStep('Measuring source and comparison stars', 0.6),
    Phase.SELECT: ProgressStep('Selecting comparison stars', 0.75),
    Phase.RENDER: ProgressStep('Creating diagnostic images', 0.9),
    Phase.SAVE: ProgressStep('Saving output images', 1.0),
}
if list(PROGRESS_STEPS) != list(Phase):
    # Fail at import rather than part-way through a user's run: the old string-keyed lookup raised
    # ValueError mid-pipeline, which the operation reported as "received invalid input".
    raise RuntimeError(f"PROGRESS_STEPS must cover every Phase in order, got {list(PROGRESS_STEPS)}")


def shared_wizard_inputs() -> dict[str, Any]:
    """The input files and aperture parameters every aperture photometry operation takes."""
    return {
        'input_files': {
            'name': 'Input Files',
            'description': 'The input FITS files with SCI and CAT extensions, in a single filter',
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


def run_aperture_photometry(
    operation: BaseDataOperation,
    submitter: User,
    *,
    locator: TargetLocator,
    comparison: ComparisonStrategy,
    output_data: dict[str, Any] | None = None,
) -> None:
    """
        Runs the photometry pipeline end to end for an operation and publishes its output.

        Everything after the target is located is the same for all three operations: input
        validation, aperture parameters, file-cache resolution, calibration, period analysis,
        diagnostic upload and output shape. output_data adds any operation-specific keys to echo
        back.
    """
    input_files = operation._validate_inputs(
        input_key='input_files',
        minimum_inputs=MINIMUM_NUMBER_OF_INPUTS,
    )
    log.info(f"{operation.name()} operation on {', '.join([image['basename'] for image in input_files])}")

    def report(phase: Phase, fraction: float) -> None:
        _report_pipeline_progress(operation, phase, fraction)

    try:
        aperture_radius = float(operation.input_data['aperture_radius'])
        annulus_inner_radius = float(operation.input_data['annulus_inner_radius'])
        annulus_outer_radius = float(operation.input_data['annulus_outer_radius'])
        min_comparisons = int(operation.input_data.get('min_comparisons', DEFAULT_MIN_COMPARISONS))
        max_comparisons = int(operation.input_data.get('max_comparisons', DEFAULT_MAX_COMPARISONS))
        # Resolve inputs to local file-cache paths only. Pixel data is loaded (and released)
        # frame by frame inside generate_light_curve, never held for all inputs at once.
        file_cache = FileCache()
        fits_paths = []
        for index, input_file in enumerate(input_files, start=1):
            fits_paths.append(file_cache.get_fits(input_file['basename'], input_file.get('source'), submitter))
            report(Phase.DOWNLOADING, index / len(input_files))

        result = generate_light_curve(
            fits_paths=fits_paths,
            locator=locator,
            comparison=comparison,
            aperture_radius=aperture_radius,
            annulus_inner_radius=annulus_inner_radius,
            annulus_outer_radius=annulus_outer_radius,
            min_comparisons=min_comparisons,
            max_comparisons=max_comparisons,
            progress_callback=report,
        )
    except LightCurveError as exc:
        log.warning(f"{operation.name()} failed: {exc}")
        raise ClientAlertException(str(exc)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise ClientAlertException(f'Operation {operation.name()} received invalid input.') from exc

    diagnostic_image_urls = save_diagnostic_images_to_s3(
        cache_key=operation.cache_key,
        temp_dir=operation.temp,
        diagnostic_image_jpegs_by_fits_basename=result.diagnostic_image_jpegs_by_fits_basename,
        on_progress=lambda fraction: report(Phase.SAVE, fraction),
    )
    # Lomb-Scargle period analysis of the finished light curve, for folding a rotation or variable
    # star curve; emits the same keys as the VariableStar operation, or nothing when the light curve
    # has too few measured points to be meaningful.
    period_output = period_output_from_light_curve_rows(result.light_curve_rows)
    if period_output is None:
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
    operation.set_operation_progress(1.0)
    operation.set_message("")
    operation.set_status('COMPLETED')
    log.info(
        f"{operation.name()} output: filter={filter_value}, "
        f"light_curve_rows={len(result.light_curve_rows)}, "
        f"selected_comparison_stars={len(result.selected_comparison_stars)}, "
        f"diagnostic_images={len(diagnostic_image_urls)}"
    )


def _report_pipeline_progress(operation: BaseDataOperation, phase: Phase, fraction: float) -> None:
    """
        Advances the operation's overall progress and status message through this phase's band. The
        band runs from the previous phase's end to this one's, filling as fraction goes 0 -> 1.
    """
    phases = list(Phase)
    band = phases.index(phase)
    band_start = PROGRESS_STEPS[phases[band - 1]].progress if band > 0 else 0.0
    step = PROGRESS_STEPS[phase]
    operation.set_operation_progress(band_start + (step.progress - band_start) * fraction)
    operation.set_message(f"{step.message}: {fraction * 100:.0f}%")


class AperturePhotometry(BaseDataOperation):
    """
        Builds a calibrated aperture photometry light curve for a target that does not move against
        the stars, using comparison stars from the source catalog.
    """
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
                **shared_wizard_inputs(),
            }
        }

    def _resolve_fixed_target(self) -> dict:
        """
            The target source, as submitted.

            A fixed target is one position that does not change with time, so it stays the plain
            {ra, dec} source the wizard's name lookup produces, the same shape light_curve and
            variable_star take. The moving-target operations take a list of timed samples instead,
            which is a different thing and deliberately a different input.

            Returns the source dict, echoed back in the output so any name the wizard resolved
            survives alongside the coordinates.
        """
        source = self.input_data.get('source')
        if not source:
            raise ClientAlertException(f'Operation {self.name()} requires a source.')
        try:
            return {**source, 'ra': float(source['ra']), 'dec': float(source['dec'])}
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientAlertException(f'Invalid source coordinates: {exc}') from exc

    def operate(self, submitter: User):
        target_source = self._resolve_fixed_target()
        run_aperture_photometry(
            self,
            submitter,
            locator=FixedPosition(ra_deg=target_source['ra'], dec_deg=target_source['dec']),
            comparison=SharedEnsemble(),
            output_data={'source': target_source},
        )


class NonSiderealAperturePhotometry(BaseDataOperation):
    """
        Builds a calibrated aperture photometry light curve for a non-sidereal (moving) target, a
        minor planet, comet, or NEO, for measuring rotation from brightness modulation.

        The target has no fixed sky position: it is read per frame from the moving-target ephemeris
        header keywords (CAT-RA/CAT-DEC), so no source is supplied. Because the star field drifts as
        the target moves, calibration falls back automatically from a single shared comparison
        ensemble to an evolving, catalog-anchored per-frame zero point when no ensemble spans the
        series.
    """
    @staticmethod
    def name():
        return 'Non-Sidereal Aperture Photometry'

    @staticmethod
    def description():
        return """The non-sidereal aperture photometry operation measures a moving solar-system target across input images, locating it per frame from the ephemeris header keywords, and calibrates the light curve against comparison stars from the source catalog, carrying the calibration across a drifting star field."""

    @staticmethod
    def wizard_description():
        return {
            'name': NonSiderealAperturePhotometry.name(),
            'description': NonSiderealAperturePhotometry.description(),
            'category': 'image',
            'inputs': shared_wizard_inputs(),
        }

    def operate(self, submitter: User):
        run_aperture_photometry(
            self,
            submitter,
            locator=EphemerisHeaders(),
            comparison=SharedThenEvolving(),
        )


class MovingTargetAperturePhotometry(BaseDataOperation):
    """
        Builds a calibrated aperture photometry light curve for a moving solar-system target imaged
        on sidereally-tracked frames, where no header keyword records where the object is.

        The user identifies the target on two or more frames and submits those as {mjd, ra, dec}
        samples. A polynomial track is fitted through them, a line from two samples and a curve from
        three or more, and evaluated at each frame's exposure midpoint to predict where the target
        is. That prediction is then used to search the frame's own source catalog, so the aperture
        lands on a detected source rather than an interpolated guess.

        The counterpart to NonSiderealAperturePhotometry: there the mount tracked the object and its
        position came from the ephemeris headers; here the mount tracked the stars, so the object's
        position has to be interpolated from the user's own samples.
    """
    @staticmethod
    def name():
        return 'Moving Target Aperture Photometry'

    @staticmethod
    def description():
        return """The moving target aperture photometry operation measures a solar-system object across sidereally-tracked images, where the object moves through a fixed star field. Identify the target on at least two frames, ideally the first, the last, and one in the middle, and the operation interpolates its position on every other frame, locates it in each frame's source catalog, and calibrates the light curve against comparison stars."""

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

        run_aperture_photometry(
            self,
            submitter,
            locator=FittedTrack(samples=track_samples, search_radius_arcsec=track_search_radius),
            comparison=SharedThenEvolving(),
            output_data={
                'track_search_radius': track_search_radius,
                'target_track': [asdict(sample) for sample in track_samples],
            },
        )
