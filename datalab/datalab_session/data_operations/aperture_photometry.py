"""
Aperture photometry of a target across a series of images.

One pipeline, three operations, differing only in the TargetLocator and ComparisonStrategy each
constructs. Three rather than one with a mode, because each advertises different wizard inputs.
"""
import logging
from abc import ABC
from dataclasses import asdict, dataclass
from typing import Any, Mapping

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
from datalab.datalab_session.utils.file_utils import temp_file_manager
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.moving_target_search import DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC
from datalab.datalab_session.utils.period_analysis import PeriodAnalysis
from datalab.datalab_session.utils.s3_utils import save_files_to_s3
from datalab.datalab_session.utils.target_location import (
    EphemerisHeaders,
    FittedTrack,
    FixedPosition,
    TargetLocator,
)
from datalab.datalab_session.utils.target_track import MINIMUM_TRACK_SAMPLES, TrackSample


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


@dataclass(frozen=True)
class ApertureParameters:
    """The aperture geometry and comparison-count limits, as submitted."""
    aperture_radius: float
    annulus_inner_radius: float
    annulus_outer_radius: float
    min_comparisons: int
    max_comparisons: int


class AperturePhotometryOperation(BaseDataOperation, ABC):
    """Shared implementation for the aperture photometry operations."""

    def _validate_target_positions(self, *, minimum: int, require_mjd: bool) -> tuple[TrackSample, ...]:
        """The submitted target positions, parsed and sorted by time."""
        raw = self.input_data.get('target_positions')
        if raw is None:
            # What the wizard sent before the target_positions input existed.
            raw = self.input_data.get('source')
        if raw is not None:
            self.input_data['target_positions'] = [raw] if isinstance(raw, Mapping) else raw

        positions = self.input_data.get('target_positions', [])
        if not positions or len(positions) < minimum:
            raise ClientAlertException(f'Operation {self.name()} requires at least {minimum} target_positions')

        return TrackSample.from_input(positions, require_mjd=require_mjd)


    def _submitted_target_name(self) -> str | None:
        """Any name the wizard's lookup resolved, so the output can echo it back with the position."""
        first = self.input_data['target_positions'][0]
        return first.get('name') if isinstance(first, Mapping) else None

    def _validate_aperture_parameters(self) -> ApertureParameters:
        """The aperture radii and comparison-count limits, coerced from the submitted input."""
        try:
            return ApertureParameters(
                aperture_radius=float(self.input_data['aperture_radius']),
                annulus_inner_radius=float(self.input_data['annulus_inner_radius']),
                annulus_outer_radius=float(self.input_data['annulus_outer_radius']),
                min_comparisons=int(self.input_data.get('min_comparisons', DEFAULT_MIN_COMPARISONS)),
                max_comparisons=int(self.input_data.get('max_comparisons', DEFAULT_MAX_COMPARISONS)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientAlertException(f'Operation {self.name()} received invalid input.') from exc

    def _run_photometry(
        self,
        submitter: User,
        *,
        locator: TargetLocator,
        comparison: ComparisonStrategy,
        output_data: dict[str, Any] | None = None,
    ) -> None:
        """Runs the pipeline and publishes the output. output_data adds operation-specific keys."""
        input_files = self._validate_file_inputs('input_files')
        log.info(f"{self.name()} operation on {', '.join([image['basename'] for image in input_files])}")
        parameters = self._validate_aperture_parameters()

        try:
            # Pixel data is loaded and released frame by frame inside generate_light_curve, so only
            # the paths are resolved here.
            file_cache = FileCache()
            fits_paths = []
            for index, input_file in enumerate(input_files, start=1):
                fits_paths.append(file_cache.get_fits(input_file['basename'], input_file.get('source'), submitter))
                self._report_progress(Phase.DOWNLOADING, index / len(input_files))

            result = generate_light_curve(
                fits_paths=fits_paths,
                locator=locator,
                comparison=comparison,
                aperture_radius=parameters.aperture_radius,
                annulus_inner_radius=parameters.annulus_inner_radius,
                annulus_outer_radius=parameters.annulus_outer_radius,
                min_comparisons=parameters.min_comparisons,
                max_comparisons=parameters.max_comparisons,
                progress_callback=self._report_progress,
            )
        except LightCurveError as exc:
            log.warning(f"{self.name()} failed: {exc}")
            raise ClientAlertException(str(exc)) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ClientAlertException(f'Operation {self.name()} received invalid input.') from exc

        diagnostic_image_urls = self._save_diagnostic_images(result.diagnostic_image_jpegs_by_fits_basename)
        period = PeriodAnalysis.from_light_curve_rows(result.light_curve_rows)
        if period is None:
            log.info(f"{self.name()}: too few measured points for a period search; skipped.")
        # 'period'/'fap'/'frequency'/'power' match VariableStar, so one frontend renderer drives both.
        period_output = {} if period is None else {
            'period': period.period,
            'fap': period.false_alarm_probability,
            'frequency': period.frequency,
            'power': period.power,
            'period_candidates': [asdict(candidate) for candidate in period.candidates],
            'window_power': period.window_power,
        }
        filter_value = input_files[0].get('filter', input_files[0].get('primary_optical_element', 'None'))
        output = {
            'output_data': [
                {
                    'aperture_radius': parameters.aperture_radius,
                    'annulus_inner_radius': parameters.annulus_inner_radius,
                    'annulus_outer_radius': parameters.annulus_outer_radius,
                    'filter': filter_value,
                    'light_curve': [asdict(row) for row in result.light_curve_rows],
                    'selected_comparison_stars': [
                        asdict(star) for star in result.selected_comparison_stars
                    ],
                    'diagnostics': result.diagnostics_by_fits_basename,
                    'pipeline_diagnostics': result.pipeline_diagnostics,
                    'diagnostic_images': diagnostic_image_urls,
                    **period_output,
                    **(output_data or {}),
                }
            ]
        }
        self.set_output(output, is_raw=True)
        self.set_operation_progress(1.0)
        self.set_message("")
        self.set_status('COMPLETED')
        log.info(
            f"{self.name()} output: filter={filter_value}, "
            f"light_curve_rows={len(result.light_curve_rows)}, "
            f"selected_comparison_stars={len(result.selected_comparison_stars)}, "
            f"diagnostic_images={len(diagnostic_image_urls)}"
        )

    def _report_progress(self, phase: Phase, fraction: float) -> None:
        """Fills this phase's progress band, which runs from the previous phase's end to its own."""
        phases = list(Phase)
        band = phases.index(phase)
        band_start = PROGRESS_STEPS[phases[band - 1]].progress if band > 0 else 0.0
        step = PROGRESS_STEPS[phase]
        self.set_operation_progress(band_start + (step.progress - band_start) * fraction)
        self.set_message(f"{step.message}: {fraction * 100:.0f}%")

    def _save_diagnostic_images(self, jpegs_by_fits_basename: dict[str, bytes]) -> dict[str, str]:
        """Uploads each frame's diagnostic overlay, returning FITS basename to presigned url."""
        urls: dict[str, str] = {}
        total = len(jpegs_by_fits_basename)
        for index, (fits_basename, jpeg_bytes) in enumerate(jpegs_by_fits_basename.items(), start=1):
            with temp_file_manager(f'{self.cache_key}-{index}-diagnostic.jpg', dir=self.temp) as jpeg_path:
                with open(jpeg_path, 'wb') as jpeg_file:
                    jpeg_file.write(jpeg_bytes)
                s3_output = save_files_to_s3(self.cache_key, Format.IMAGE, {'diagnostic_jpg_path': jpeg_path}, index=index)
            urls[fits_basename] = s3_output['diagnostic_url']
            self._report_progress(Phase.SAVE, index / total)
        return urls


class AperturePhotometry(AperturePhotometryOperation):
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
                'target_positions': {
                    'name': 'Target',
                    'type': Format.TARGET_POSITIONS,
                    'description': 'The target to measure, as one {ra, dec} in decimal degrees',
                    'name_lookup': True,
                    'minimum': 1,
                },
                **shared_wizard_inputs(),
            }
        }

    def operate(self, submitter: User):
        positions = self._validate_target_positions(minimum=1, require_mjd=False)
        source = {'ra': positions[0].ra_deg, 'dec': positions[0].dec_deg}
        name = self._submitted_target_name()
        if name:
            source['name'] = name
        self._run_photometry(
            submitter,
            locator=FixedPosition(ra_deg=source['ra'], dec_deg=source['dec']),
            comparison=SharedEnsemble(),
            output_data={'source': source},
        )


class NonSiderealAperturePhotometry(AperturePhotometryOperation):
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
        self._run_photometry(
            submitter,
            locator=EphemerisHeaders(),
            comparison=SharedThenEvolving(),
        )


class MovingTargetAperturePhotometry(AperturePhotometryOperation):
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
                'target_positions': {
                    'name': 'Target Positions',
                    'description': (
                        'Where the target is on two or more frames, as {mjd, ra, dec} in decimal degrees, '
                        'with mjd the UTC exposure midpoint. Two samples interpolate along a straight '
                        'line, which holds for a night; add a third near the middle for a series spanning '
                        'more than about half a day, since apparent tracks curve.'
                    ),
                    'type': Format.TARGET_POSITIONS,
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
        track_samples = self._validate_target_positions(
            minimum=MINIMUM_TRACK_SAMPLES, require_mjd=True
        )
        try:
            track_search_radius = float(
                self.input_data.get('track_search_radius', DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC)
            )
        except (TypeError, ValueError) as exc:
            raise ClientAlertException(f'Operation {self.name()}: invalid target search radius.') from exc

        self._run_photometry(
            submitter,
            locator=FittedTrack(samples=track_samples, search_radius_arcsec=track_search_radius),
            comparison=SharedThenEvolving(),
            output_data={
                'track_search_radius': track_search_radius,
                'target_positions': [asdict(sample) for sample in track_samples],
            },
        )
