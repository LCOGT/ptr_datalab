import logging
from dataclasses import asdict

from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.moving_target_photometry import (
    run_light_curve,
    shared_wizard_inputs,
)
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.aperture_light_curve import TARGET_POSITION_TRACK
from datalab.datalab_session.utils.format import Format
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
        curve from three or more -- and evaluated at each frame's exposure midpoint to predict where
        the target is. That prediction is then used to search the frame's own source catalog for the
        target, so the aperture lands on a detected source rather than an interpolated guess.

        This is the counterpart to NonSiderealAperturePhotometry: there the mount tracked the object
        and its position came from the ephemeris headers; here the mount tracked the stars, so the
        object's position has to be interpolated from the user's own sightings.
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
                    'name': 'Target Sightings',
                    'description': (
                        'Where the target is on two or more frames, as {mjd, ra, dec} in decimal degrees, '
                        'with mjd the UTC exposure midpoint. Two sightings interpolate along a straight '
                        'line, which holds for a night; add a third near the middle for a series spanning '
                        'more than about half a day, since apparent tracks curve.'
                    ),
                    # The shared target-position contract across all aperture photometry operations:
                    # a list of {mjd, ra, dec}. The fixed and header operations take one or zero
                    # positions; this one takes MINIMUM_TRACK_SEEDS or more to fit a track through.
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
            }
        }

    def operate(self, submitter: User):
        """
            Runs moving-target aperture photometry for the submitted input FITS files.

            The target's position on each frame is interpolated from the sightings the user supplied
            and then refined against the frame's source catalog, so no ephemeris header keywords are
            needed. Returns a calibrated light curve and diagnostic data for the frontend.
        """
        raw_track = self.input_data.get('target_track')
        if not raw_track:
            raise ClientAlertException(
                f'Operation {self.name()} requires the target to be identified on at least '
                f'{MINIMUM_TRACK_SEEDS} frames.'
            )
        try:
            track_seeds = track_seeds_from_input(raw_track)
            track_search_radius = float(
                self.input_data.get('track_search_radius', DEFAULT_TRACK_SEARCH_RADIUS_ARCSEC)
            )
        except (TypeError, ValueError) as exc:
            raise ClientAlertException(f'Invalid target sightings: {exc}') from exc

        run_light_curve(
            self,
            submitter,
            target_position_mode=TARGET_POSITION_TRACK,
            light_curve_kwargs={
                'target_track_seeds': track_seeds,
                'track_search_radius_arcsec': track_search_radius,
            },
            output_data={
                'track_search_radius': track_search_radius,
                'target_track': [asdict(seed) for seed in track_seeds],
            },
            log_summary=f", track_seeds={len(track_seeds)}",
        )
