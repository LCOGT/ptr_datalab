import logging

from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.moving_target_photometry import (
    run_light_curve,
    shared_wizard_inputs,
)
from datalab.datalab_session.utils.aperture_light_curve import TARGET_POSITION_HEADER


log = logging.getLogger()
log.setLevel(logging.INFO)


class NonSiderealAperturePhotometry(BaseDataOperation):
    """
        Builds a calibrated aperture photometry light curve for a non-sidereal (moving) target -- a
        minor planet, comet, or NEO -- across input images, for measuring rotation from brightness
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
        return """The non-sidereal aperture photometry operation measures a moving solar-system target across input images, locating it per frame from the ephemeris header keywords, and calibrates the light curve against comparison stars from the source catalog -- carrying the calibration across a drifting star field. Extended (cometary) targets may need a larger aperture."""

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
