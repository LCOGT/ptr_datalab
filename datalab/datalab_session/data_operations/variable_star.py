import logging

import numpy as np
from astropy.timeseries import LombScargle
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.light_curve import light_curve
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format

log = logging.getLogger()
log.setLevel(logging.INFO)


class VariableStar(BaseDataOperation):
    MINIMUM_NUMBER_OF_INPUTS = 8
    MAXIMUM_NUMBER_OF_INPUTS = 999
    PROGRESS_STEPS = {
        'DOWNLOADING_INPUT_FILES': 0.8,
        'PERIOD_FOLDING_DONE': 0.9,
        'OUTPUT_PERCENTAGE_COMPLETION': 1.0
    }
    @staticmethod
    def name():
        return 'Variable Star'
    
    @staticmethod
    def description():
        return """The variable star operation takes in a target star, filter, and list of image ids to examine. Astropy's LombScargle algorithm is used on the photometry for period stacking.

The output is the lightcurve and periodogram data for period stacking of the photometry."""

    @staticmethod
    def wizard_description():
        description = {
            'name': VariableStar.name(),
            'description': VariableStar.description(),
            'category': 'image',
            'inputs': {
                'source': {
                    'name': 'Source Star',
                    'type': Format.SOURCE,
                    'description': 'The source star to analyze',
                    'name_lookup': True
                },
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to pull photometry from',
                    'type': Format.FITS,
                    'single_filter': True,
                    'filter_options': ['rp', 'ip', 'gp', 'zs'],
                    'minimum': VariableStar.MINIMUM_NUMBER_OF_INPUTS,
                    'maximum': VariableStar.MAXIMUM_NUMBER_OF_INPUTS,
                }
            }
        }
        return description

    def operate(self, submitter: User):
        source = self.input_data.get("source")
        target_ra = source.get("ra")
        target_dec = source.get("dec")
        input_files = self._validate_inputs(
            input_key='input_files',
            minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS
        )
        comment= f'Datalab Variable Star analysis for target ra {target_ra}, dec {target_dec} on images: {", ".join([image["basename"] for image in input_files])}'
        log.info(comment)

        light_curve_result = light_curve(
            self.input_data,
            submitter,
            allow_flux_fallback=False,
            progress_callback=lambda progress: self.set_operation_progress(
                VariableStar.PROGRESS_STEPS['DOWNLOADING_INPUT_FILES'] * progress
            )
        )
        light_curve_data = light_curve_result['light_curve']
        excluded_images = light_curve_result['excluded_images']
        flux_fallback = light_curve_result['flux_fallback']
        filter = input_files[0].get('filter', input_files[0].get('primary_optical_element', 'None'))
        
        if len(light_curve_data) < VariableStar.MINIMUM_NUMBER_OF_INPUTS:
            raise ClientAlertException(f'Only found extractable photometry data in {len(light_curve_data)} inputs - this is not enough to get a period. Try increasing your date range to get more input images and try again.')
        
        try:
            frequency, power, period, fap = calculate_period(light_curve_data)
        except Exception as e:
            log.error(f"Error calculating period: {e}")
            period, fap = 0, 0

        self.set_operation_progress(VariableStar.PROGRESS_STEPS['PERIOD_FOLDING_DONE'])

        output = {
            'output_data': [
                {
                    'light_curve': light_curve_data,
                    'flux_fallback': flux_fallback,
                    'excluded_images': excluded_images,
                    'source': source,
                    'filter': filter,
                    'period': period,
                    'fap': fap,
                    'frequency': frequency,
                    'power': power
                }
            ]
        }
        log.info(f'Variable Star output found best period of {period} with false alarm prob. of {fap} using {len(light_curve_data)} / {len(input_files)} images')

        self.set_output(output, is_raw=True)
        self.set_operation_progress(VariableStar.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')

def calculate_period(light_curve):
  """
  Use the astropy lomb scargle to perform the periodogram analysis on the light curve
  """
  ls = LombScargle(
    [lc['julian_date'] for lc in light_curve],
    [lc['mag'] for lc in light_curve],
    [lc['magerr'] for lc in light_curve]
  )

  frequency, power = ls.autopower()

  # Find the best frequency
  best_frequency = frequency[np.argmax(power)]
  period = 1 / best_frequency

  fap = ls.false_alarm_probability(power.max())

  log.info(f"Best period found: {period} days with FAP: {fap}")
  return frequency, power, period, fap
