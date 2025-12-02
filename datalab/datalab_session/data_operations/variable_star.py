import logging

import numpy as np
from astropy.timeseries import LombScargle
from astropy.time import Time
import numpy as np
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import get_hdu
from datalab.datalab_session.utils.filecache import FileCache

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

        light_curve = []
        excluded_images = []
        flux_fallback = False

        for index, image in enumerate(input_files, start = 1):
            basename = image.get('basename')
            filter = image.get('filter', image.get('primary_optical_element', 'None'))
            try:
                file_path = FileCache().get_fits(basename, image.get('source', 'archive'), submitter)
                cat_hdu = get_hdu(file_path, extension='CAT')
            except Exception as e:
                log.error(f"Error retrieving catalog for image {basename}: {e}")
                excluded_images.append(basename)
                continue

            self.set_operation_progress(VariableStar.PROGRESS_STEPS['DOWNLOADING_INPUT_FILES'] * (index / len(input_files)))

            target_source = find_target_source(cat_hdu, target_ra, target_dec)

            if target_source is None:
                log.info(f"No source found matching coordinates: RA={target_ra}, DEC={target_dec} in image {basename}")
                excluded_images.append(basename)
                continue

            try:
                mag = target_source['mag']
                magerr = target_source['magerr']
            except Exception as e:
                log.warning(f"No calibrated magnitude or magnitude error for source in image {basename}")
                excluded_images.append(basename)
                continue

            light_curve.append({
            'mag': mag,
            'magerr': magerr,
            'julian_date': Time(image.get("observation_date")).jd,
            'observation_date': image.get("observation_date")
            })
        
        if len(light_curve) < VariableStar.MINIMUM_NUMBER_OF_INPUTS:
            raise ClientAlertException(f'Only found extractable photometry data in {len(light_curve)} inputs - this is not enough to get a period. Try increasing your date range to get more input images and try again.')
        
        try:
            frequency, power, period, fap = calculate_period(light_curve)
        except Exception as e:
            log.error(f"Error calculating period: {e}")
            period, fap = 0, 0

        self.set_operation_progress(VariableStar.PROGRESS_STEPS['PERIOD_FOLDING_DONE'])

        output = {
            'output_data': [
                {
                    'light_curve': light_curve,
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
        log.info(f'Variable Star output found best period of {period} with false alarm prob. of {fap} using {len(light_curve)} / {len(input_files)} images')

        self.set_output(output, is_raw=True)
        self.set_operation_progress(VariableStar.PROGRESS_STEPS['OUTPUT_PERCENTAGE_COMPLETION'])
        self.set_status('COMPLETED')


def find_target_source(cat_hdu, target_ra, target_dec):
  """
  Find the source in the catalog relative to the target coordinates.
  """
  cat_data = cat_hdu.data
  MATCH_PRECISION = 0.001

  if 'ra' not in cat_data.names or 'dec' not in cat_data.names:
    log.warning("CAT data does not have ra or dec names!")
    return None
  for source in cat_data:
    target_ra = float(target_ra)
    target_dec = float(target_dec)

    if abs(source['ra'] - target_ra) <= MATCH_PRECISION and abs(source['dec'] - target_dec) <= MATCH_PRECISION:
      return source

  return None


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
