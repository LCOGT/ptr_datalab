import logging

from django.contrib.auth.models import User
from astropy.timeseries import LombScargle
import numpy as np

from datalab.datalab_session.data_operations.light_curve import light_curve

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def variable_star(input: dict, user: User):
  """
  Function to perform variable star analysis on a given image.

  input (dict): Input dictionary containing 
    target_coords(dict): ra,dec coordinates for target star
    images(list):
      image(dict):
        basename(str): Name of the image file to be analyzed
        id(str): Unique identifier for the image
        observation_date(str): Date of the observation
  """

  light_curve_result = light_curve(input, user)
  light_curve_data = light_curve_result["light_curve"]

  try:
    frequency, power, period, fap = calculate_period(light_curve_data)
  except Exception as e:
    log.error(f"Error calculating period: {e}")
    frequency, power, period, fap = [], [], 0, 0

  return {
    **light_curve_result,
    "period": period,
    "fap": fap,
    "frequency": frequency,
    "power": power,
  }

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
