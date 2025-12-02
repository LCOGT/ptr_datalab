import logging

from django.contrib.auth.models import User
from astropy.timeseries import LombScargle
from astropy.time import Time
import numpy as np

from datalab.datalab_session.utils.file_utils import get_hdu
from datalab.datalab_session.utils.filecache import FileCache

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

  coords = input.get("target_coords")
  target_ra = coords.get("ra")
  target_dec = coords.get("dec")

  light_curve = []
  excluded_images = []
  flux_fallback = False

  # Loop through each image's catalog and extract the target source's mag/magerr for the light curve
  for image in input.get("images", []):
    basename = image.get('basename')

    try:
      file_path = FileCache().get_fits(basename, input.get('source', 'archive'), user)
      cat_hdu = get_hdu(file_path, extension='CAT')
    except Exception as e:
      log.error(f"Error retrieving catalog for image {basename}: {e}")
      excluded_images.append(basename)
      continue

    target_source = find_target_source(cat_hdu, target_ra, target_dec)

    if target_source is None:
      log.info(f"No source found matching target coordinates: RA={target_ra}, DEC={target_dec} in image {basename}")
      excluded_images.append(basename)
      continue

    try:
      mag = target_source['mag']
      magerr = target_source['magerr']
    except KeyError as e:
      # If mag or magerr is not present, fallback convert flux to mag
      mag, magerr = flux_to_mag(target_source['flux'], target_source['fluxerr'])
      flux_fallback = True
    except Exception as e:
      log.warning(f"Invalid magnitude or magnitude error for target in image {basename}")
      excluded_images.append(basename)
      continue

    light_curve.append({
      'mag': mag,
      'magerr': magerr,
      'julian_date': Time(image.get("observation_date")).jd,
      'observation_date': image.get("observation_date")
    })
  
  try:
    frequency, power, period, fap = calculate_period(light_curve)
  except Exception as e:
    log.error(f"Error calculating period: {e}")
    period, fap = 0, 0

  return {
    'target_coords': coords,
    'light_curve': light_curve,
    'flux_fallback': flux_fallback,
    'excluded_images': excluded_images,
    'period': period,
    'fap': fap,
    'frequency': frequency,
    'power': power
  }

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

def flux_to_mag(flux, fluxerr):
  """
  Convert flux and fluxerr to magnitude and magnitude error.
  """
  CONVERSION_FACTOR = 2.5
  FLUX2MAG = CONVERSION_FACTOR / np.log(10)

  if flux <= 0:
    return None, None
  
  mag = -CONVERSION_FACTOR * np.log10(flux)
  magerr = FLUX2MAG * (fluxerr / flux)
  
  return mag, magerr

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
