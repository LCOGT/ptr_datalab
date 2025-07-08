import logging

from django.contrib.auth.models import User
import numpy as np

from datalab.datalab_session.utils.file_utils import get_hdu
from datalab.datalab_session.utils.filecache import FileCache

log = logging.getLogger()
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

  # Loop through each image's catalog and extract the target source's mag/magerr for the light curve
  for image in input.get("images"):
    basename = image.get('basename')

    try:
      file_path = FileCache().get_fits(basename, input.get('source', 'archive'), user)
      cat_hdu = get_hdu(file_path, extension='CAT')
    except Exception as e:
      log.error(f"Error retrieving catalog for image {basename}: {e}")
      continue
    
    target_source = find_target_source(cat_hdu, target_ra, target_dec)

    if target_source is None:
      log.info(f"No matching source found for target coordinates: RA={target_ra}, DEC={target_dec} in image {basename}")
      continue

    # Fallback calculating mag/magerr from flux/fluxerr if not in catalog columns
    if(not 'mag' in target_source or not 'magerr' in target_source):
      target_source['mag'], target_source['magerr'] = flux_to_mag(target_source['flux'], target_source['fluxerr'])

    if target_source['mag'] is None or target_source['magerr'] is None:
      log.warning(f"Invalid magnitude or magnitude error for target source in image {basename}. Skipping this source.")
      continue

    light_curve.append({
      'mag': target_source['mag'],
      'magerr': target_source['magerr'],
      'observation_date': image.get("observation_date"),
    })

  return {
    'target_coords': coords,
    'light_curve': light_curve
  }

def find_target_source(cat_hdu, target_ra, target_dec):
  """
  Find the source in the catalog relative to the target coordinates.
  """
  cat_data = cat_hdu.data
  MATCH_PRECISION = 0.001

  for source in cat_data:
    target_ra = float(target_ra)
    target_dec = float(target_dec)

    if abs(source['ra'] - target_ra) <= MATCH_PRECISION and abs(source['dec'] - target_dec) <= MATCH_PRECISION:
      return source

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
