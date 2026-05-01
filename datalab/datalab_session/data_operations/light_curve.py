import logging

from astropy.time import Time
from django.contrib.auth.models import User

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.file_utils import get_hdu
from datalab.datalab_session.utils.filecache import FileCache
from datalab.datalab_session.utils.flux_to_mag import flux_to_mag

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class LightCurve(BaseDataOperation):
  MINIMUM_NUMBER_OF_INPUTS = 1
  MAXIMUM_NUMBER_OF_INPUTS = 999
  PROGRESS_STEPS = {
    "LIGHT_CURVE_DONE": 1.0,
  }

  @staticmethod
  def name():
    return "Light Curve"

  @staticmethod
  def description():
    return "Builds a light curve for a source from calibrated catalog photometry across input images."

  @staticmethod
  def wizard_description():
    return {
      "name": LightCurve.name(),
      "description": LightCurve.description(),
      "category": "image",
      "inputs": {
        "source": {
          "name": "Source Star",
          "type": Format.SOURCE,
          "description": "The source star to analyze",
          "name_lookup": True,
        },
        "input_files": {
          "name": "Input Files",
          "description": "The input files to pull photometry from",
          "type": Format.FITS,
          "single_filter": True,
          "filter_options": ["rp", "ip", "gp", "zs"],
          "minimum": LightCurve.MINIMUM_NUMBER_OF_INPUTS,
          "maximum": LightCurve.MAXIMUM_NUMBER_OF_INPUTS,
        },
      },
    }

  def operate(self, submitter: User):
    input_files = self._validate_inputs(
      input_key="input_files",
      minimum_inputs=self.MINIMUM_NUMBER_OF_INPUTS,
    )
    source = self.input_data.get("source")
    if not source:
      raise ClientAlertException(f"Operation {self.name()} requires a source.")

    output = light_curve(
      self.input_data,
      submitter,
      allow_flux_fallback=False,
      progress_callback=self.set_operation_progress,
    )
    if not output["light_curve"]:
      raise ClientAlertException("No extractable photometry data found for this source.")

    output["source"] = source
    output["filter"] = input_files[0].get("filter", input_files[0].get("primary_optical_element", "None"))

    self.set_output({"output_data": [output]}, is_raw=True)
    self.set_operation_progress(LightCurve.PROGRESS_STEPS["LIGHT_CURVE_DONE"])
    self.set_status("COMPLETED")


def light_curve(input: dict, user: User, allow_flux_fallback=True, progress_callback=None):
  """
  Build a light curve from a target coordinate and a set of image catalogs.
  """
  coords = input.get("target_coords") or input.get("source")
  target_ra = coords.get("ra")
  target_dec = coords.get("dec")
  images = input.get("images") or input.get("input_files") or []

  light_curve_data = []
  excluded_images = []
  flux_fallback = False

  for index, image in enumerate(images, start=1):
    basename = image.get("basename")
    image_source = image.get("source")
    if image_source is None and isinstance(input.get("source"), str):
      image_source = input.get("source")
    if image_source is None:
      image_source = "archive"

    try:
      file_path = FileCache().get_fits(basename, image_source, user)
      cat_hdu = get_hdu(file_path, extension="CAT")
    except Exception as e:
      log.error(f"Error retrieving catalog for image {basename}: {e}")
      excluded_images.append(basename)
      continue

    if progress_callback:
      progress_callback(index / len(images))

    target_source = find_target_source(cat_hdu, target_ra, target_dec)
    if target_source is None:
      log.info(f"No source found matching target coordinates: RA={target_ra}, DEC={target_dec} in image {basename}")
      excluded_images.append(basename)
      continue

    try:
      mag = target_source["mag"]
      magerr = target_source["magerr"]
    except KeyError:
      if not allow_flux_fallback:
        log.warning(f"No calibrated magnitude or magnitude error for source in image {basename}")
        excluded_images.append(basename)
        continue
      mag, magerr = flux_to_mag(target_source["flux"], target_source["fluxerr"])
      flux_fallback = True
    except Exception:
      log.warning(f"Invalid magnitude or magnitude error for target in image {basename}")
      excluded_images.append(basename)
      continue

    light_curve_data.append({
      "mag": mag,
      "magerr": magerr,
      "julian_date": Time(image.get("observation_date")).jd,
      "observation_date": image.get("observation_date"),
    })

  return {
    "target_coords": coords,
    "light_curve": light_curve_data,
    "flux_fallback": flux_fallback,
    "excluded_images": excluded_images,
  }


def find_target_source(cat_hdu, target_ra, target_dec):
  """
  Find the source in the catalog relative to the target coordinates.
  """
  cat_data = cat_hdu.data
  match_precision = 0.001

  if "ra" not in cat_data.names or "dec" not in cat_data.names:
    log.warning("CAT data does not have ra or dec names!")
    return None

  target_ra = float(target_ra)
  target_dec = float(target_dec)

  for source in cat_data:
    if abs(source["ra"] - target_ra) <= match_precision and abs(source["dec"] - target_dec) <= match_precision:
      return source

  return None
