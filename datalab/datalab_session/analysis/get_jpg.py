import base64
from datalab.datalab_session.utils.file_utils import create_jpgs
from datalab.datalab_session.utils.s3_utils import get_fits

def get_jpg(input: dict):
  """
    Generates a new jpg file and returns the image
    input: dict
      basename: str
      zmin: int
      zmax: int
  """

  basename = input["basename"]
  zmin = input["zmin"]
  zmax = input["zmax"]

  fits_path = get_fits(basename)

  with create_jpgs(basename, fits_path, zmin=zmin, zmax=zmax) as (large_jpg_path, small_jpg_path):
    print('Created scaled jpg with zmin:', zmin, 'zmax:', zmax)
    with open(large_jpg_path, "rb") as img_file:
      img_data = img_file.read()
  
  # Encode image in Base64
  img_base64 = base64.b64encode(img_data).decode("utf-8")

  print("Returning image")

  return {"jpg_base64": img_base64}
