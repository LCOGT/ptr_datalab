import base64
from datalab.datalab_session.utils.file_utils import create_jpgs, temp_file_manager
from datalab.datalab_session.utils.filecache import FileCache
from django.contrib.auth.models import User
from datalab.datalab_session.exceptions import ClientAlertException

def get_jpg(input: dict, user: User):
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

  try:
    file_path = FileCache().get_fits(basename, input.get('source', 'archive'), user)
  except TimeoutError as e:
    raise ClientAlertException(f"Download of {basename} timed out")

  with temp_file_manager(f'{basename}-large.jpg', f'{basename}-small.jpg') as (large_jpg, small_jpg):
    create_jpgs(file_path, large_jpg, small_jpg, zmin=zmin, zmax=zmax)
    with open(large_jpg, "rb") as img_file:
      img_data = img_file.read()
  
  # Encode image in Base64
  img_base64 = base64.b64encode(img_data).decode("utf-8")

  print("Returning image")

  return {"jpg_base64": img_base64}
