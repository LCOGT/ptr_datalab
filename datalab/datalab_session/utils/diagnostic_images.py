import logging
from typing import Callable, Mapping

from datalab.datalab_session.utils.file_utils import temp_file_manager
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.s3_utils import save_files_to_s3


log = logging.getLogger()
log.setLevel(logging.INFO)


def save_diagnostic_images_to_s3(
    *,
    cache_key: str,
    temp_dir: str,
    diagnostic_image_jpegs_by_fits_basename: Mapping[str, bytes],
    on_progress: Callable[[float], None] | None = None,
) -> dict[str, str]:
    """
        Uploads each frame's diagnostic overlay JPEG to the operation bucket.

        Returns a dict mapping FITS basename to the presigned bucket url for its overlay. Takes the
        cache key and temp directory rather than an operation instance so every photometry operation
        can share one implementation regardless of how it reports progress; on_progress, if given,
        receives the completed fraction after each upload.
    """
    diagnostic_image_urls: dict[str, str] = {}
    total = len(diagnostic_image_jpegs_by_fits_basename)
    for index, (fits_basename, jpeg_bytes) in enumerate(diagnostic_image_jpegs_by_fits_basename.items(), start=1):
        with temp_file_manager(f'{cache_key}-{index}-diagnostic.jpg', dir=temp_dir) as jpeg_path:
            with open(jpeg_path, 'wb') as jpeg_file:
                jpeg_file.write(jpeg_bytes)
            s3_output = save_files_to_s3(cache_key, Format.IMAGE, {'diagnostic_jpg_path': jpeg_path}, index=index)
        diagnostic_image_urls[fits_basename] = s3_output['diagnostic_url']
        if on_progress is not None and total:
            on_progress(index / total)
    return diagnostic_image_urls
