from abc import ABC, abstractmethod
import hashlib
import json
import tempfile

from django.core.cache import cache
from fits2image.conversions import fits_to_jpg
from astropy.io import fits
import numpy as np

from datalab.datalab_session.tasks import execute_data_operation
from datalab.datalab_session.util import add_file_to_bucket, get_archive_from_basename

CACHE_DURATION = 60 * 60 * 24 * 30  # cache for 30 days


class BaseDataOperation(ABC):

    def __init__(self, input_data: dict = None):
        """ The data inputs are passed in in the format described from the wizard_description """
        self.input_data = self._normalize_input_data(input_data)
        self.cache_key = self.generate_cache_key()

    def _normalize_input_data(self, input_data):
        if input_data == None:
            return {}
        input_schema = self.wizard_description().get('inputs', {})
        for key, value in input_data.items():
            if input_schema.get(key, {}).get('type', '') == 'file' and type(value) is list:
                # If there are file type inputs with multiple files, sort them by basename since order doesn't matter
                value.sort(key=lambda x: x['basename'])
        return input_data

    @staticmethod
    @abstractmethod
    def name():
        """ A unique name for your DataOperation """

    @staticmethod
    @abstractmethod
    def description():
        """ A text description of the DataOperation, to be shown to the user """

    @staticmethod
    @abstractmethod
    def wizard_description():
        """ A json-formatted DSL describing the expected inputs for this DataOperation,
            for the frontend to create custom input widgets for it in a wizard
        """

    @abstractmethod
    def operate(self):
        """ The method that performs the data operation.
            It should periodically update the percent completion during its operation.
            It should set the output and status into the cache when done.
        """

    def perform_operation(self):
        """ The generic method to perform perform the operation if its not in progress """
        status = self.get_status()
        if status == 'PENDING':
            self.set_status('IN_PROGRESS')
            self.set_percent_completion(0.0)
            # This asynchronous task will call the operate() method on the proper operation
            execute_data_operation.send(self.name(), self.input_data)

    def generate_cache_key(self) -> str:
        """ Generate a unique cache key hashed from the input_data and operation name """
        string_key = f'{self.name()}_{json.dumps(sorted(self.input_data.items()))}'
        return hashlib.sha256(string_key.encode('utf-8')).hexdigest()

    def set_status(self, status: str):
        cache.set(f'operation_{self.cache_key}_status', status, CACHE_DURATION)

    def get_status(self) -> str:
        return cache.get(f'operation_{self.cache_key}_status', 'PENDING')

    def set_message(self, message: str):
        cache.set(f'operation_{self.cache_key}_message', message, CACHE_DURATION)

    def get_message(self) -> str:
        return cache.get(f'operation_{self.cache_key}_message', '')

    def set_percent_completion(self, percent_completed: float):
        cache.set(f'operation_{self.cache_key}_percent_completion', percent_completed, CACHE_DURATION)

    def get_percent_completion(self) -> float:
        return cache.get(f'operation_{self.cache_key}_percent_completion', 0.0)

    def set_output(self, output_data: dict):
        self.set_status('COMPLETED')
        self.set_percent_completion(1.0)
        cache.set(f'operation_{self.cache_key}_output', output_data, CACHE_DURATION)

    def get_output(self) -> dict:
        return cache.get(f'operation_{self.cache_key}_output')
    
    def set_failed(self, message: str):
        self.set_status('FAILED')
        self.set_message(message)

    # percent lets you allocate a fraction of the operation that this takes up in time
    # cur_percent is the current completion of the operation
    def create_and_store_fits(self, hdu_list: fits.HDUList, percent=None, cur_percent=None) -> list:
        if not type(hdu_list) == list:
            hdu_list = [hdu_list]

        output = []
        total_files = len(hdu_list)

        # Create temp file paths for storing the products
        fits_path           = tempfile.NamedTemporaryFile(suffix=f'{self.cache_key}.fits').name
        large_jpg_path      = tempfile.NamedTemporaryFile(suffix=f'{self.cache_key}-large.jpg').name
        thumbnail_jpg_path  = tempfile.NamedTemporaryFile(suffix=f'{self.cache_key}-small.jpg').name

        for index, hdu in enumerate(hdu_list, start=1):
            height, width = hdu[1].shape

            hdu.writeto(fits_path)
            fits_to_jpg(fits_path, large_jpg_path, width=width, height=height)
            fits_to_jpg(fits_path, thumbnail_jpg_path)

            # Save Fits and Thumbnails in S3 Buckets
            fits_url            = add_file_to_bucket(f'{self.cache_key}/{self.cache_key}-{index}.fits', fits_path)
            large_jpg_url       = add_file_to_bucket(f'{self.cache_key}/{self.cache_key}-{index}-large.jpg', large_jpg_path)
            thumbnail_jpg_url   = add_file_to_bucket(f'{self.cache_key}/{self.cache_key}-{index}-small.jpg', thumbnail_jpg_path)
            
            output.append({
                'large_url': large_jpg_url,
                'thumbnail_url': thumbnail_jpg_url,
                'basename': f'{self.cache_key}-{index}',
                'source': 'datalab'}
            )

            if percent is not None and cur_percent is not None:
                self.set_percent_completion(cur_percent + index/total_files * percent)
        
        return output

    def get_fits_npdata(self, input_files: list[dict], percent=None, cur_percent=None) -> list[np.memmap]:
        total_files = len(input_files)
        image_data_list = []

        # get the fits urls and extract the image data
        for index, file_info in enumerate(input_files, start=1):
            basename = file_info.get('basename', 'No basename found')
            archive_record = get_archive_from_basename(basename)

            try:
                fits_url = archive_record[0].get('url', 'No URL found')
            except IndexError as e:
                raise FileNotFoundError(f"No image found with specified basename: {basename} Error: {e}")

            with fits.open(fits_url) as hdu_list:
                data = hdu_list['SCI'].data
                image_data_list.append(data)
            
            if percent is not None and cur_percent is not None:
                self.set_percent_completion(cur_percent + index/total_files * percent)

        return image_data_list
