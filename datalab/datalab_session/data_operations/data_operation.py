from abc import ABC, abstractmethod
import hashlib
import json

from django.core.cache import cache
import numpy as np

from datalab.datalab_session.utils.s3_utils import get_fits
from datalab.datalab_session.tasks import execute_data_operation
from datalab.datalab_session.utils.file_utils import get_hdu

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
        """ The generic method to perform the operation if its not in progress """
        status = self.get_status()
        if status == 'PENDING' or status == 'FAILED':
            self.set_status('IN_PROGRESS')
            self.set_operation_progress(0.0)
            # This asynchronous task will call the operate() method on the proper operation
            execute_data_operation.send(self.name(), self.input_data)

    def generate_cache_key(self) -> str:
        """ Generate a unique cache key hashed from the input_data and operation name """
        string_key = f'{self.name()}_{json.dumps(sorted(self.input_data.items()), sort_keys=True)}'
        return hashlib.sha256(string_key.encode('utf-8')).hexdigest()

    def set_status(self, status: str):
        cache.set(f'operation_{self.cache_key}_status', status, CACHE_DURATION)

    def get_status(self) -> str:
        return cache.get(f'operation_{self.cache_key}_status', 'PENDING')

    def set_message(self, message: str):
        cache.set(f'operation_{self.cache_key}_message', message, CACHE_DURATION)

    def get_message(self) -> str:
        return cache.get(f'operation_{self.cache_key}_message', '')

    def set_operation_progress(self, percent_completed: float):
        cache.set(f'operation_{self.cache_key}_progress', percent_completed, CACHE_DURATION)

    def get_operation_progress(self) -> float:
        return cache.get(f'operation_{self.cache_key}_progress', 0.0)

    def set_output(self, output):
        output_data = {'output_files': output if isinstance(output, list) else [output]}
        self.set_status('COMPLETED')
        self.set_operation_progress(1.0)
        cache.set(f'operation_{self.cache_key}_output', output_data, CACHE_DURATION)

    def get_output(self) -> dict:
        return cache.get(f'operation_{self.cache_key}_output')
    
    def set_failed(self, message: str):
        self.set_status('FAILED')
        self.set_message(message)

    def get_fits_npdata(self, input_files: list[dict]) -> list[np.memmap]:
        image_data_list = []

        # get the fits urls and extract the image data
        for index, file_info in enumerate(input_files, start=1):
            basename = file_info.get('basename', 'No basename found')
            source = file_info.get('source', 'No source found')

            fits_path = get_fits(file_info['basename'], file_info['source'])
            sci_hdu = get_hdu(fits_path, 'SCI')
            image_data_list.append(sci_hdu.data)

            self.set_operation_progress(index / len(input_files) * 0.5)

        return image_data_list
