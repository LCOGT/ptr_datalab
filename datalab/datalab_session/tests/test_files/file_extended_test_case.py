import pathlib as pl
from hashlib import md5
from os import path, remove, listdir

from django.test import TestCase

# extending the TestCase class to include a custom assertions for file operations
class FileExtendedTestCase(TestCase):
    def assertIsFile(self, path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))
    
    def assertFilesEqual(self, image_1: str, image_2: str):
        with open(image_1, 'rb') as file_1, open(image_2, 'rb') as file_2:
            self.assertEqual(md5(file_1.read()).hexdigest(), md5(file_2.read()).hexdigest())

    def clean_test_dir(self):
        test_files_dir = 'datalab/datalab_session/tests/test_files'
        for file_name in listdir(test_files_dir):
            if file_name.startswith('temp_'):
                file_path = path.join(test_files_dir, file_name)
                if path.isfile(file_path):
                    remove(file_path)
