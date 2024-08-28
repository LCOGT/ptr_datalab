from unittest import mock

from django.test import TestCase

class TestAnalysis(TestCase):
    def setUp(self):
        pass

    def test_get_tif(self):
        # TODO use a test fits file to create a tif file

        # TODO mock the get_s3_url call
        # TODO mock the get_fits call
        # TODO mock the add_file_to_bucket call

        # TODO assert tif file exists
        # TODO assert tif file exists pregenerated tif file
        pass

    def test_line_profile(self):
        # TODO use a test fits file that doesn't have a WCS header

        # TODO assert that we throw a WCS error if the header is invalid
        pass

    def test_source_catalog(self):
        # TODO use a test fits file 
        pass
