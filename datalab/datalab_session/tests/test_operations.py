from contextlib import contextmanager
from datetime import datetime, timezone
import shutil
from types import SimpleNamespace
from unittest import mock
import math
import os

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from datalab.datalab_session.data_operations.data_operation import BaseDataOperation
from datalab.datalab_session.data_operations.aperture_photometry import AperturePhotometry
from datalab.datalab_session.data_operations.color_image import Color_Image
from datalab.datalab_session.exceptions import ClientAlertException
from datalab.datalab_session.data_operations.hr_diagram import HRDiagram
from datalab.datalab_session.data_operations.light_curve import LightCurve
from datalab.datalab_session.data_operations.median import Median
from datalab.datalab_session.data_operations.stacking import Stack
from datalab.datalab_session.tests.test_files.file_extended_test_case import FileExtendedTestCase
from datalab.datalab_session.utils.format import Format
from datalab.datalab_session.utils.aperture_light_curve import LightCurveRow

wizard_description = {
            'name': 'SampleDataOperation',
            'description': 'Testing class for DataOperation',
            'category': 'test',
            'inputs': {
                'input_files': {
                    'name': 'Input Files',
                    'description': 'The input files to operate on',
                    'type': Format.FITS,
                    'minimum': 1,
                    'maximum': 999
                },
                'input_number': {
                    'name': 'Input Number',
                    'description': 'The input number to operate on',
                    'type': 'number',
                    'minimum': 1,
                    'maximum': 999
                },
                'input_string': {
                    'name': 'Input String',
                    'description': 'The input string to operate on',
                    'type': 'string',
                    'minimum': 1,
                    'maximum': 999
                }
            }
        }
test_path = 'datalab/datalab_session/tests/test_files/'

class SampleDataOperation(BaseDataOperation):
    
    @staticmethod
    def name():
        return 'SampleDataOperation'
    
    @staticmethod
    def description():
        return 'Testing class for DataOperation'
    
    @staticmethod
    def wizard_description():
        return wizard_description
    
    def operate(self, submitter):
        self.set_output([])
        self.set_operation_progress(1.0)
        self.set_status('COMPLETED')


class TestDataOperation(FileExtendedTestCase):
    def setUp(self):
        input_data = {
            'input_files': [],
            'input_number': 1,
            'input_string': 'test'
        }
        self.data_operation = SampleDataOperation(input_data)

    def test_init(self):
        self.assertEqual(self.data_operation.input_data, {
            'input_files': [],
            'input_number': 1,
            'input_string': 'test'
        })
        self.assertIsInstance(self.data_operation.cache_key, str)
        self.assertGreater(len(self.data_operation.cache_key), 0)
    
    def test_normalize_input_data(self):
        # out of order files
        input_data = {
            'input_files': [
                {'basename': 'file2'},
                {'basename': 'file1'}
            ],
            'input_number': 1,
            'input_string': 'test'
        }
        normalized_input_data = self.data_operation._normalize_input_data(input_data)
        self.assertEqual(normalized_input_data, {
            'input_files': [
                {'basename': 'file1'},
                {'basename': 'file2'}
            ],
            'input_number': 1,
            'input_string': 'test'
        })

        # in order files
        input_data = {
            'input_files': [
                {'basename': 'file1'},
                {'basename': 'file2'}
            ],
            'input_number': 1,
            'input_string': 'test'
        }
        normalized_input_data = self.data_operation._normalize_input_data(input_data)
        self.assertEqual(normalized_input_data, {
            'input_files': [
                {'basename': 'file1'},
                {'basename': 'file2'}
            ],
            'input_number': 1,
            'input_string': 'test'
        })

        # missing files
        input_data = {
            'input_number': 1,
            'input_string': 'test'
        }
        normalized_input_data = self.data_operation._normalize_input_data(input_data)
        self.assertEqual(normalized_input_data, {
            'input_number': 1,
            'input_string': 'test'
        })
      
    def test_name(self):
        self.assertEqual(self.data_operation.name(), 'SampleDataOperation')
    
    def test_description(self):
        self.assertEqual(self.data_operation.description(), 'Testing class for DataOperation')
    
    def test_wizard_description(self):
        self.assertEqual(self.data_operation.wizard_description(), wizard_description)
    
    def test_operate(self):
        self.data_operation.operate(None)
        self.assertEqual(self.data_operation.get_operation_progress(), 1.0)
        self.assertEqual(self.data_operation.get_status(), 'COMPLETED')
        self.assertEqual(self.data_operation.get_output(), {'output_files': []})
    
    def test_generate_cache_key(self):
        pregenerated_cache_key = '9c8d4f7c82d357c95416c43560d0a1f66f1b2e3cb6a39c0c8a004ee162482ea7'
        self.assertEqual(self.data_operation.generate_cache_key(), pregenerated_cache_key)

    def test_set_get_output(self):
        self.data_operation.set_output([])
        self.data_operation.set_operation_progress(1.0)
        self.data_operation.set_status('COMPLETED')
        self.assertEqual(self.data_operation.get_operation_progress(), 1.0)
        self.assertEqual(self.data_operation.get_status(), 'COMPLETED')
        self.assertEqual(self.data_operation.get_output(), {'output_files': []})

    def test_set_failed(self):
        self.data_operation.set_failed('Test message')
        self.assertEqual(self.data_operation.get_status(), 'FAILED')
        self.assertEqual(self.data_operation.get_message(), 'Test message')


class TestMedianOperation(FileExtendedTestCase):
    temp_median_path = f'{test_path}temp_median.fits'
    test_median_path = f'{test_path}median/median_1_2.fits'
    test_fits_1_path = f'{test_path}fits_1.fits.fz'
    test_fits_2_path = f'{test_path}fits_2.fits.fz'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()

    @mock.patch('datalab.datalab_session.utils.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.data_operations.input_data_handler.FileCache')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.FileCache', new=mock.MagicMock)
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.save_files_to_s3')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.create_jpgs')
    def test_operate(self, mock_create_jpgs, mock_save_files_to_s3, mock_file_cache, mock_named_tempfile):
        # return the test fits paths in order of the input_files instead of aws fetch
        mock_fc_instance1 = mock.MagicMock()
        mock_fc_instance1.get_fits.return_value = self.test_fits_1_path
        mock_fc_instance2 = mock.MagicMock()
        mock_fc_instance2.get_fits.return_value = self.test_fits_2_path
        mock_file_cache.side_effect = [mock_fc_instance1, mock_fc_instance2]

        # save temp output to a known path so we can test it
        mock_named_tempfile.return_value.__enter__.return_value.name = self.temp_median_path
        # avoids overwriting our output
        mock_create_jpgs.return_value.__enter__.return_value = ('test_path', 'test_path')
        # don't save to s3
        mock_save_files_to_s3.return_value = self.temp_median_path

        input_data = {
            'input_files': [
                {'basename': 'fits_1','source': 'local'},
                {'basename': 'fits_2','source': 'local'}
            ]
        }

        median = Median(input_data)
        median.operate(None)
        output = median.get_output().get('output_files')

        self.assertEqual(median.get_operation_progress(), 1.0)
        self.assertTrue(os.path.exists(output[0]))
        self.assertFilesEqual(self.test_median_path, output[0])

    def test_not_enough_files(self):
        input_data = {
            'input_files': [
                {'basename': 'sample_lco_fits_1'}
            ]
        }
        median = Median(input_data)
        
        with self.assertRaises(ClientAlertException):
            median.operate(None)


class TestLightCurveOperation(FileExtendedTestCase):

    @mock.patch('datalab.datalab_session.data_operations.light_curve.light_curve')
    def test_operate(self, mock_light_curve):
        mock_light_curve.return_value = {
            'target_coords': {'ra': 10.0, 'dec': 20.0},
            'light_curve': [{
                'mag': 15.2,
                'magerr': 0.03,
                'julian_date': 2460310.5,
                'observation_date': '2024-01-01T00:00:00',
            }],
            'flux_fallback': False,
            'excluded_images': [],
        }
        input_data = {
            'source': {'ra': 10.0, 'dec': 20.0},
            'input_files': [{
                'basename': 'fits_1',
                'source': 'local',
                'filter': 'rp',
                'observation_date': '2024-01-01T00:00:00',
            }],
        }

        light_curve = LightCurve(input_data)
        light_curve.operate(None)
        output = light_curve.get_output()

        self.assertEqual(light_curve.get_operation_progress(), 1.0)
        self.assertEqual(light_curve.get_status(), 'COMPLETED')
        self.assertEqual(output['output_data'][0]['source'], input_data['source'])
        self.assertEqual(output['output_data'][0]['filter'], 'rp')
        self.assertEqual(output['output_data'][0]['light_curve'], mock_light_curve.return_value['light_curve'])

    def test_not_enough_files(self):
        light_curve = LightCurve({
            'source': {'ra': 10.0, 'dec': 20.0},
            'input_files': [],
        })

        with self.assertRaises(ClientAlertException):
            light_curve.operate(None)


class TestAperturePhotometryOperation(FileExtendedTestCase):

    def valid_input_data(self):
        return {
            'source': {'ra': 10.0, 'dec': 20.0},
            'input_files': [{
                'basename': 'fits_1',
                'source': 'local',
                'filter': 'rp',
            }],
            'aperture_radius': 7.64,
            'annulus_inner_radius': 12.73,
            'annulus_outer_radius': 19.10,
        }

    @mock.patch('datalab.datalab_session.data_operations.aperture_photometry.generate_light_curve')
    @mock.patch('datalab.datalab_session.data_operations.aperture_photometry.InputDataHandler')
    @mock.patch.object(AperturePhotometry, 'set_status')
    @mock.patch.object(AperturePhotometry, 'set_output')
    @mock.patch.object(AperturePhotometry, 'set_operation_progress')
    def test_operate_requires_and_passes_explicit_radii_and_filter(
        self,
        mock_set_operation_progress,
        mock_set_output,
        mock_set_status,
        mock_input_data_handler,
        mock_generate_light_curve,
    ):
        input_handler = SimpleNamespace(fits_file='/tmp/fits_1.fits')
        mock_input_data_handler.return_value = input_handler
        mock_generate_light_curve.return_value = SimpleNamespace(
            light_curve_rows=[
                LightCurveRow(
                    fits_path='/tmp/fits_1.fits',
                    date_obs=datetime(2026, 5, 13, tzinfo=timezone.utc),
                    target_centroid_x=1.0,
                    target_centroid_y=2.0,
                    target_net_source_counts=3.0,
                    target_source_uncertainty=4.0,
                    comparison_ensemble_total_counts=5.0,
                    comparison_ensemble_uncertainty=6.0,
                    target_differential_flux=7.0,
                    target_differential_flux_uncertainty=8.0,
                    target_calibrated_apparent_magnitude=float('nan'),
                    target_calibrated_apparent_magnitude_uncertainty=float('nan'),
                )
            ],
            selected_comparison_stars=[],
            diagnostics=('loaded 1 frame', 'selected 5 comparison stars'),
            diagnostics_by_fits_basename={
                'fits_1.fits': ['loaded 1 frame', 'selected 5 comparison stars'],
            },
            diagnostic_images_by_fits_basename={},
        )
        input_data = self.valid_input_data()

        aperture_photometry = AperturePhotometry(input_data)
        aperture_photometry.operate(None)

        mock_generate_light_curve.assert_called_once_with(
            input_handlers=[input_handler],
            target_ra_deg=10.0,
            target_dec_deg=20.0,
            aperture_radius=7.64,
            annulus_inner_radius=12.73,
            annulus_outer_radius=19.10,
            min_comparisons=5,
            max_comparisons=10,
        )
        output, is_raw = mock_set_output.call_args.args[0], mock_set_output.call_args.kwargs['is_raw']
        self.assertTrue(is_raw)
        self.assertEqual(output['output_data'][0]['filter'], 'rp')
        self.assertEqual(
            output['output_data'][0]['diagnostics'],
            {
                'fits_1.fits': ['loaded 1 frame', 'selected 5 comparison stars'],
            },
        )
        self.assertTrue(math.isnan(output['output_data'][0]['light_curve'][0]['target_calibrated_apparent_magnitude']))
        self.assertTrue(math.isnan(
            output['output_data'][0]['light_curve'][0]['target_calibrated_apparent_magnitude_uncertainty']
        ))
        mock_set_status.assert_called_once_with('COMPLETED')

    def test_operate_requires_aperture_radius(self):
        input_data = self.valid_input_data()
        del input_data['aperture_radius']

        with self.assertRaisesRegex(ClientAlertException, 'received invalid input'):
            AperturePhotometry(input_data).operate(None)

    def test_operate_requires_annulus_inner_radius(self):
        input_data = self.valid_input_data()
        del input_data['annulus_inner_radius']

        with self.assertRaisesRegex(ClientAlertException, 'received invalid input'):
            AperturePhotometry(input_data).operate(None)

    def test_operate_requires_annulus_outer_radius(self):
        input_data = self.valid_input_data()
        del input_data['annulus_outer_radius']

        with self.assertRaisesRegex(ClientAlertException, 'received invalid input'):
            AperturePhotometry(input_data).operate(None)

    def test_operate_allows_missing_filter(self):
        input_data = self.valid_input_data()
        del input_data['input_files'][0]['filter']

        with mock.patch('datalab.datalab_session.data_operations.aperture_photometry.generate_light_curve') as mock_generate_light_curve, \
                mock.patch('datalab.datalab_session.data_operations.aperture_photometry.InputDataHandler') as mock_input_data_handler, \
                mock.patch.object(AperturePhotometry, 'set_output') as mock_set_output, \
                mock.patch.object(AperturePhotometry, 'set_operation_progress'), \
                mock.patch.object(AperturePhotometry, 'set_status'):
            mock_input_data_handler.return_value = SimpleNamespace(fits_file='/tmp/fits_1.fits')
            mock_generate_light_curve.return_value = SimpleNamespace(
                light_curve_rows=[],
                selected_comparison_stars=[],
                diagnostics=[],
                diagnostics_by_fits_basename={},
                diagnostic_images_by_fits_basename={},
            )

            AperturePhotometry(input_data).operate(None)

        output = mock_set_output.call_args.args[0]
        self.assertEqual(output['output_data'][0]['filter'], 'None')


class TestColorImageOperation(FileExtendedTestCase):
    temp_color_path = f'{test_path}temp_color.fits'
    test_color_path = f'{test_path}color_image/color_image.fits'
    test_red_path = f'{test_path}color_image/red.fits'
    test_green_path = f'{test_path}color_image/green.fits'
    test_blue_path = f'{test_path}color_image/blue.fits'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()
    
    @mock.patch('datalab.datalab_session.data_operations.color_image.save_files_to_s3')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.create_jpgs')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.data_operations.input_data_handler.FileCache')
    def test_operate(self, mock_file_cache, mock_named_tempfile, mock_create_jpgs, mock_save_files_to_s3):
        # return the test fits paths in order of the input_files instead of aws fetch
        mock_fc_instance1 = mock.MagicMock()
        mock_fc_instance1.get_fits.return_value = self.test_red_path
        mock_fc_instance2 = mock.MagicMock()
        mock_fc_instance2.get_fits.return_value = self.test_green_path
        mock_fc_instance3 = mock.MagicMock()
        mock_fc_instance3.get_fits.return_value = self.test_blue_path
        mock_file_cache.side_effect = [mock_fc_instance1, mock_fc_instance2, mock_fc_instance3]

        # save temp output to a known path so we can test
        mock_named_tempfile.return_value.__enter__.return_value.name = self.temp_color_path
        # avoids overwriting our output
        mock_create_jpgs.return_value.__enter__.return_value = ('test_path', 'test_path')
        # don't save to s3
        mock_save_files_to_s3.return_value = self.temp_color_path

        input_data = {
            'color_channels': [
                {'basename': 'red_fits', 'source': 'local', 'zmin': 0, 'zmax': 255, 'color': {'r': 255, 'g': 0, 'b': 0}},
                {'basename': 'green_fits', 'source': 'local', 'zmin': 0, 'zmax': 255, 'color': {'r': 0, 'g': 255, 'b': 0}},
                {'basename': 'blue_fits', 'source': 'local', 'zmin': 0, 'zmax': 255, 'color': {'r': 0, 'g': 0, 'b': 255}}
            ]
        }

        color_image = Color_Image(input_data)
        color_image.operate(None)
        output = color_image.get_output().get('output_files')

        self.assertEqual(color_image.get_operation_progress(), 1.0)
        self.assertEqual(output, [self.temp_color_path])


class TestHRDiagramOperation(FileExtendedTestCase):
    temp_blue_path = f'{test_path}temp_hr_blue.fits'
    temp_red_path = f'{test_path}temp_hr_red.fits'

    CENTER_RA = 150.0
    CENTER_DEC = 30.0

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()

    @staticmethod
    def create_test_fits(path, center_ra, center_dec, ra, dec, mag, magerr, include_radec=True, include_mag=True):
        """ Writes a fits file with a WCS'd SCI HDU and a CAT HDU like a reduced image's catalog """
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [center_ra, center_dec]
        wcs.wcs.crpix = [100.0, 100.0]
        wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
        sci_hdu = fits.ImageHDU(data=np.zeros((200, 200), dtype=np.float32), header=wcs.to_header(), name='SCI')

        x, y = wcs.all_world2pix(np.asarray(ra), np.asarray(dec), 1)
        columns = [
            fits.Column(name='x', format='D', array=x),
            fits.Column(name='y', format='D', array=y),
        ]
        if include_radec:
            columns += [
                fits.Column(name='ra', format='D', array=np.asarray(ra)),
                fits.Column(name='dec', format='D', array=np.asarray(dec)),
            ]
        if include_mag:
            columns += [
                fits.Column(name='mag', format='D', array=np.asarray(mag)),
                fits.Column(name='magerr', format='D', array=np.asarray(magerr)),
            ]
        cat_hdu = fits.BinTableHDU.from_columns(columns, name='CAT')
        fits.HDUList([fits.PrimaryHDU(), sci_hdu, cat_hdu]).writeto(path, overwrite=True)

    def hr_input_data(self, search_radius_arcmin=None):
        input_data = {
            'blue_filter_files': [{'basename': 'blue_fits', 'source': 'local', 'filter': 'gp'}],
            'red_filter_files': [{'basename': 'red_fits', 'source': 'local', 'filter': 'rp'}],
            'cluster': {'name': 'Test Cluster', 'ra': self.CENTER_RA, 'dec': self.CENTER_DEC},
        }
        if search_radius_arcmin is not None:
            input_data['search_radius_arcmin'] = search_radius_arcmin
        return input_data

    def mock_band_fits(self, mock_file_cache):
        """ FileCache is instantiated once per band, blue first """
        mock_fc_blue = mock.MagicMock()
        mock_fc_blue.get_fits.return_value = self.temp_blue_path
        mock_fc_red = mock.MagicMock()
        mock_fc_red.get_fits.return_value = self.temp_red_path
        mock_file_cache.side_effect = [mock_fc_blue, mock_fc_red]

    @classmethod
    def empty_gaia_data(cls):
        return cls.gaia_field(ra=[], dec=[], pmra=[], pmdec=[], parallax=[])

    @staticmethod
    def gaia_field(ra, dec, pmra, pmdec, parallax, g_mag=None, synth_g=None, synth_r=None, distance=None):
        """ Builds a gaia_cone_search-shaped dict; synthetic i/z bands mirror r, and the
        Bailer-Jones distance bounds bracket r_med_geo asymmetrically """
        n = len(ra)
        data = {
            'ra': np.asarray(ra, dtype=float),
            'dec': np.asarray(dec, dtype=float),
            'pmra': np.asarray(pmra, dtype=float),
            'pmra_error': np.full(n, 0.05),
            'pmdec': np.asarray(pmdec, dtype=float),
            'pmdec_error': np.full(n, 0.06),
            'parallax': np.asarray(parallax, dtype=float),
            'parallax_error': np.full(n, 0.04),
            'phot_g_mean_mag': np.full(n, 14.0) if g_mag is None else np.asarray(g_mag, dtype=float),
        }
        distance = np.full(n, 800.0) if distance is None else np.asarray(distance, dtype=float)
        data['r_med_geo'] = distance
        data['r_lo_geo'] = distance - 50.0
        data['r_hi_geo'] = distance + 60.0
        synth_g = np.full(n, 15.0) if synth_g is None else np.asarray(synth_g, dtype=float)
        synth_r = np.full(n, 14.4) if synth_r is None else np.asarray(synth_r, dtype=float)
        for band, mags in [('g_sdss', synth_g), ('r_sdss', synth_r), ('i_sdss', synth_r), ('z_sdss', synth_r)]:
            data[f'{band}_mag'] = mags
            data[f'{band}_mag_error'] = np.where(np.isfinite(mags), 0.032, np.nan)
        return data

    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.gaia_cone_search')
    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.FileCache')
    def test_operate(self, mock_file_cache, mock_gaia_cone_search):
        # five cluster stars present in both bands, at 10-50 arcsec north of the center
        common_dec = self.CENTER_DEC + np.array([10.0, 20.0, 30.0, 40.0, 50.0]) / 3600.0
        common_ra = np.full(5, self.CENTER_RA)
        blue_common_mag = np.array([15.0, 15.5, 16.0, 16.5, 17.0])
        red_common_mag = np.array([14.5, 14.8, 15.2, 15.6, 16.0])

        # a sixth red star 10 arcsec south with two blue sources nearby (0 and 1 arcsec away),
        # to check one-to-one dedup keeps only the closest pair
        crowded_dec = self.CENTER_DEC - 10.0 / 3600.0
        # plus per-band strays with no counterpart within the match radius
        blue_ra = np.concatenate([common_ra, [self.CENTER_RA, self.CENTER_RA, self.CENTER_RA]])
        blue_dec = np.concatenate([common_dec, [crowded_dec, crowded_dec + 1.0 / 3600.0, self.CENTER_DEC + 80.0 / 3600.0]])
        blue_mag = np.concatenate([blue_common_mag, [18.0, 19.0, 15.5]])
        red_ra = np.concatenate([common_ra, [self.CENTER_RA, self.CENTER_RA]])
        red_dec = np.concatenate([common_dec, [crowded_dec, self.CENTER_DEC - 40.0 / 3600.0]])
        red_mag = np.concatenate([red_common_mag, [17.0, 15.0]])

        self.create_test_fits(self.temp_blue_path, self.CENTER_RA, self.CENTER_DEC,
                              blue_ra, blue_dec, blue_mag, np.full(len(blue_mag), 0.02))
        self.create_test_fits(self.temp_red_path, self.CENTER_RA, self.CENTER_DEC,
                              red_ra, red_dec, red_mag, np.full(len(red_mag), 0.03))
        self.mock_band_fits(mock_file_cache)

        # Gaia knows the four brightest cluster stars, plus a stray with no photometric counterpart
        gaia_dec = np.concatenate([common_dec[:4], [self.CENTER_DEC - 60.0 / 3600.0]])
        mock_gaia_cone_search.return_value = self.gaia_field(
            ra=np.full(5, self.CENTER_RA),
            dec=gaia_dec,
            pmra=[-5.1, -4.9, -5.0, -5.05, 12.0],
            pmdec=[3.0, 3.1, 2.9, 3.02, -8.0],
            parallax=[1.2, 1.3, 1.25, 1.22, 0.1],
            synth_g=[15.0, 15.5, 16.0, 16.5, 16.2],
            synth_r=[14.5, 14.8, 15.2, 15.6, 15.4],
            distance=[810.0, 820.0, 830.0, 840.0, np.nan],
        )

        hr_diagram = HRDiagram(self.hr_input_data())
        hr_diagram.operate(None)
        output = hr_diagram.get_output()['output_data'][0]

        self.assertEqual(hr_diagram.get_status(), 'COMPLETED')
        self.assertEqual(hr_diagram.get_operation_progress(), 1.0)
        self.assertEqual(output['n_stars'], 6)
        self.assertEqual(output['n_stars_matched'], 6)
        self.assertEqual(output['blue_filter'], 'gp')
        self.assertEqual(output['red_filter'], 'rp')
        self.assertEqual(output['mag_band'], 'rp')
        self.assertEqual(output['cluster']['name'], 'Test Cluster')
        mock_gaia_cone_search.assert_called_once_with(self.CENTER_RA, self.CENTER_DEC, 15.0)

        # cmd points are sorted brightest first in the red (y-axis) band
        expected_mags = [14.5, 14.8, 15.2, 15.6, 16.0, 17.0]
        expected_colors = [0.5, 0.7, 0.8, 0.9, 1.0, 1.0]
        for point, expected_mag, expected_color in zip(output['cmd'], expected_mags, expected_colors):
            self.assertAlmostEqual(point['mag'], expected_mag, places=3)
            self.assertAlmostEqual(point['color'], expected_color, places=3)
            self.assertAlmostEqual(point['color_err'], np.sqrt(0.02 ** 2 + 0.03 ** 2), places=3)

        # the four brightest stars picked up Gaia proper motions and parallaxes, the rest did not
        self.assertEqual(output['n_gaia_matched'], 4)
        self.assertEqual([point['gaia_match'] for point in output['cmd'][:6]], [True] * 4 + [False] * 2)
        self.assertAlmostEqual(output['cmd'][0]['pmra'], -5.1)
        self.assertAlmostEqual(output['cmd'][0]['pmdec'], 3.0)
        self.assertAlmostEqual(output['cmd'][0]['parallax'], 1.2)
        self.assertIsNone(output['cmd'][4]['pmra'])
        self.assertIsNone(output['cmd'][5]['parallax'])
        # Bailer-Jones geometric distance rides along with each Gaia match, bracketed by its bounds
        self.assertAlmostEqual(output['cmd'][0]['distance'], 810.0)
        self.assertAlmostEqual(output['cmd'][0]['distance_lo'], 760.0)
        self.assertAlmostEqual(output['cmd'][0]['distance_hi'], 870.0)
        # an unmatched image star has no Gaia distance
        self.assertIsNone(output['cmd'][5]['distance'])
        # five pool stars is below the clump threshold, so no automatic membership guess
        self.assertIsNone(output['membership_guess'])

        # the unmatched Gaia stray is appended as a gaia_only star with synthetic photometry
        self.assertEqual(output['n_gaia_only'], 1)
        self.assertEqual(len(output['cmd']), 7)
        self.assertEqual([point['gaia_only'] for point in output['cmd']], [False] * 6 + [True])
        gaia_only_point = output['cmd'][6]
        self.assertAlmostEqual(gaia_only_point['pmra'], 12.0)
        self.assertAlmostEqual(gaia_only_point['g_mag'], 14.0)
        self.assertAlmostEqual(gaia_only_point['color'], 0.8, places=3)
        self.assertAlmostEqual(gaia_only_point['mag'], 15.4, places=3)
        # this stray has no Bailer-Jones entry (nan distance), so its distance is null while its pm rides along
        self.assertIsNone(gaia_only_point['distance'])

    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.gaia_cone_search')
    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.FileCache')
    def test_gaia_only_stars_capped_and_brightest_first(self, mock_file_cache, mock_gaia_cone_search):
        # two image stars near the center with no Gaia counterparts
        star_dec = self.CENTER_DEC + np.array([10.0, 20.0]) / 3600.0
        star_ra = np.full(2, self.CENTER_RA)
        red_mags = np.array([15.0, 16.0])
        self.create_test_fits(self.temp_blue_path, self.CENTER_RA, self.CENTER_DEC,
                              star_ra, star_dec, red_mags + 0.5, np.full(2, 0.02))
        self.create_test_fits(self.temp_red_path, self.CENTER_RA, self.CENTER_DEC,
                              star_ra, star_dec, red_mags, np.full(2, 0.03))
        self.mock_band_fits(mock_file_cache)

        # five unmatched Gaia field stars: G mags shuffled, one with no G mag at all,
        # and the brightest one missing its synthetic photometry
        gaia_dec = self.CENTER_DEC - np.array([60.0, 70.0, 80.0, 90.0, 100.0]) / 3600.0
        mock_gaia_cone_search.return_value = self.gaia_field(
            ra=np.full(5, self.CENTER_RA),
            dec=gaia_dec,
            pmra=np.full(5, 4.0), pmdec=np.full(5, -2.0), parallax=np.full(5, 0.8),
            g_mag=[15.0, 11.0, 13.0, np.nan, 12.0],
            synth_g=[15.8, np.nan, 13.8, 14.1, 12.8],
            synth_r=[15.1, np.nan, 13.1, 13.5, 12.1],
        )

        with mock.patch.object(HRDiagram, 'MAX_GAIA_ONLY_SOURCES', 3):
            hr_diagram = HRDiagram(self.hr_input_data())
            hr_diagram.operate(None)
        output = hr_diagram.get_output()['output_data'][0]

        self.assertEqual(output['n_stars'], 2)
        self.assertEqual(output['n_gaia_matched'], 0)
        self.assertEqual(output['n_gaia_only'], 3)
        gaia_only_points = output['cmd'][2:]
        # the cap keeps the brightest three by G mag, ascending; the source with no G mag lost out
        self.assertEqual([point['g_mag'] for point in gaia_only_points], [11.0, 12.0, 13.0])
        self.assertTrue(all(point['gaia_only'] for point in gaia_only_points))
        # no synthetic photometry -> the star appears on the membership plots but not the CMD
        self.assertIsNone(gaia_only_points[0]['color'])
        self.assertIsNone(gaia_only_points[0]['mag'])
        self.assertAlmostEqual(gaia_only_points[1]['color'], 0.7, places=3)
        self.assertAlmostEqual(gaia_only_points[1]['mag'], 12.1, places=3)
        # Bailer-Jones distance rides along even for gaia-only stars
        self.assertAlmostEqual(gaia_only_points[1]['distance'], 800.0)
        self.assertAlmostEqual(gaia_only_points[1]['distance_lo'], 750.0)
        self.assertAlmostEqual(gaia_only_points[1]['distance_hi'], 860.0)

    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.gaia_cone_search')
    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.FileCache')
    def test_cone_filter_excludes_far_stars(self, mock_file_cache, mock_gaia_cone_search):
        # two stars near the center and one 3 arcmin away, with a 1 arcmin search radius
        ra = np.full(3, self.CENTER_RA)
        dec = self.CENTER_DEC + np.array([10.0, 20.0, 180.0]) / 3600.0
        mag = np.array([15.0, 16.0, 17.0])
        magerr = np.full(3, 0.02)

        self.create_test_fits(self.temp_blue_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag + 0.5, magerr)
        self.create_test_fits(self.temp_red_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag, magerr)
        self.mock_band_fits(mock_file_cache)
        mock_gaia_cone_search.return_value = self.empty_gaia_data()

        hr_diagram = HRDiagram(self.hr_input_data(search_radius_arcmin=1.0))
        hr_diagram.operate(None)
        output = hr_diagram.get_output()['output_data'][0]

        self.assertEqual(output['n_stars'], 2)
        self.assertEqual(output['cluster']['radius_arcmin'], 1.0)
        # an empty Gaia field is not an error, the stars are just unmatched
        self.assertEqual(output['n_gaia_matched'], 0)
        self.assertIsNone(output['membership_guess'])

    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.gaia_cone_search')
    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.FileCache')
    def test_radec_from_wcs_fallback(self, mock_file_cache, mock_gaia_cone_search):
        # the blue catalog has no ra/dec columns, so they are computed from its x/y and WCS
        ra = np.full(3, self.CENTER_RA)
        dec = self.CENTER_DEC + np.array([10.0, 20.0, 30.0]) / 3600.0
        mag = np.array([15.0, 16.0, 17.0])
        magerr = np.full(3, 0.02)

        self.create_test_fits(self.temp_blue_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag + 0.5, magerr,
                              include_radec=False)
        self.create_test_fits(self.temp_red_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag, magerr)
        self.mock_band_fits(mock_file_cache)
        mock_gaia_cone_search.return_value = self.empty_gaia_data()

        hr_diagram = HRDiagram(self.hr_input_data())
        hr_diagram.operate(None)
        output = hr_diagram.get_output()['output_data'][0]

        self.assertEqual(hr_diagram.get_status(), 'COMPLETED')
        self.assertEqual(output['n_stars'], 3)
        self.assertAlmostEqual(output['cmd'][0]['ra'], self.CENTER_RA, places=5)
        self.assertAlmostEqual(output['cmd'][0]['color'], 0.5, places=3)

    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.gaia_cone_search')
    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.FileCache')
    def test_gaia_failure_fails_operation(self, mock_file_cache, mock_gaia_cone_search):
        # a Gaia outage fails the operation (re-runnable) instead of caching a degraded result
        ra = np.full(2, self.CENTER_RA)
        dec = self.CENTER_DEC + np.array([10.0, 20.0]) / 3600.0
        mag = np.array([15.0, 16.0])
        magerr = np.full(2, 0.02)

        self.create_test_fits(self.temp_blue_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag + 0.5, magerr)
        self.create_test_fits(self.temp_red_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag, magerr)
        self.mock_band_fits(mock_file_cache)
        mock_gaia_cone_search.side_effect = ClientAlertException('Could not fetch Gaia data for the cluster field')

        hr_diagram = HRDiagram(self.hr_input_data())
        with self.assertRaisesRegex(ClientAlertException, 'Gaia'):
            hr_diagram.operate(None)

    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.FileCache')
    def test_uncalibrated_band_fails(self, mock_file_cache):
        ra = np.full(2, self.CENTER_RA)
        dec = self.CENTER_DEC + np.array([10.0, 20.0]) / 3600.0
        mag = np.array([15.0, 16.0])
        magerr = np.full(2, 0.02)

        self.create_test_fits(self.temp_blue_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag, magerr,
                              include_mag=False)
        self.create_test_fits(self.temp_red_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag, magerr)
        self.mock_band_fits(mock_file_cache)

        hr_diagram = HRDiagram(self.hr_input_data())
        with self.assertRaisesRegex(ClientAlertException, 'zero-point calibrated'):
            hr_diagram.operate(None)

    @mock.patch('datalab.datalab_session.data_operations.hr_diagram.FileCache')
    def test_disjoint_fields_fail(self, mock_file_cache):
        # the red image is a degree away, so no stars cross-match
        ra = np.full(2, self.CENTER_RA)
        dec = self.CENTER_DEC + np.array([10.0, 20.0]) / 3600.0
        mag = np.array([15.0, 16.0])
        magerr = np.full(2, 0.02)

        self.create_test_fits(self.temp_blue_path, self.CENTER_RA, self.CENTER_DEC, ra, dec, mag, magerr)
        self.create_test_fits(self.temp_red_path, self.CENTER_RA + 1.0, self.CENTER_DEC,
                              ra + 1.0, dec, mag, magerr)
        self.mock_band_fits(mock_file_cache)

        hr_diagram = HRDiagram(self.hr_input_data())
        with self.assertRaisesRegex(ClientAlertException, 'No stars matched'):
            hr_diagram.operate(None)

    def test_missing_cluster_fails(self):
        input_data = self.hr_input_data()
        del input_data['cluster']
        hr_diagram = HRDiagram(input_data)

        with self.assertRaisesRegex(ClientAlertException, 'cluster center'):
            hr_diagram.operate(None)

    def test_bad_search_radius_fails(self):
        hr_diagram = HRDiagram(self.hr_input_data(search_radius_arcmin=-5.0))

        with self.assertRaisesRegex(ClientAlertException, 'search radius'):
            hr_diagram.operate(None)

    def test_missing_input_files_fails(self):
        input_data = self.hr_input_data()
        input_data['red_filter_files'] = []
        hr_diagram = HRDiagram(input_data)

        with self.assertRaises(ClientAlertException):
            hr_diagram.operate(None)


class TestStackOperation(FileExtendedTestCase):
    test_fits_1_path = f'{test_path}fits_1.fits.fz'
    test_fits_2_path = f'{test_path}fits_2.fits.fz'

    temp_stacked_path = f'{test_path}temp_stacked.fits'  
    temp_fits_1_negative_path = f'{test_path}temp_fits_1_negative.fits'
    temp_fits_2_negative_path = f'{test_path}temp_fits_2_negative.fits'

    def tearDown(self):
        self.clean_test_dir()
        return super().tearDown()

    @mock.patch('datalab.datalab_session.utils.file_utils.tempfile.NamedTemporaryFile')
    @mock.patch('datalab.datalab_session.data_operations.input_data_handler.FileCache')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.FileCache', new=mock.MagicMock)
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.save_files_to_s3')
    @mock.patch('datalab.datalab_session.data_operations.fits_output_handler.create_jpgs')
    def test_operate(self, mock_create_jpgs, mock_save_files_to_s3, mock_file_cache, mock_named_tempfile):
        # Generate negative images
        for original, negative_path in [
            (self.test_fits_1_path, self.temp_fits_1_negative_path),
            (self.test_fits_2_path, self.temp_fits_2_negative_path)
        ]:
            hdul = fits.open(original)
            hdul['SCI'].data *= -1
            hdul.writeto(negative_path, overwrite=True)

        # Mock behavior
        mock_fc_instance1 = mock.MagicMock()
        mock_fc_instance1.get_fits.return_value = self.test_fits_1_path
        mock_fc_instance2 = mock.MagicMock()
        mock_fc_instance2.get_fits.return_value = self.test_fits_2_path
        mock_fc_instance1n = mock.MagicMock()
        mock_fc_instance1n.get_fits.return_value = self.temp_fits_1_negative_path
        mock_fc_instance2n = mock.MagicMock()
        mock_fc_instance2n.get_fits.return_value = self.temp_fits_2_negative_path
        mock_file_cache.side_effect = [mock_fc_instance1, mock_fc_instance2, mock_fc_instance1n, mock_fc_instance2n]

        mock_named_tempfile.return_value.__enter__.return_value.name = self.temp_stacked_path
        mock_create_jpgs.return_value.__enter__.return_value = ('test_path', 'test_path')
        mock_save_files_to_s3.return_value = self.temp_stacked_path

        input_data = {
            # input_data satisfies the Stack operation argument check, but the data comes from the mock_get_fits (above)
            'input_files': [
                {'basename': 'fits_1', 'source': 'local'},
                {'basename': 'fits_2', 'source': 'local'},
                {'basename': 'fits_1', 'source': 'local'},
                {'basename': 'fits_2', 'source': 'local'},
            ]
        }

        # Perform the stacking operation
        stack = Stack(input_data)
        stack.operate(None)
        output = stack.get_output().get('output_files')

        self.assertEqual(stack.get_operation_progress(), 1.0)
        self.assertEqual(self.temp_stacked_path, output[0])
        self.assertTrue(os.path.exists(self.temp_stacked_path))

        # Verify output is a blank image
        output_hdul = fits.open(self.temp_stacked_path)

        self.assertTrue(np.sum(output_hdul['SCI'].data) < 0.01)  # Changed to be close to zero because I was getting some tiny non-zero values when stacking multiple images

    def test_not_enough_files(self):
        input_data = {
            'input_files': [{'basename': 'sample_lco_fits_1'}]
        }
        median = Median(input_data)
        
        with self.assertRaises(ClientAlertException):
            median.operate(None)
