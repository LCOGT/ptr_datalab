from rest_framework.test import APITestCase
from mixer.backend.django import mixer
from django.contrib.auth.models import User
from django.urls import reverse
from unittest import mock
from types import SimpleNamespace

import numpy as np

from datalab.datalab_session.models import DataOperation, DataSession


class TestOperationsApi(APITestCase):
    def setUp(self) -> None:
        super().setUp()
        self.user = mixer.blend(User)
        self.client.force_login(self.user)
        self.session = mixer.blend(DataSession)

    def test_bulk_delete(self):
        operation1 = mixer.blend(DataOperation, session=self.session)
        operation2 = mixer.blend(DataOperation, session=self.session)
        operation3 = mixer.blend(DataOperation, session=self.session)

        to_delete = {'ids': [operation1.id, operation3.id]}
        response = self.client.post(reverse('api:datasession-operations-bulk-delete', args=(self.session.id,)), data=to_delete)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['deleted'], 2)
        # Only operation2 was not deleted
        self.assertEqual(DataOperation.objects.all().count(), 1)
        self.assertEqual(DataOperation.objects.first().id, operation2.id)

    @mock.patch('datalab.datalab_session.analysis.centroiding.get_hdu')
    @mock.patch('datalab.datalab_session.analysis.centroiding.FileCache')
    def test_centroiding_analysis_endpoint(self, mock_file_cache, mock_get_hdu):
        mock_instance = mock_file_cache.return_value
        mock_instance.get_fits.return_value = 'test.fits'

        fits_image = np.zeros((80, 120), dtype=float)
        fits_image[48, 36] = 1200.0
        mock_get_hdu.return_value = SimpleNamespace(data=fits_image)

        response = self.client.post(
            reverse('analysis', args=('centroiding',)),
            data={
                'basename': 'fits_1',
                'height': 160,
                'width': 240,
                'x': 72.0,
                'y': 96.0,
                'radius': 3.0,
                'r_back1': 4.0,
                'r_back2': 5.0,
                'source': 'archive',
            },
            format='json',
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['success'])
        self.assertAlmostEqual(response.json()['x'], 73.0, places=9)
        self.assertAlmostEqual(response.json()['y'], 97.0, places=9)
        self.assertEqual(response.json()['background'], 0.0)
        self.assertEqual(response.json()['peak'], 1200.0)
