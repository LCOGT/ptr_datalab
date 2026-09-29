from rest_framework.test import APITestCase
from mixer.backend.django import mixer
from django.contrib.auth.models import User
from django.core.cache import caches
from django.urls import reverse
from unittest import mock
from types import SimpleNamespace

import numpy as np

from datalab.datalab_session.models import DataOperation, DataSession


class TestDataSessionApi(APITestCase):
    def setUp(self) -> None:
        super().setUp()
        self.user = mixer.blend(User)
        self.client.force_login(self.user)
        self.input_data = [
            {'basename': 'fits_1', 'source': 'archive'},
            {'basename': 'fits_2', 'source': 'archive'},
        ]
        self.session = mixer.blend(DataSession, user=self.user, persist=False, input_data=self.input_data)

    def test_patch_sets_persist(self):
        response = self.client.patch(reverse('api:datasessions-detail', args=(self.session.id,)), data={'persist': True})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['persist'])
        # Ensure input_data doesn't change
        self.assertEqual(response.json()['input_data'], self.input_data)

        self.session.refresh_from_db()
        self.assertTrue(self.session.persist)

    def test_patch_unsets_persist(self):
        self.session.persist = True
        self.session.save()

        response = self.client.patch(reverse('api:datasessions-detail', args=(self.session.id,)), data={'persist': False})

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['persist'])
        # Ensure input_data doesn't change
        self.assertEqual(response.json()['input_data'], self.input_data)

        self.session.refresh_from_db()
        self.assertFalse(self.session.persist)

    def test_patching_persist_alone_leaves_operations_untouched(self):
        operation = mixer.blend(DataOperation, session=self.session)

        response = self.client.patch(reverse('api:datasessions-detail', args=(self.session.id,)), data={'persist': True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual([op['id'] for op in response.json()['operations']], [operation.id])
        self.assertEqual(DataOperation.objects.filter(session=self.session).count(), 1)

    def test_patching_input_data_appends_without_duplicating(self):
        new_item = {'basename': 'fits_3', 'source': 'archive'}
        data = {'input_data': [self.input_data[0], new_item]}

        response = self.client.patch(reverse('api:datasessions-detail', args=(self.session.id,)), data=data, format='json')

        self.assertEqual(response.status_code, 200)
        self.session.refresh_from_db()
        self.assertEqual(self.session.input_data, self.input_data + [new_item])

    def test_cannot_patch_persist_on_another_users_session(self):
        other_session = mixer.blend(DataSession, persist=False)

        response = self.client.patch(reverse('api:datasessions-detail', args=(other_session.id,)), data={'persist': True})

        self.assertEqual(response.status_code, 404)
        other_session.refresh_from_db()
        self.assertFalse(other_session.persist)

    def test_patch_without_response_fields_returns_every_field(self):
        response = self.client.patch(reverse('api:datasessions-detail', args=(self.session.id,)), data={'persist': True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json().keys()),
            {'id', 'user', 'name', 'input_data', 'persist', 'created', 'accessed', 'modified', 'operations'}
        )

    def test_patch_with_response_fields_returns_only_those_fields(self):
        url = reverse('api:datasessions-detail', args=(self.session.id,)) + '?response_fields=id,persist,modified'

        response = self.client.patch(url, data={'persist': True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json().keys()), {'id', 'persist', 'modified'})
        self.assertTrue(response.json()['persist'])
        self.session.refresh_from_db()
        self.assertTrue(self.session.persist)

    def test_response_fields_does_not_affect_what_is_written(self):
        # persist is written even though the response only carries the id
        url = reverse('api:datasessions-detail', args=(self.session.id,)) + '?response_fields=id'

        response = self.client.patch(url, data={'persist': True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json().keys()), {'id'})
        self.session.refresh_from_db()
        self.assertTrue(self.session.persist)

    def test_response_fields_skips_serializing_operations(self):
        mixer.blend(DataOperation, session=self.session, cache_key='abc')
        url = reverse('api:datasessions-detail', args=(self.session.id,)) + '?response_fields=id,persist'

        # Every operation reads status, progress, output and message from the cache, so a cache
        # read here means the operations were serialized despite not being asked for
        with mock.patch.object(caches['default'], 'get', side_effect=AssertionError('operations were serialized')):
            response = self.client.patch(url, data={'persist': True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json().keys()), {'id', 'persist'})

    def test_response_fields_still_merges_input_data(self):
        new_item = {'basename': 'fits_3', 'source': 'archive'}
        url = reverse('api:datasessions-detail', args=(self.session.id,)) + '?response_fields=id,input_data'

        response = self.client.patch(url, data={'input_data': [self.input_data[0], new_item]}, format='json')

        self.assertEqual(response.status_code, 200)
        # The response echoes the stored merge, not the two items that were sent
        self.assertEqual(response.json()['input_data'], self.input_data + [new_item])
        self.session.refresh_from_db()
        self.assertEqual(self.session.input_data, self.input_data + [new_item])

    def test_response_fields_ignores_unknown_field_names(self):
        url = reverse('api:datasessions-detail', args=(self.session.id,)) + '?response_fields=id,bogus'

        response = self.client.patch(url, data={'persist': True})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json().keys()), {'id'})

    def test_response_fields_applies_to_get(self):
        url = reverse('api:datasessions-detail', args=(self.session.id,)) + '?response_fields=id,name'

        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.json().keys()), {'id', 'name'})


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
        data = {
            'basename': 'fits_1',
            'height': 160,
            'width': 240,
            'x': 72.0,
            'y': 96.0,
            'radius': 3.0,
            'r_back1': 4.0,
            'r_back2': 5.0,
            'source': 'archive',
        }

        response = self.client.post(reverse('analysis', args=('centroiding',)), data=data, format='json')
        response_data = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response_data['success'])
        self.assertAlmostEqual(response_data['x'], 73.0, places=9)
        self.assertAlmostEqual(response_data['y'], 97.0, places=9)
        self.assertEqual(response_data['background'], 0.0)
        self.assertEqual(response_data['peak'], 1200.0)
        self.assertEqual(response_data['message'], 'Centroid calculation completed.')
        self.assertIsNone(response_data['ra'])
        self.assertIsNone(response_data['dec'])
