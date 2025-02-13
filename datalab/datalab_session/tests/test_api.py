from rest_framework.test import APITestCase
from mixer.backend.django import mixer
from django.contrib.auth.models import User
from django.urls import reverse

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
