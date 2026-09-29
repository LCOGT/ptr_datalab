from datetime import timedelta
from io import StringIO

from django.core.management import call_command
from django.test import TestCase
from django.utils import timezone
from mixer.backend.django import mixer

from datalab.datalab_session.management.commands import delete_old_datasessions as command_module
from datalab.datalab_session.models import DataOperation, DataSession


class TestDeleteOldDataSessions(TestCase):
    def _session_created_days_ago(self, days: int, persist: bool = False) -> DataSession:
        session = mixer.blend(DataSession, persist=persist)
        # created is auto_now_add, so it has to be backdated with an update
        DataSession.objects.filter(pk=session.pk).update(created=timezone.now() - timedelta(days=days))
        return session

    def _run(self, *args) -> None:
        call_command('delete_old_datasessions', *args, stdout=StringIO())

    def test_deletes_expired_sessions_and_cascades_to_their_operations(self):
        expired = self._session_created_days_ago(31)
        mixer.blend(DataOperation, session=expired)
        mixer.blend(DataOperation, session=expired)

        self._run()

        self.assertEqual(DataSession.objects.count(), 0)
        self.assertEqual(DataOperation.objects.count(), 0)

    def test_keeps_sessions_inside_the_cutoff(self):
        recent = self._session_created_days_ago(29)
        mixer.blend(DataOperation, session=recent)

        self._run()

        self.assertEqual(DataSession.objects.count(), 1)
        self.assertEqual(DataOperation.objects.count(), 1)

    def test_keeps_expired_sessions_marked_persist(self):
        persisted = self._session_created_days_ago(365, persist=True)
        mixer.blend(DataOperation, session=persisted)

        self._run()

        self.assertEqual(DataSession.objects.count(), 1)
        self.assertEqual(DataOperation.objects.count(), 1)

    def test_cutoff_days_argument_overrides_the_default(self):
        self._session_created_days_ago(10)

        self._run('--cutoff-days', '5')

        self.assertEqual(DataSession.objects.count(), 0)

    def test_logs_the_number_of_deleted_objects(self):
        expired = self._session_created_days_ago(31)
        mixer.blend(DataOperation, session=expired)
        self._session_created_days_ago(31)
        self._session_created_days_ago(1)

        with self.assertLogs(level='INFO') as logs:
            self._run()

        self.assertTrue(
            any('deleted 2 DataSessions and 1 DataOperations' in line for line in logs.output),
            msg=f'Unexpected log output: {logs.output}'
        )

    def test_deletes_more_sessions_than_fit_in_one_batch(self):
        for _ in range(5):
            self._session_created_days_ago(31)

        original_batch_size = command_module.DELETE_BATCH_SIZE
        command_module.DELETE_BATCH_SIZE = 2
        try:
            self._run()
        finally:
            command_module.DELETE_BATCH_SIZE = original_batch_size

        self.assertEqual(DataSession.objects.count(), 0)
