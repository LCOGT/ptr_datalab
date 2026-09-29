import logging
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from datalab.datalab_session.models import DataOperation, DataSession

log = logging.getLogger()
log.setLevel(logging.INFO)

DEFAULT_CUTOFF_DAYS = 30  # DataSessions with persist=False are deleted after this long
DELETE_BATCH_SIZE = 500  # Number of DataSessions to delete per query, to bound memory and transaction size


class Command(BaseCommand):
    help = 'Deletes DataSessions with persist=False older than the cutoff. Meant to run daily from a CronJob'

    def add_arguments(self, parser):
        parser.add_argument(
            '--cutoff-days', type=int, default=DEFAULT_CUTOFF_DAYS,
            help=f'Delete non-persisted DataSessions created more than this many days ago (default: {DEFAULT_CUTOFF_DAYS})'
        )

    def handle(self, *args, **options):
        ''' The DataOperations belonging to each session cascade away with the session. '''
        cutoff = timezone.now() - timedelta(days=options['cutoff_days'])
        expired = DataSession.objects.filter(persist=False, created__lt=cutoff)

        sessions_deleted = 0
        operations_deleted = 0
        while True:
            batch_pks = list(expired.values_list('pk', flat=True)[:DELETE_BATCH_SIZE])
            if not batch_pks:
                break
            _, deleted_by_model = DataSession.objects.filter(pk__in=batch_pks).delete()
            sessions_deleted += deleted_by_model.get(DataSession._meta.label, 0)
            operations_deleted += deleted_by_model.get(DataOperation._meta.label, 0)

        log.info(
            f"delete_old_datasessions: deleted {sessions_deleted} DataSessions and "
            f"{operations_deleted} DataOperations created before {cutoff.isoformat()}"
        )
