import sys
from django.core.management.base import BaseCommand

from datalab.datalab_session.utils.filecache import FileCache


class Command(BaseCommand):
    help = 'Reconciles the filecache with the files currently in the tmp dir of the system. Meant to run on pod creation'
    
    def handle(self, *args, **kwargs):
        FileCache().reconcile_cache()

        sys.exit(0)
