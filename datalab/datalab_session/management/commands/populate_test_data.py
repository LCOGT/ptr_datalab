from typing import Any
import sys
from django.core.management.base import BaseCommand, CommandParser
from django.contrib.auth.models import User
from django.db.utils import IntegrityError

from rest_framework.authtoken.models import Token

from datalab.datalab_session.models import DataSession, DataOperation
import logging


logger = logging.getLogger(__name__)

TEST_PASSWORD = 'test_pass'
TEST_TOKEN = '123456789abcdefg'

INPUT_DATA_1 = [
    {
        'type': 'fitsfile',
        'source': 'archive',
        'basename': 'elp1m008-fa16-20231227-0058-e91'
    },
    {
        'type': 'fitsfile',
        'source': 'archive',
        'basename': 'elp1m008-fa16-20231225-0058-e91'
    },
    {
        'type': 'fitsfile',
        'source': 'archive',
        'basename': 'ogg2m001-ep04-20231114-0229-e91'
    },
]

OPERATION_INPUT_DATA_1 = {
    'input_files': INPUT_DATA_1
}


class Command(BaseCommand):
    help = 'Populates the DB with a set of example data sessions and data operations'
    
    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('-u', '--user', type=str, default='test_user',
                            help='Username to create an account for to associate with the Sessions that are created')

    def handle(self, *args: Any, **options: Any) -> str | None:
        try:
            user = User.objects.create_superuser(options['user'], '', TEST_PASSWORD)
            # Create token only if the user is new. If existing, leave it alone.
            token, _ = Token.objects.get_or_create(user=user)
            token.delete()
            Token.objects.create(user=user, key=TEST_TOKEN)
        except IntegrityError:
            logger.warning(f"User {options['user']} already exists")
            user = User.objects.get(username=options['user'])

        # Create an new datasession with just input files
        DataSession.objects.create(user=user, name='Empty Data Session', input_data=INPUT_DATA_1)        

        # Create a datasession with Operations
        ds = DataSession.objects.create(user=user, name='MyDataSession1', input_data=INPUT_DATA_1)
        DataOperation.objects.create(session=ds, name='NoOp', input_data=OPERATION_INPUT_DATA_1)
        DataOperation.objects.create(session=ds, name='Median', input_data=OPERATION_INPUT_DATA_1)

        sys.exit(0)
