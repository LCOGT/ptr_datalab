from django.dispatch import receiver
from django.db.models.signals import post_delete

from datalab.datalab_session.models import DataOperation


@receiver(post_delete, sender=DataOperation)
def cb_dataoperation_post_delete(sender, instance, *args, **kwargs):
    # If the status of the data operation FAILED, delete it from cache
    if instance.status == 'FAILED':
        instance.clear_cache()
