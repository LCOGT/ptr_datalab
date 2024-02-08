from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.cache import cache


class DataSession(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE,
        help_text='The user that this DataSession belongs too'
    )
    
    name = models.CharField(max_length=256,
        help_text='User specified name for this data session'
    )

    input_data = models.JSONField(blank=True, default=list,
        help_text='List of input Data objects for this session in serialized format'
    )

    created = models.DateTimeField(
        auto_now_add=True,
        help_text='Time when this DataSession was created'
    )
    accessed = models.DateTimeField(
        auto_now=True,
        help_text='Time when this DataSession was last requested'
    )
    modified = models.DateTimeField(
        auto_now=True,
        help_text='Time when this DataSession was last changed'
    )

    @classmethod
    def from_db(cls, db, field_names, values):
        """ Increment the accessed property every time this instance is retrieved
        """
        obj = super().from_db(db, field_names, values)
        accessed = timezone.now()
        DataSession.objects.filter(pk=obj.pk).update(accessed=accessed)
        obj.accessed = accessed
        return obj


class DataOperation(models.Model):
    session = models.ForeignKey(
        DataSession, related_name='operations', on_delete=models.CASCADE,
        help_text='The DataSession to which this DataOperation belongs'
    )

    name = models.CharField(max_length=128,
        help_text='The name of this operation. Must be a valid operation as defined in available_operations()'
    )
    
    input_data = models.JSONField(blank=True, default=list,
        help_text='List of input Data objects for this session in serialized format'
    )
    
    created = models.DateTimeField(
        auto_now_add=True,
        help_text='Time when this DataSession was created'
    )

    cache_key = models.CharField(max_length=64, default='', blank=True, help_text='Cache key for this operation')

    @property
    def status(self):
        return cache.get(f'operation_{self.cache_key}_status', 'PENDING')

    @property
    def percent_completion(self):
        return cache.get(f'operation_{self.cache_key}_percent_completion', 0.0)

    @property
    def output(self):
        return cache.get(f'operation_{self.cache_key}_output')

    @property
    def message(self):
        return cache.get(f'operation_{self.cache_key}_message', '')
