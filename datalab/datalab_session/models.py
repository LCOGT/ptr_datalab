from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator


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
    STATUS_CHOICES = (
        ('PENDING', 'PENDING'),
        ('STARTED', 'STARTED'),
        ('COMPLETED', 'COMPLETED'),
        ('FAILED', 'FAILED')
    )

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

    percent_complete = models.IntegerField(default=0,
        help_text='Completion status (in percent) of this DataOperation',
        validators=[MinValueValidator(0),
                    MaxValueValidator(100)]
    )

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING',
        help_text='Status of this DataOperation'
    )

    message = models.CharField(max_length=200, default='', blank=True,
        help_text = 'Contextual message related to this DataOperation'
    )
