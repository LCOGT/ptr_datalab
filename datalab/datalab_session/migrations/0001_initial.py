# Generated by Django 4.2.7 on 2023-11-17 21:41

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='DataSession',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='User specified name for this data session', max_length=256)),
                ('input_data', models.JSONField(blank=True, default=list, help_text='List of input Data objects for this session in serialized format')),
                ('created', models.DateTimeField(auto_now_add=True, help_text='Time when this DataSession was created')),
                ('accessed', models.DateTimeField(auto_now=True, help_text='Time when this DataSession was last requested')),
                ('modified', models.DateTimeField(auto_now=True, help_text='Time when this DataSession was last changed')),
                ('user', models.ForeignKey(help_text='The user that this DataSession belongs too', on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='DataOperation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(help_text='The name of this operation. Must be a valid operation as defined in available_operations()', max_length=128)),
                ('input_data', models.JSONField(blank=True, default=list, help_text='List of input Data objects for this session in serialized format')),
                ('created', models.DateTimeField(auto_now_add=True, help_text='Time when this DataSession was created')),
                ('session', models.ForeignKey(help_text='The DataSession to which this DataOperation belongs', on_delete=django.db.models.deletion.CASCADE, related_name='operations', to='datalab_session.datasession')),
            ],
        ),
    ]
