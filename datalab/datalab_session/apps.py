from django.apps import AppConfig


class DatalabSessionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'datalab.datalab_session'

    def ready(self):
        import datalab.datalab_session.signals.handlers  # noqa
        return super().ready()
