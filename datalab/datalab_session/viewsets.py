from rest_framework import viewsets
from rest_framework.decorators import action
from django_filters.rest_framework import DjangoFilterBackend

from datalab.datalab_session.serializers import DataSessionSerializer, DataOperationSerializer
from datalab.datalab_session.models import DataSession, DataOperation
from datalab.datalab_session.filters import DataSessionFilterSet


class DataOperationViewSet(viewsets.ModelViewSet):
    serializer_class = DataOperationSerializer
    
    def get_queryset(self):
        return DataOperation.objects.filter(session=self.kwargs['session_pk'])
    
    def perform_create(self, serializer):
        serializer.save(session_id=self.kwargs['session_pk'])


class DataSessionViewSet(viewsets.ModelViewSet):
    serializer_class = DataSessionSerializer
    filterset_class = DataSessionFilterSet
    filter_backends = (
        DjangoFilterBackend,
    )
    ordering = ('created',)
    
    def get_queryset(self):
        return DataSession.objects.all()

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
