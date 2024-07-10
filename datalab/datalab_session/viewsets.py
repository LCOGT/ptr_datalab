from rest_framework import viewsets
from django_filters.rest_framework import DjangoFilterBackend

from datalab.datalab_session.serializers import DataSessionSerializer, DataOperationSerializer
from datalab.datalab_session.models import DataSession, DataOperation
from datalab.datalab_session.filters import DataSessionFilterSet
from datalab.datalab_session.data_operations.utils import available_operations


class DataOperationViewSet(viewsets.ModelViewSet):
    serializer_class = DataOperationSerializer
    
    def get_queryset(self):
        return DataOperation.objects.filter(session=self.kwargs['session_pk'], session__user=self.request.user)
    
    def perform_create(self, serializer):
        operation = available_operations().get(serializer.validated_data['name'])(serializer.validated_data['input_data'])
        serializer.save(session_id=self.kwargs['session_pk'], cache_key=operation.cache_key)
        operation.perform_operation()


class DataSessionViewSet(viewsets.ModelViewSet):
    serializer_class = DataSessionSerializer
    filterset_class = DataSessionFilterSet
    filter_backends = (
        DjangoFilterBackend,
    )
    ordering = ('created',)
    
    def get_queryset(self):
        return DataSession.objects.filter(user=self.request.user).prefetch_related('operations')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
