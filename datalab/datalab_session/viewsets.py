from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework.decorators import action
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
        print(f"Original Operation {operation.name()} - {operation.cache_key} - {operation.input_data}")
        serializer.save(session_id=self.kwargs['session_pk'], cache_key=operation.cache_key)
        operation.perform_operation(self.request.user.username)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        if instance.status == 'PENDING' and not instance.output:
            print(f"Retrying operation {instance.id} - {instance.name} - {instance.cache_key} - {instance.status} - {instance.input_data}")
            operation = available_operations().get(instance.name)(instance.input_data)
            operation.perform_operation(self.request.user.username)

        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def bulk_delete(self, request, session_pk=None):
        ''' Bulk Delete given a list of operation ids '''
        ids_to_delete = request.data.get('ids')
        num_deleted, _ = DataOperation.objects.filter(pk__in=ids_to_delete, session__pk=session_pk).delete()
        return Response({'deleted': num_deleted})


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
