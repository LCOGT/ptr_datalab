from rest_framework.generics import RetrieveAPIView
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

from datalab.datalab_session.data_operations.utils import available_operations


class OperationOptionsApiView(RetrieveAPIView):
    """ View to retrieve the set of operations available, for the UI to use """
    renderer_classes = [JSONRenderer]

    def get(self, request):
        operations = available_operations()
        operation_details = {}
        for operation_clazz in operations.values():
            operation_details[operation_clazz.name()] = operation_clazz.wizard_description()
        return Response(operation_details)
