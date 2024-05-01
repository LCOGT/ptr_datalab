from rest_framework.generics import RetrieveAPIView
from rest_framework.views import APIView
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

from datalab.datalab_session.data_operations.utils import available_operations
from datalab.datalab_session.analysis.line_profile import line_profile
from datalab.datalab_session.util import get_hdus


class OperationOptionsApiView(RetrieveAPIView):
    """ View to retrieve the set of operations available, for the UI to use """
    renderer_classes = [JSONRenderer]

    def get(self, request):
        operations = available_operations()
        operation_details = {}
        for operation_clazz in operations.values():
            operation_details[operation_clazz.name()] = operation_clazz.wizard_description()
        return Response(operation_details)

class AnalysisView(APIView):
    """ View to handle analysis actions and return the results """
    def post(self, request, action):
        input = request.data

        hdu = get_hdus(input['basename'])
        sci = hdu['SCI']

        match action:
            case 'line-profile':
                output = line_profile(input, sci)
            case _:
                raise Exception(f'Analysis action {action} not found')
        
        hdu.close()

        return Response(output)
