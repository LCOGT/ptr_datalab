import logging

from rest_framework.generics import RetrieveAPIView
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

from datalab.datalab_session.data_operations.utils import available_operations
from datalab.datalab_session.analysis.line_profile import line_profile
from datalab.datalab_session.analysis.source_catalog import source_catalog
from datalab.datalab_session.analysis.get_tif import get_tif
from datalab.datalab_session.analysis.raw_data import raw_data
from datalab.datalab_session.exceptions import ClientAlertException


class OperationOptionsApiView(RetrieveAPIView):
    """ View to retrieve the set of operations available, for the UI to use """
    renderer_classes = [JSONRenderer]

    def get(self, request):
        operations = available_operations()
        operation_details = {}
        for operation_clazz in operations.values():
            operation_details[operation_clazz.name()] = operation_clazz.wizard_description()
        return Response(operation_details)

class AnalysisView(RetrieveAPIView):
    """ 
        View to handle analysis actions and return the results
        To add a new analysis action, add a case to the switch statement and create a new file in the analysis directory
    """
    def post(self, request, action):
        log = logging.getLogger()
        log.setLevel(logging.INFO)

        try:
            # TODO replace switch case with an AnalysisAction class
            # AnalysisAction Class Definition
            # class AnalysisAction:
            #     def __init__(self, action):
            #     def run(self, input):
            #     def name(self):
            input = request.data

            match action:
                case 'line-profile':
                    output = line_profile(input)
                case 'source-catalog':
                    output = source_catalog(input)
                case 'get-tif':
                    output = get_tif(input)
                case 'raw-data':
                    output = raw_data(input)
                case _:
                    raise Exception(f'Analysis action {action} not found')

            return Response(output)
        
        except ClientAlertException as error:
            log.error(f"Error running analysis action {action}: {error}")
            return Response({"error": f"Image doesn't support {action}"}, status=400)
