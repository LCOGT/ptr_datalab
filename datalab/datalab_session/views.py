import logging

from rest_framework.generics import RetrieveAPIView
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response

from datalab.datalab_session.data_operations.utils import available_operations
from datalab.datalab_session.analysis.line_profile import line_profile
from datalab.datalab_session.analysis.source_catalog import source_catalog
from datalab.datalab_session.analysis.get_tif import get_tif
from datalab.datalab_session.analysis.get_jpg import get_jpg
from datalab.datalab_session.analysis.raw_data import raw_data
from datalab.datalab_session.analysis.variable_star import variable_star
from datalab.datalab_session.exceptions import ClientAlertException

log = logging.getLogger()
log.setLevel(logging.INFO)

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
    """ View to handle analysis actions and return the results. """

    ACTIONS = {
        "line-profile": line_profile,
        "source-catalog": source_catalog,
        "get-tif": get_tif,
        "get-jpg": get_jpg,
        "raw-data": raw_data,
        "variable-star": variable_star,
    }

    def post(self, request, action):
        log.info(f"Received analysis action request: {action}")

        try:
            input_data = request.data
            action_function = self.ACTIONS.get(action)

            if not action_function:
                log.warning(f"Invalid action: {action}")
                return Response({"error": f"Analysis action '{action}' not found"}, status=400)

            output = action_function(input_data)

            return Response(output)

        except ClientAlertException as error:
            log.error(f"Error running analysis action {action}: {error}")
            return Response({"error": f"Image doesn't support {action}"}, status=400)
