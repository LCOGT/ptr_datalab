from rest_framework import serializers

from datalab.datalab_session.models import DataSession, DataOperation
from datalab.datalab_session.data_operations.utils import available_operations


class DataOperationSerializer(serializers.ModelSerializer):
    session_id = serializers.IntegerField(write_only=True, required=False)
    name = serializers.ChoiceField(choices=[name for name in available_operations().keys()])
    cache_key = serializers.CharField(write_only=True, required=False)
    status = serializers.ReadOnlyField()
    message = serializers.ReadOnlyField()
    percent_completion = serializers.ReadOnlyField()
    output = serializers.ReadOnlyField()

    class Meta:
        model = DataOperation
        exclude = ('session',)
        read_only_fields = (
            'id', 'created', 'status', 'percent_completion', 'message', 'output',
        )

class DataSessionSerializer(serializers.ModelSerializer):
    operations = DataOperationSerializer(many=True, read_only=True)
    user = serializers.StringRelatedField(default=serializers.CurrentUserDefault(), read_only=True)

    class Meta:
        model = DataSession
        fields = '__all__'
