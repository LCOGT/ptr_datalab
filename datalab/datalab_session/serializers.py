from rest_framework import serializers

from datalab.datalab_session.models import DataSession, DataOperation
from datalab.datalab_session.data_operations.data_operation import available_operations


class DataOperationSerializer(serializers.ModelSerializer):
    session_id = serializers.IntegerField(write_only=True, required=False)
    name = serializers.ChoiceField(choices=[name for name in available_operations().keys()])

    class Meta:
        model = DataOperation
        exclude = ('session',)


class DataSessionSerializer(serializers.ModelSerializer):
    operations = DataOperationSerializer(many=True, read_only=True)
    user = serializers.StringRelatedField(default=serializers.CurrentUserDefault(), read_only=True)

    class Meta:
        model = DataSession
        fields = '__all__'
