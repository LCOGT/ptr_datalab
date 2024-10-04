from rest_framework import serializers

from datalab.datalab_session.models import DataSession, DataOperation
from datalab.datalab_session.data_operations.utils import available_operations


class DataOperationSerializer(serializers.ModelSerializer):
    session_id = serializers.IntegerField(write_only=True, required=False)
    name = serializers.ChoiceField(choices=[name for name in available_operations().keys()])
    cache_key = serializers.CharField(write_only=True, required=False)
    status = serializers.ReadOnlyField()
    message = serializers.ReadOnlyField()
    operation_progress = serializers.ReadOnlyField()
    output = serializers.ReadOnlyField()

    class Meta:
        model = DataOperation
        exclude = ('session',)
        read_only_fields = (
            'id', 'created', 'status', 'operation_progress', 'message', 'output',
        )

class DataSessionSerializer(serializers.ModelSerializer):
    operations = DataOperationSerializer(many=True, read_only=True)
    user = serializers.StringRelatedField(default=serializers.CurrentUserDefault(), read_only=True)

    class Meta:
        model = DataSession
        fields = '__all__'

    def update(self, instance, validated_data):
      input_data = validated_data.get('input_data', [])
      
			# Check for duplicates
      existing_input_data = instance.input_data
      new_input_data = []
      for item in input_data:
              if item not in existing_input_data:
                      new_input_data.append(item)

      validated_data['input_data'] = existing_input_data + new_input_data
      return super().update(instance, validated_data)
