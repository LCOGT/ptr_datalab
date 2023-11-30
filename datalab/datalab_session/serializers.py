from rest_framework import serializers

from datalab.datalab_session.models import DataSession, DataOperation


class DataOperationSerializer(serializers.ModelSerializer):
    session_id = serializers.IntegerField(write_only=True, required=False)

    class Meta:
        model = DataOperation
        exclude = ('session',)


class DataSessionSerializer(serializers.ModelSerializer):
    operations = DataOperationSerializer(many=True, read_only=True)
    user = serializers.StringRelatedField(default=serializers.CurrentUserDefault(), read_only=True)

    class Meta:
        model = DataSession
        fields = '__all__'
