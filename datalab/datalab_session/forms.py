from django import forms

from datalab.datalab_session.models import DataOperation
from datalab.datalab_session.data_operations.data_operation import available_operations_tuples

class DataOperationForm(forms.ModelForm):
    class Meta:
        model = DataOperation
        fields = '__all__'
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['name'] = forms.ChoiceField(choices=available_operations_tuples())
