from django.contrib import admin

# Register your models here.
from datalab.datalab_session.models import DataOperation, DataSession
from datalab.datalab_session.forms import DataOperationForm


class DataOperationInline(admin.TabularInline):
    model = DataOperation
    forms = DataOperationForm


class DataOperationAdmin(admin.ModelAdmin):
    model = DataOperation
    form = DataOperationForm
    list_display = (
        'id',
        'session',
        'name',
        'created',
    )
    raw_id_fields = ('session',)

class DataSessionAdmin(admin.ModelAdmin):
    model=DataSession
    list_display = (
        'id',
        'name',
        'user',
        'operations_count',
        'created',
        'accessed',
        'modified',
    )
    list_filter = ('user', 'created', 'accessed')
    search_fields = ('name', 'user')
    readonly_fields = ('operations_count',)
    raw_id_fields = ('user',)
    inlines = [DataOperationInline,]
    
    def operations_count(self, obj):
        return obj.operations.count()


admin.site.register(DataSession, DataSessionAdmin)
admin.site.register(DataOperation, DataOperationAdmin)
