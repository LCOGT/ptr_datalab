import django_filters


class DataSessionFilterSet(django_filters.FilterSet):
    created_after = django_filters.DateTimeFilter(field_name='created', lookup_expr='gte', label='Created after')
    created_before = django_filters.DateTimeFilter(field_name='created', lookup_expr='lte', label='Created before')
    name = django_filters.CharFilter(field_name='name', lookup_expr='icontains', label='Name contains')
    name_exact = django_filters.CharFilter(field_name='name', lookup_expr='exact', label='Name exact')
    user = django_filters.CharFilter(field_name='user__username', lookup_expr='icontains', label='Username contains')
    user_exact = django_filters.CharFilter(field_name='user__username', lookup_expr='icontains', label='Username contains')
    modified_after = django_filters.DateTimeFilter(field_name='modified', lookup_expr='gte', label='Modified After', distinct=True)
    modified_before = django_filters.DateTimeFilter(field_name='modified', lookup_expr='lte', label='Modified Before', distinct=True)
