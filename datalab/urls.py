"""
URL configuration for datalab project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path, include
from rest_framework_nested import routers
import ocs_authentication.auth_profile.urls as authprofile_urls

from datalab.datalab_session.viewsets import DataSessionViewSet, DataOperationViewSet
from datalab.datalab_session.views import OperationOptionsApiView

router = routers.SimpleRouter()
router.register(r'datasessions', DataSessionViewSet, 'datasessions')
operations_router = routers.NestedSimpleRouter(router, r'datasessions', lookup='session')
operations_router.register(r'operations', DataOperationViewSet, basename='datasession-operations')

api_urlpatterns = ([
    re_path(r'^', include(router.urls)),
    re_path(r'^', include(operations_router.urls)),
], 'api')

urlpatterns = [
    path('admin/', admin.site.urls),
    re_path(r'^api/', include(api_urlpatterns)),
    path('api/available_operations/', OperationOptionsApiView.as_view(), name='available_operations'),
    re_path(r'^authprofile/', include(authprofile_urls)),
]
