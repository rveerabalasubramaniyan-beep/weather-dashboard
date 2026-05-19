"""Root URL configuration for the Django weather project."""

from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("dashboard.urls")),
]
