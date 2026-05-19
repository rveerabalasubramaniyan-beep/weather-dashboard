"""Django app configuration for the dashboard module."""

from django.apps import AppConfig


class DashboardConfig(AppConfig):
    """Register the dashboard app with Django."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "dashboard"
