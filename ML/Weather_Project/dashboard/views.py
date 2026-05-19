"""Django views for rendering and serving weather predictions."""

from __future__ import annotations

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

try:
    from inference import get_dashboard_context, predict_weather
except ModuleNotFoundError:  # pragma: no cover - supports package-style imports.
    from ..inference import get_dashboard_context, predict_weather


def index(request):
    """Render the shared dashboard template."""

    context = get_dashboard_context(static_prefix="/static/", api_url="/predict/")
    return render(request, "index.html", context)


@csrf_exempt
def predict(request):
    """Run the ML model and return a JSON response to the dashboard."""

    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests are allowed."}, status=405)

    if request.body:
        payload = json.loads(request.body.decode("utf-8"))
    else:
        payload = request.POST.dict()

    try:
        return JsonResponse(predict_weather(payload))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
