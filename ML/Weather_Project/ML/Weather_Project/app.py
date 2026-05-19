"""Flask entry point for the weather prediction dashboard."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request, url_for

try:
    from inference import get_dashboard_context, predict_weather
except ModuleNotFoundError:  # pragma: no cover - supports package-style imports.
    from .inference import get_dashboard_context, predict_weather


app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index() -> str:
    """Render the main weather dashboard."""

    context = get_dashboard_context(static_prefix="/static/", api_url=url_for("predict"))
    return render_template("index.html", **context)


@app.route("/predict", methods=["POST"])
def predict():
    """Return the model prediction as JSON for the frontend fetch call."""

    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify(predict_weather(payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
