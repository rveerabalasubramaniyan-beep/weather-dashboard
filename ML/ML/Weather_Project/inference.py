"""Shared inference utilities used by both the Flask and Django apps."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"

FIELD_LABELS = {
    "MinTemp": "Minimum Temperature (deg C)",
    "MaxTemp": "Maximum Temperature (deg C)",
    "WindGustDir": "Wind Gust Direction",
    "WindGustSpeed": "Wind Gust Speed (km/h)",
    "Humidity": "Humidity (%)",
    "Pressure": "Pressure (hPa)",
    "Temp": "Current Temperature (deg C)",
}

FIELD_HELP_TEXT = {
    "MinTemp": "Lowest temperature expected in the period.",
    "MaxTemp": "Highest temperature expected in the period.",
    "WindGustDir": "Compass direction of the strongest wind gust.",
    "WindGustSpeed": "Peak gust speed for the day.",
    "Humidity": "Relative humidity percentage.",
    "Pressure": "Atmospheric pressure in hectopascals.",
    "Temp": "Observed or current temperature.",
}


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, dict[str, Any]]:
    """Load the trained model and preprocessing metadata once per process."""

    model = joblib.load(MODEL_DIR / "weather_model.pkl")
    metadata = joblib.load(MODEL_DIR / "preprocessing.pkl")
    return model, metadata


def positive_class_index(classes: list[str]) -> int:
    """Identify the class that best represents rain/positive weather risk."""

    positive_tokens = ("yes", "rain", "true", "1", "storm")
    lowered = [label.lower() for label in classes]
    for index, label in enumerate(lowered):
        if any(token in label for token in positive_tokens):
            return index
    return min(1, len(classes) - 1)


def get_form_fields() -> list[dict[str, Any]]:
    """Build field metadata for the frontend form."""

    _, metadata = load_artifacts()
    fields: list[dict[str, Any]] = []

    for name in metadata["feature_columns"]:
        is_numeric = name in metadata["numeric_features"]
        field: dict[str, Any] = {
            "name": name,
            "label": FIELD_LABELS.get(name, name),
            "help_text": FIELD_HELP_TEXT.get(name, ""),
            "default": metadata["default_inputs"][name],
        }

        if is_numeric:
            field.update(
                {
                    "type": "number",
                    "step": "0.1",
                    "placeholder": str(metadata["default_inputs"][name]),
                }
            )
        else:
            field.update(
                {
                    "type": "select",
                    "options": metadata["categorical_options"].get(name, []),
                }
            )

        fields.append(field)

    return fields


def normalize_inputs(raw_payload: dict[str, Any]) -> dict[str, Any]:
    """Convert request payload values into the schema expected by the model."""

    _, metadata = load_artifacts()
    normalized: dict[str, Any] = {}

    for feature in metadata["feature_columns"]:
        fallback = metadata["default_inputs"][feature]
        raw_value = raw_payload.get(feature, fallback)

        if feature in metadata["numeric_features"]:
            if raw_value in ("", None):
                raw_value = fallback
            try:
                normalized[feature] = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value provided for {feature}.") from exc
        else:
            normalized[feature] = str(raw_value or fallback)

    return normalized


def derive_climate_condition(values: dict[str, Any], rain_probability: float) -> str:
    """Create a richer climate condition label for the dashboard UI.

    The dataset predicts rain/no-rain, so this helper converts the rain
    probability and the live inputs into a more expressive condition label.
    """

    temperature = float(values.get("Temp", 0.0))
    max_temp = float(values.get("MaxTemp", temperature))
    humidity = float(values.get("Humidity", 0.0))
    pressure = float(values.get("Pressure", 1013.0))
    wind_speed = float(values.get("WindGustSpeed", 0.0))

    if temperature <= 2 or max_temp <= 3:
        return "Snow"
    if pressure <= 995 and wind_speed >= 45 and rain_probability >= 0.45:
        return "Storm"
    if humidity >= 92 and rain_probability < 0.35:
        return "Fog"
    if rain_probability >= 0.55:
        return "Rainy"
    if humidity >= 70 or wind_speed >= 35:
        return "Cloudy"
    return "Sunny"


def build_summary(condition: str, rain_probability: float, confidence: float) -> str:
    """Generate a short human-readable summary for the prediction card."""

    if condition == "Rainy":
        return (
            f"Rain risk is elevated at {rain_probability:.1%}, with a "
            f"{confidence:.1%} model confidence."
        )
    if condition == "Storm":
        return (
            f"Stormy signals are present due to low pressure and strong winds. "
            f"Confidence is {confidence:.1%}."
        )
    if condition == "Snow":
        return (
            f"Very cold temperatures suggest snow-like conditions. "
            f"Confidence is {confidence:.1%}."
        )
    if condition == "Fog":
        return (
            f"High humidity with limited rain pressure points to foggy conditions. "
            f"Confidence is {confidence:.1%}."
        )
    if condition == "Cloudy":
        return (
            f"Cloud cover looks more likely than clear skies. "
            f"Confidence is {confidence:.1%}."
        )
    return (
        f"Conditions look mostly clear, with a rain chance of {rain_probability:.1%} "
        f"and confidence of {confidence:.1%}."
    )


def predict_weather(raw_payload: dict[str, Any]) -> dict[str, Any]:
    """Run the trained model and return website-friendly prediction data."""

    model, metadata = load_artifacts()
    cleaned = normalize_inputs(raw_payload)
    frame = pd.DataFrame([cleaned], columns=metadata["feature_columns"])

    probabilities = model.predict_proba(frame)[0]
    classes = metadata["target_classes"]
    class_index = int(model.predict(frame)[0])
    predicted_label = classes[class_index]

    rain_index = positive_class_index(classes)
    rain_probability = float(probabilities[rain_index])
    confidence = float(max(probabilities))
    climate_condition = derive_climate_condition(cleaned, rain_probability)

    return {
        "target_column": metadata["target_column"],
        "predicted_label": predicted_label,
        "rain_probability": round(rain_probability * 100, 2),
        "confidence_score": round(confidence * 100, 2),
        "climate_condition": climate_condition,
        "temperature": round(float(cleaned.get("Temp", 0.0)), 1),
        "humidity": round(float(cleaned.get("Humidity", 0.0)), 1),
        "wind_speed": round(float(cleaned.get("WindGustSpeed", 0.0)), 1),
        "pressure": round(float(cleaned.get("Pressure", 0.0)), 1),
        "summary": build_summary(climate_condition, rain_probability, confidence),
        "model_name": metadata["best_model_name"],
    }


def get_dashboard_context(static_prefix: str, api_url: str) -> dict[str, Any]:
    """Collect the values required to render the shared dashboard template."""

    _, metadata = load_artifacts()
    best_row = metadata["model_results"][0]

    return {
        "app_title": "Weather Intelligence Dashboard",
        "form_fields": get_form_fields(),
        "api_url": api_url,
        "static_prefix": static_prefix,
        "target_column": metadata["target_column"],
        "best_model_name": metadata["best_model_name"],
        "model_metrics": {
            "accuracy": f"{best_row['accuracy'] * 100:.2f}%",
            "precision": f"{best_row['precision'] * 100:.2f}%",
            "recall": f"{best_row['recall'] * 100:.2f}%",
            "f1": f"{best_row['f1'] * 100:.2f}%",
            "roc_auc": f"{best_row['roc_auc'] * 100:.2f}%",
        },
        "condition_note": metadata["notes"]["condition_mapping"],
    }
