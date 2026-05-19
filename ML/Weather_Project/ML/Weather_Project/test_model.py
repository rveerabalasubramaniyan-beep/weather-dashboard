"""Quick smoke test for verifying the trained model on another machine.

This script loads the saved artifacts and runs one sample prediction using the
same shared inference code as the website.
"""

from __future__ import annotations

from pprint import pprint

try:
    from inference import predict_weather
except ModuleNotFoundError:  # pragma: no cover - supports package-style imports.
    from .inference import predict_weather


def main() -> None:
    """Run one known-good sample prediction and print the result."""

    sample_payload = {
        "MinTemp": 8,
        "MaxTemp": 24.3,
        "WindGustDir": "NW",
        "WindGustSpeed": 30,
        "Humidity": 29,
        "Pressure": 1015,
        "Temp": 23.6,
    }

    print("Running smoke test with sample weather input...")
    pprint(sample_payload)
    print("\nPrediction output:")
    pprint(predict_weather(sample_payload))


if __name__ == "__main__":
    main()
