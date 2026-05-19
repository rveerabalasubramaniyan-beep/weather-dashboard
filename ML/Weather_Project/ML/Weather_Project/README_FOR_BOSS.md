# Quick Check for Boss

This file is the simplest way to verify that the machine learning model works
on another Windows system.

## Recommended Python Version

Use **Python 3.11 or Python 3.12** for the smoothest installation experience.

## Option 1: Fastest Verification

1. Extract the `Weather_Project` zip folder.
2. Double-click `run_smoke_test.bat`
3. Wait for dependencies to install.
4. A sample prediction will print in the command window.

If the output shows a prediction dictionary with fields like:

- `predicted_label`
- `rain_probability`
- `confidence_score`
- `climate_condition`

then the model is working correctly.

## Option 2: Open the Website Locally

1. Extract the `Weather_Project` zip folder.
2. Double-click `run_flask.bat`
3. Wait for setup to finish.
4. Open this address in the browser:

```text
http://127.0.0.1:5000
```

This launches the weather dashboard website locally and uses the saved model
for live predictions.

## If Python Is Not Installed

Install Python first from:

```text
https://www.python.org/downloads/
```

During installation, enable:

```text
Add Python to PATH
```

## Important Files

- `model/weather_model.pkl`
- `model/preprocessing.pkl`
- `app.py`
- `inference.py`

These files must stay together inside the project folder.
