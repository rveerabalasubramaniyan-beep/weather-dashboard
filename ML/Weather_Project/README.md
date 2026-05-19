# Weather Intelligence Dashboard Project

This project turns the uploaded `weather.csv` dataset into a complete machine
learning website with both Flask and Django backends. The same trained model
and preprocessing metadata power both web apps, so the prediction behavior is
consistent whichever framework you run.

## Dataset Summary

- Source dataset: `data/weather.csv`
- Detected target column: `RainTomorrow`
- Problem type: binary classification
- Dataset shape: 366 rows x 8 columns
- Missing values:
  - `WindGustDir`: 3
  - `WindGustSpeed`: 2

## What the Model Predicts

The dataset's real target is `RainTomorrow`, so the model predicts rain vs.
no-rain. The dashboard still shows richer climate conditions such as
Sunny/Rainy/Cloudy/Storm/Snow/Fog by combining the rain probability with the
user's live weather inputs like temperature, humidity, pressure, and wind.

## Project Structure

```text
Weather_Project/
├── app.py
├── manage.py
├── inference.py
├── train_pipeline.py
├── data/
│   └── weather.csv
├── dashboard/
│   ├── apps.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
├── model/
│   ├── weather_model.pkl
│   └── preprocessing.pkl
├── reports/
│   ├── dataset_profile.json
│   ├── dataset_summary.md
│   ├── model_comparison.csv
│   └── plots/
├── static/
│   ├── css/
│   ├── icons/
│   ├── images/
│   └── js/
├── templates/
│   └── index.html
└── weather_site/
    ├── settings.py
    ├── urls.py
    ├── wsgi.py
    └── asgi.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Train or Rebuild the Model

```bash
python train_pipeline.py
```

The script performs:

- Exploratory data analysis
- Missing value handling
- Categorical encoding
- Feature scaling
- Multi-model training and comparison
- Automatic best-model selection
- Artifact generation for the website

## Run the Flask App

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

## Deploy Free on Render

Use the Flask app for the easiest free deployment.

If your repository root is the folder that contains `Weather_Project`, set:

- Root Directory: `Weather_Project`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`

If the repository root is already `Weather_Project`, leave Root Directory empty.

You can also deploy using the included `render.yaml`.

## Run the Django App

```bash
python manage.py migrate
python manage.py runserver
```

Open: `http://127.0.0.1:8000`

## EDA Outputs

After running `train_pipeline.py`, the following reports are generated:

- `reports/dataset_profile.json`
- `reports/dataset_summary.md`
- `reports/model_comparison.csv`
- `reports/plots/missing_values.png`
- `reports/plots/target_distribution.png`
- `reports/plots/correlation_heatmap.png`
- `reports/plots/model_comparison.png`

## Notes

- The frontend is shared between Flask and Django.
- Weather condition backgrounds are local SVG assets in `static/images`.
- Additional decorative icons are available in `static/icons`.
- Font Awesome icons are also used for animated dashboard feedback.
