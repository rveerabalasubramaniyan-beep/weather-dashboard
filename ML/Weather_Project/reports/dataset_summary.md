# Weather Dataset Analysis

- Dataset shape: 366 rows x 8 columns
- Target column: `RainTomorrow`

## Missing values

- MinTemp: 0
- MaxTemp: 0
- WindGustDir: 3
- WindGustSpeed: 2
- Humidity: 0
- Pressure: 0
- Temp: 0
- RainTomorrow: 0

## Model comparison

- XGBoost: accuracy=0.824, precision=0.500, recall=0.538, f1=0.518, roc_auc=0.773
- Logistic Regression: accuracy=0.716, precision=0.367, recall=0.846, f1=0.512, roc_auc=0.902
- Random Forest: accuracy=0.851, precision=0.625, recall=0.385, f1=0.476, roc_auc=0.809
- Decision Tree: accuracy=0.649, precision=0.259, recall=0.538, f1=0.350, roc_auc=0.639