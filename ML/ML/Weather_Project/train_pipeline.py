"""Train the weather prediction pipeline and generate EDA artifacts.

This script analyzes the uploaded CSV dataset, automatically selects an
appropriate target column for weather prediction, compares multiple machine
learning models, and saves the best-performing pipeline for the web apps.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback is used when seaborn is absent.
    sns = None


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "weather.csv"
MODEL_DIR = ROOT_DIR / "model"
REPORTS_DIR = ROOT_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"


def ensure_directories() -> None:
    """Create the output directories used by the training pipeline."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """Load the uploaded weather dataset into a DataFrame."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset at {DATA_PATH}, but the file was not found."
        )
    return pd.read_csv(DATA_PATH)


def select_target_column(df: pd.DataFrame) -> str:
    """Automatically identify the most suitable target column.

    The heuristic prefers weather-style labels such as rain/condition/target,
    then falls back to low-cardinality columns that look like classification
    targets.
    """

    keywords = (
        "target",
        "label",
        "class",
        "condition",
        "weather",
        "rain",
        "forecast",
        "prediction",
    )

    candidates: list[tuple[int, float, int, str]] = []
    row_count = len(df)

    for position, column in enumerate(df.columns):
        series = df[column]
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(row_count, 1)
        is_low_cardinality = 2 <= unique_count <= min(20, max(4, int(row_count * 0.2)))

        if not is_low_cardinality:
            continue

        column_name = column.lower()
        keyword_score = sum(keyword in column_name for keyword in keywords)
        dtype_score = 1 if series.dtype == "object" else 0
        score = (keyword_score * 10) + (dtype_score * 3) + int(unique_ratio < 0.1)
        candidates.append((score, unique_ratio, position, column))

    if not candidates:
        raise ValueError(
            "Unable to automatically identify a suitable categorical target column."
        )

    candidates.sort(key=lambda item: (item[0], -item[1], item[2]), reverse=True)
    return candidates[0][3]


def classify_problem(series: pd.Series) -> str:
    """Determine whether the selected target should be treated as classification."""

    unique_count = series.nunique(dropna=True)
    unique_ratio = unique_count / max(len(series), 1)
    if series.dtype == "object" or unique_count <= 20 or unique_ratio < 0.1:
        return "classification"
    raise ValueError(
        "The detected target does not look like a classification label, "
        "which is required by the requested model set."
    )


def create_preprocessor(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build the preprocessing pipeline for numeric and categorical features."""

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def get_models(random_state: int = 42) -> dict[str, Any]:
    """Return the candidate models requested for comparison."""

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            random_state=random_state,
            class_weight="balanced",
        ),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
) -> tuple[pd.DataFrame, Pipeline, str]:
    """Train, evaluate, and rank the candidate models."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    evaluation_rows: list[dict[str, Any]] = []
    best_pipeline: Pipeline | None = None
    best_model_name = ""
    best_score = (-np.inf, -np.inf, -np.inf)

    for model_name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        evaluation_rows.append(row)

        comparison_key = (row["f1"], row["accuracy"], row["roc_auc"])
        if comparison_key > best_score:
            best_score = comparison_key
            best_pipeline = pipeline
            best_model_name = model_name

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    results = pd.DataFrame(evaluation_rows).sort_values(
        by=["f1", "accuracy", "roc_auc"],
        ascending=False,
    )
    return results, best_pipeline, best_model_name


def default_inputs(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Build sensible default inputs from the dataset distribution."""

    defaults: dict[str, Any] = {}
    for column in feature_columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            defaults[column] = float(series.median())
        else:
            defaults[column] = str(series.mode(dropna=True).iloc[0])
    return defaults


def build_profile(
    df: pd.DataFrame,
    target_column: str,
    results: pd.DataFrame,
    numeric_features: list[str],
) -> dict[str, Any]:
    """Create the analysis summary written to disk and referenced by the apps."""

    correlations = df[numeric_features].corr(numeric_only=True).round(3).to_dict()
    summary_statistics = df.describe(include="all").fillna("").round(3).to_dict()

    return {
        "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "target_column": target_column,
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "data_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "summary_statistics": summary_statistics,
        "correlations": correlations,
        "model_results": results.round(4).to_dict(orient="records"),
    }


def save_profile(profile: dict[str, Any]) -> None:
    """Persist the EDA summary as JSON and Markdown."""

    (REPORTS_DIR / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Weather Dataset Analysis",
        "",
        f"- Dataset shape: {profile['dataset_shape']['rows']} rows x "
        f"{profile['dataset_shape']['columns']} columns",
        f"- Target column: `{profile['target_column']}`",
        "",
        "## Missing values",
        "",
    ]
    for column, missing in profile["missing_values"].items():
        lines.append(f"- {column}: {missing}")

    lines.extend(
        [
            "",
            "## Model comparison",
            "",
        ]
    )
    for row in profile["model_results"]:
        lines.append(
            "- {model}: accuracy={accuracy:.3f}, precision={precision:.3f}, "
            "recall={recall:.3f}, f1={f1:.3f}, roc_auc={roc_auc:.3f}".format(**row)
        )

    (REPORTS_DIR / "dataset_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def set_plot_style() -> None:
    """Apply a clean plotting theme shared by all charts."""

    if sns is not None:
        sns.set_theme(style="whitegrid", palette="crest")
    else:
        plt.style.use("ggplot")


def save_plots(
    df: pd.DataFrame,
    target_column: str,
    numeric_features: list[str],
    model_results: pd.DataFrame,
) -> None:
    """Generate exploratory plots and a model comparison chart."""

    set_plot_style()

    missing = df.isna().sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(missing.index, missing.values, color="#f28c28")
    ax.set_title("Missing Values by Column")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "missing_values.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    target_counts = df[target_column].value_counts()
    ax.bar(target_counts.index.astype(str), target_counts.values, color="#2a9d8f")
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "target_distribution.png", dpi=200)
    plt.close(fig)

    correlation = df[numeric_features].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(correlation, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    else:
        image = ax.imshow(correlation, cmap="YlGnBu")
        ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
        ax.set_yticks(range(len(correlation.index)), correlation.index)
        fig.colorbar(image, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = model_results.sort_values(by="f1", ascending=True)
    ax.barh(ordered["model"], ordered["f1"], color="#264653")
    ax.set_title("Model Comparison by F1 Score")
    ax.set_xlabel("F1 Score")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)


def save_artifacts(
    df: pd.DataFrame,
    target_column: str,
    label_encoder: LabelEncoder,
    numeric_features: list[str],
    categorical_features: list[str],
    model_results: pd.DataFrame,
    best_model_name: str,
    best_pipeline: Pipeline,
) -> None:
    """Save the training artifacts consumed by Flask and Django."""

    feature_columns = [column for column in df.columns if column != target_column]
    preprocessing_payload = {
        "target_column": target_column,
        "target_classes": label_encoder.classes_.tolist(),
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_options": {
            column: sorted(df[column].dropna().astype(str).unique().tolist())
            for column in categorical_features
        },
        "default_inputs": default_inputs(df, feature_columns),
        "best_model_name": best_model_name,
        "model_results": model_results.round(6).to_dict(orient="records"),
        "notes": {
            "condition_mapping": (
                "The dataset target predicts rain/no-rain, so the dashboard's "
                "climate condition is derived from rain probability plus the "
                "live weather inputs."
            )
        },
    }

    joblib.dump(best_pipeline, MODEL_DIR / "weather_model.pkl")
    joblib.dump(preprocessing_payload, MODEL_DIR / "preprocessing.pkl")
    (MODEL_DIR / "metrics.json").write_text(
        model_results.round(6).to_json(orient="records", indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Execute the end-to-end analysis and training workflow."""

    warnings.filterwarnings("ignore")
    ensure_directories()

    df = load_dataset()
    target_column = select_target_column(df)
    problem_type = classify_problem(df[target_column])

    if problem_type != "classification":
        raise ValueError("Only classification workflows are supported by this project.")

    feature_columns = [column for column in df.columns if column != target_column]
    X = df[feature_columns].copy()
    y_raw = df[target_column].astype(str)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    preprocessor, numeric_features, categorical_features = create_preprocessor(X)
    results, best_pipeline, best_model_name = evaluate_models(X, y, preprocessor)

    save_artifacts(
        df=df,
        target_column=target_column,
        label_encoder=label_encoder,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_results=results,
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
    )

    profile = build_profile(df, target_column, results, numeric_features)
    save_profile(profile)
    save_plots(df, target_column, numeric_features, results)
    results.round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    print(
        f"Training complete. Best model: {best_model_name}. "
        f"Artifacts saved to {MODEL_DIR}."
    )


if __name__ == "__main__":
    main()
