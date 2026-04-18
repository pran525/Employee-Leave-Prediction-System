from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from retrain_model import evaluate_predictions, infer_default_as_of_date, load_forecasting_namespace


PROJECT_DIR = Path(__file__).resolve().parent
APP_PATH = PROJECT_DIR / "streamlit_app.py"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
ARCHIVE_DIR = ARTIFACT_DIR / "archive"
PRODUCTION_MODEL_PATH = ARTIFACT_DIR / "leave_forecasting_model.pkl"
PRODUCTION_METADATA_PATH = ARTIFACT_DIR / "leave_forecasting_metadata.pkl"
DATA_PATH = PROJECT_DIR / "Data" / "Combined_All_Leave_Data.csv"
EMPLOYEE_MASTER_PATH = PROJECT_DIR / "Employee Master - Feb 2026 Team Member.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate runnable ML artifacts for the leave forecasting dashboard.")
    parser.add_argument(
        "--as-of-date",
        help="Training cutoff date in YYYY-MM-DD format. Defaults to the latest approved leave date or today.",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=60,
        help="Number of days to generate in the saved forecast artifact.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=60000,
        help="Maximum number of model rows to keep for the fast bootstrap training run.",
    )
    return parser.parse_args()


def ensure_inputs() -> None:
    missing = [path for path in [DATA_PATH, EMPLOYEE_MASTER_PATH] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def build_training_frame(dataset_bundle: dict[str, object], max_rows: int) -> pd.DataFrame:
    model_df = dataset_bundle["model_df"].sort_values("Date").reset_index(drop=True)
    if len(model_df) > max_rows:
        model_df = model_df.tail(max_rows).reset_index(drop=True)
    if len(model_df) < 120:
        raise ValueError("Not enough model rows to build a usable forecast bundle.")
    return model_df


def save_artifact_bundle(
    model,
    dataset_bundle: dict[str, object],
    feature_columns: list[str],
    test_df: pd.DataFrame,
    y_test: pd.Series,
    test_predictions: np.ndarray,
    train_df: pd.DataFrame,
    y_train: pd.Series,
    train_predictions: np.ndarray,
    valid_df: pd.DataFrame,
    y_valid: pd.Series,
    valid_predictions: np.ndarray,
    forecast_horizon: int,
    as_of_date: str,
    best_model_name: str,
) -> None:
    test_comparison = pd.DataFrame(
        {
            "Date": test_df["Date"].to_numpy(),
            "Actual_Leave_Count": y_test.to_numpy(),
            "Predicted_Leave_Count": test_predictions,
            "Naive_Lag1_Prediction": np.clip(test_df["leave_lag_1"].to_numpy(), 0, None),
        }
    )
    test_comparison["Residual"] = test_comparison["Actual_Leave_Count"] - test_comparison["Predicted_Leave_Count"]
    test_comparison["Absolute_Error"] = test_comparison["Residual"].abs()

    metrics_df = pd.DataFrame(
        [
            evaluate_predictions(y_test, test_predictions, "Saved Forecast Model", {
                "weighted_absolute_percentage_error": lambda y_true, y_pred: float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()) if np.abs(y_true).sum() else 0.0,
                "mean_absolute_percentage_error_safe": lambda y_true, y_pred: float(np.mean(np.abs((np.asarray(y_true)[np.asarray(y_true) != 0] - np.asarray(y_pred)[np.asarray(y_true) != 0]) / np.asarray(y_true)[np.asarray(y_true) != 0]))) if np.any(np.asarray(y_true) != 0) else 0.0,
                "symmetric_mean_absolute_percentage_error": lambda y_true, y_pred: float(np.mean(2 * np.abs(np.asarray(y_true)[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0] - np.asarray(y_pred)[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0]) / (np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred)))[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0])) if np.any((np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0) else 0.0,
            }),
            evaluate_predictions(np.zeros_like(y_test), np.zeros_like(test_predictions), "Naive Lag-1 Baseline", {
                "weighted_absolute_percentage_error": lambda y_true, y_pred: float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()) if np.abs(y_true).sum() else 0.0,
                "mean_absolute_percentage_error_safe": lambda y_true, y_pred: float(np.mean(np.abs((np.asarray(y_true)[np.asarray(y_true) != 0] - np.asarray(y_pred)[np.asarray(y_true) != 0]) / np.asarray(y_true)[np.asarray(y_true) != 0]))) if np.any(np.asarray(y_true) != 0) else 0.0,
                "symmetric_mean_absolute_percentage_error": lambda y_true, y_pred: float(np.mean(2 * np.abs(np.asarray(y_true)[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0] - np.asarray(y_pred)[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0]) / (np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred)))[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0])) if np.any((np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0) else 0.0,
            }),
        ]
    )

    feature_importance_df = pd.DataFrame(columns=["feature", "importance"])
    if hasattr(model, "feature_importances_"):
        feature_importance_df = pd.DataFrame(
            {"feature": feature_columns, "importance": np.asarray(model.feature_importances_, dtype=float)}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

    residuals = test_comparison["Residual"].to_numpy()
    absolute_errors = test_comparison["Absolute_Error"].to_numpy()
    prediction_interval = {
        "residual_p05": float(np.quantile(residuals, 0.05)) if len(residuals) else 0.0,
        "residual_p95": float(np.quantile(residuals, 0.95)) if len(residuals) else 0.0,
        "absolute_error_p90": float(np.quantile(absolute_errors, 0.90)) if len(absolute_errors) else 0.0,
    }

    train_wape = float(np.abs(y_train.to_numpy() - train_predictions).sum() / np.abs(y_train.to_numpy()).sum()) if np.abs(y_train.to_numpy()).sum() else 0.0
    valid_wape = float(np.abs(y_valid.to_numpy() - valid_predictions).sum() / np.abs(y_valid.to_numpy()).sum()) if np.abs(y_valid.to_numpy()).sum() else 0.0
    test_wape = float(np.abs(y_test.to_numpy() - test_predictions).sum() / np.abs(y_test.to_numpy()).sum()) if np.abs(y_test.to_numpy()).sum() else 0.0
    model_balance = {
        "Training_WAPE": train_wape,
        "Validation_WAPE": valid_wape,
        "Test_WAPE": test_wape,
        "Generalization_Gap_WAPE": abs(valid_wape - test_wape),
        "Overfitting_Signal": valid_wape - train_wape,
        "Training_MAE": float(np.mean(np.abs(y_train.to_numpy() - train_predictions))),
        "Validation_MAE": float(np.mean(np.abs(y_valid.to_numpy() - valid_predictions))),
        "Test_MAE": float(np.mean(np.abs(y_test.to_numpy() - test_predictions))),
        "MAE_Gap": abs(float(np.mean(np.abs(y_valid.to_numpy() - valid_predictions))) - float(np.mean(np.abs(y_test.to_numpy() - test_predictions)))),
        "Training_R2": 0.0,
        "Validation_R2": 0.0,
        "Test_R2": 0.0,
        "Stability_Score": max(0.0, 1.0 - (abs(valid_wape - test_wape) / (test_wape + 0.001))),
    }

    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    version_prefix = f"leave_forecasting_bootstrap_{timestamp}"
    versioned_model_path = ARTIFACT_DIR / f"{version_prefix}.pkl"
    versioned_metadata_path = ARTIFACT_DIR / f"{version_prefix}_metadata.pkl"
    versioned_metrics_path = ARTIFACT_DIR / f"{version_prefix}_test_metrics.csv"
    versioned_predictions_path = ARTIFACT_DIR / f"{version_prefix}_test_predictions.csv"
    versioned_importance_path = ARTIFACT_DIR / f"{version_prefix}_feature_importance.csv"
    versioned_card_path = ARTIFACT_DIR / f"{version_prefix}_model_card.json"
    forecast_output_path = ARTIFACT_DIR / f"leave_forecast_next_{forecast_horizon}days_{timestamp}.csv"

    bundle_for_forecast = dict(dataset_bundle)
    bundle_for_forecast["model"] = model
    bundle_for_forecast["feature_columns"] = feature_columns
    bundle_for_forecast["metadata"] = {}
    future_frame = ns["iterative_forecast"](bundle_for_forecast, forecast_horizon).copy()
    future_frame["Predicted_Leave_Count"] = future_frame["Predicted_Leave_Count"].round().clip(lower=0).astype(int)
    future_frame["Day_of_Week"] = pd.to_datetime(future_frame["Date"]).dt.day_name()
    future_frame["Lower_Bound"] = np.maximum(future_frame["Predicted_Leave_Count"] - prediction_interval["absolute_error_p90"], 0).round().astype(int)
    future_frame["Upper_Bound"] = (future_frame["Predicted_Leave_Count"] + prediction_interval["absolute_error_p90"]).round().astype(int)

    metadata = {
        "best_model_name": best_model_name,
        "training_timestamp_utc": timestamp,
        "as_of_date": as_of_date,
        "training_start_date": str(train_df["Date"].min().date()),
        "training_end_date": str(pd.concat([train_df, valid_df])["Date"].max().date()),
        "test_start_date": str(test_df["Date"].min().date()),
        "test_end_date": str(test_df["Date"].max().date()),
        "feature_columns": feature_columns,
        "forecast_horizon": forecast_horizon,
        "current_live_headcount_from_master": dataset_bundle["current_live_headcount"],
        "validation_results": [
            {"Model": "Train", **evaluate_predictions(y_train, train_predictions, "Train", {
                "weighted_absolute_percentage_error": lambda y_true, y_pred: float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()) if np.abs(y_true).sum() else 0.0,
                "mean_absolute_percentage_error_safe": lambda y_true, y_pred: float(np.mean(np.abs((np.asarray(y_true)[np.asarray(y_true) != 0] - np.asarray(y_pred)[np.asarray(y_true) != 0]) / np.asarray(y_true)[np.asarray(y_true) != 0]))) if np.any(np.asarray(y_true) != 0) else 0.0,
                "symmetric_mean_absolute_percentage_error": lambda y_true, y_pred: float(np.mean(2 * np.abs(np.asarray(y_true)[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0] - np.asarray(y_pred)[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0]) / (np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred)))[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0])) if np.any((np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0) else 0.0,
            })},
            {"Model": "Validation", **evaluate_predictions(y_valid, valid_predictions, "Validation", {
                "weighted_absolute_percentage_error": lambda y_true, y_pred: float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()) if np.abs(y_true).sum() else 0.0,
                "mean_absolute_percentage_error_safe": lambda y_true, y_pred: float(np.mean(np.abs((np.asarray(y_true)[np.asarray(y_true) != 0] - np.asarray(y_pred)[np.asarray(y_true) != 0]) / np.asarray(y_true)[np.asarray(y_true) != 0]))) if np.any(np.asarray(y_true) != 0) else 0.0,
                "symmetric_mean_absolute_percentage_error": lambda y_true, y_pred: float(np.mean(2 * np.abs(np.asarray(y_true)[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0] - np.asarray(y_pred)[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0]) / (np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred)))[(np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0])) if np.any((np.abs(np.asarray(y_true)) + np.abs(np.asarray(y_pred))) != 0) else 0.0,
            })},
        ],
        "test_metrics": metrics_df.to_dict(orient="records"),
        "prediction_interval": prediction_interval,
        "model_balance": model_balance,
        "best_iteration": None,
        "walk_forward_folds": 0,
        "training_sample_weight_max": 1.0,
        "versioned_model_path": str(versioned_model_path),
        "versioned_metadata_path": str(versioned_metadata_path),
        "next_30_days_forecast": future_frame[["Date", "Day_of_Week", "Predicted_Leave_Count", "Lower_Bound", "Upper_Bound"]].assign(Date=lambda frame: pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
    }

    joblib.dump(model, versioned_model_path)
    joblib.dump(metadata, versioned_metadata_path)
    metrics_df.to_csv(versioned_metrics_path, index=False)
    test_comparison.to_csv(versioned_predictions_path, index=False)
    feature_importance_df.to_csv(versioned_importance_path, index=False)
    future_frame.to_csv(forecast_output_path, index=False)
    with open(versioned_card_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)

    shutil.copy2(versioned_model_path, PRODUCTION_MODEL_PATH)
    shutil.copy2(versioned_metadata_path, PRODUCTION_METADATA_PATH)

    print("Artifacts generated successfully.")
    print(f"Production model: {PRODUCTION_MODEL_PATH}")
    print(f"Production metadata: {PRODUCTION_METADATA_PATH}")
    print(f"Versioned model: {versioned_model_path}")
    print(f"Test metrics: {versioned_metrics_path}")
    print(f"Forecast export: {forecast_output_path}")


if __name__ == "__main__":
    args = parse_args()
    ensure_inputs()
    as_of_date = args.as_of_date or infer_default_as_of_date(DATA_PATH)
    ns = load_forecasting_namespace(APP_PATH)
    dataset_bundle = ns["build_feature_dataset"](PROJECT_DIR, as_of_date=as_of_date)
    model_df = build_training_frame(dataset_bundle, args.max_rows)

    n_rows = len(model_df)
    test_size = max(30, int(round(n_rows * 0.15)))
    valid_size = max(30, int(round(n_rows * 0.15)))
    train_size = n_rows - valid_size - test_size
    if train_size <= 0:
        raise ValueError("Not enough rows to create train/validation/test splits.")

    train_df = model_df.iloc[:train_size].copy()
    valid_df = model_df.iloc[train_size:train_size + valid_size].copy()
    test_df = model_df.iloc[train_size + valid_size:].copy()

    feature_columns = list(dataset_bundle["feature_columns"])
    X_train = train_df[feature_columns]
    y_train = train_df["Leave_Count"]
    X_valid = valid_df[feature_columns]
    y_valid = valid_df["Leave_Count"]
    X_test = test_df[feature_columns]
    y_test = test_df["Leave_Count"]

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=16,
        min_samples_leaf=3,
        min_samples_split=6,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_predictions = np.clip(model.predict(X_train), 0, None)
    valid_predictions = np.clip(model.predict(X_valid), 0, None)
    test_predictions = np.clip(model.predict(X_test), 0, None)

    save_artifact_bundle(
        model=model,
        dataset_bundle=dataset_bundle,
        feature_columns=feature_columns,
        test_df=test_df,
        y_test=y_test,
        test_predictions=test_predictions,
        train_df=train_df,
        y_train=y_train,
        train_predictions=train_predictions,
        valid_df=valid_df,
        y_valid=y_valid,
        valid_predictions=valid_predictions,
        forecast_horizon=args.forecast_horizon,
        as_of_date=as_of_date,
        best_model_name="RandomForestBootstrap",
    )
