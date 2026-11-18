"""King County house-price training pipeline.

Load, clean, engineer features, compare models, and persist the best
sklearn pipeline. This module is also imported by the Streamlit app for
reusable feature engineering at inference time.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "kc_house_data.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"

RAW_INPUT_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_basement",
    "sale_year",
    "sale_month",
]

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "yr_built",
    "yr_renovated",
    "lat",
    "long",
    "sqft_basement",
    "sale_year",
    "sale_month",
    "house_age",
    "renovated",
    "years_since_renovation",
    "living_lot_ratio",
    "basement_pct",
    "bathrooms_per_bedroom",
    "living_per_floor",
]

CATEGORICAL_FEATURES = ["zipcode"]
TARGET = "price"

# sqft_living15 / sqft_lot15 are excluded: they are not collected in the
# valuation form and would be unavailable at inference.
# id is excluded to prevent identifier leakage.
# date is converted to sale_year / sale_month only.


def get_data_path() -> Path:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Place kc_house_data.csv in data/."
        )
    return DATA_PATH


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or get_data_path()
    frame = pd.read_csv(csv_path)
    expected = {
        "id",
        "date",
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "zipcode",
        "lat",
        "long",
    }
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def validate_raw_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    profile = {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "missing_by_column": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_ids": int(frame["id"].duplicated().sum()),
        "price_min": float(frame["price"].min()),
        "price_max": float(frame["price"].max()),
        "price_mean": float(frame["price"].mean()),
        "price_median": float(frame["price"].median()),
        "zero_bedrooms": int((frame["bedrooms"] == 0).sum()),
        "zero_bathrooms": int((frame["bathrooms"] == 0).sum()),
        "extreme_bedrooms_33": int((frame["bedrooms"] == 33).sum()),
    }
    return profile


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues without inventing prices."""
    cleaned = frame.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date", "price", "sqft_living", "sqft_lot"])

    # Classic King County typo: 33 bedrooms in a ~1,620 sqft house.
    typo_mask = (cleaned["bedrooms"] == 33) & (cleaned["sqft_living"] < 3000)
    cleaned.loc[typo_mask, "bedrooms"] = 3

    invalid = (
        (cleaned["price"] <= 0)
        | (cleaned["sqft_living"] <= 0)
        | (cleaned["sqft_lot"] <= 0)
        | (cleaned["bedrooms"] < 0)
        | (cleaned["bathrooms"] < 0)
        | (cleaned["yr_built"] < 1800)
    )
    cleaned = cleaned.loc[~invalid].copy()

    # Studios / missing counts: drop records that cannot represent a livable house.
    cleaned = cleaned.loc[(cleaned["bedrooms"] > 0) & (cleaned["bathrooms"] > 0)].copy()

    above_plus_basement = cleaned["sqft_above"].fillna(0) + cleaned["sqft_basement"].fillna(0)
    inconsistent_area = above_plus_basement != cleaned["sqft_living"]
    # Dataset is internally consistent; keep a defensive clip if a future file is not.
    if inconsistent_area.any():
        cleaned.loc[inconsistent_area, "sqft_basement"] = np.maximum(
            cleaned.loc[inconsistent_area, "sqft_living"]
            - cleaned.loc[inconsistent_area, "sqft_above"].fillna(0),
            0,
        )

    cleaned["yr_renovated"] = cleaned["yr_renovated"].fillna(0).astype(int)
    cleaned["sqft_basement"] = cleaned["sqft_basement"].fillna(0).clip(lower=0)
    cleaned["zipcode"] = cleaned["zipcode"].astype(str)
    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    return cleaned


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe property features.

    sale_year / sale_month must already be present. They come from the
    historical sale date during training, and from a valuation date that
    stays inside the training market window during inference.
    """
    out = frame.copy()
    required = set(RAW_INPUT_COLUMNS)
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Missing raw columns for feature engineering: {sorted(missing)}")

    out["zipcode"] = out["zipcode"].astype(str)
    out["sale_year"] = out["sale_year"].astype(int)
    out["sale_month"] = out["sale_month"].astype(int)

    out["house_age"] = (out["sale_year"] - out["yr_built"]).clip(lower=0)
    out["renovated"] = (out["yr_renovated"] > 0).astype(int)
    years_since = np.where(
        out["yr_renovated"] > 0,
        out["sale_year"] - out["yr_renovated"],
        out["house_age"],
    )
    out["years_since_renovation"] = np.clip(years_since, a_min=0, a_max=None)
    out["living_lot_ratio"] = out["sqft_living"] / out["sqft_lot"].replace(0, np.nan)
    out["basement_pct"] = np.where(
        out["sqft_living"] > 0,
        out["sqft_basement"] / out["sqft_living"],
        0.0,
    )
    out["bathrooms_per_bedroom"] = out["bathrooms"] / out["bedrooms"].replace(0, np.nan)
    out["living_per_floor"] = out["sqft_living"] / out["floors"].replace(0, np.nan)

    out[NUMERIC_FEATURES] = out[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def prepare_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_dataset(raw)
    cleaned["sale_year"] = cleaned["date"].dt.year
    cleaned["sale_month"] = cleaned["date"].dt.month
    engineered = engineer_features(cleaned)
    return engineered


def chronological_split(
    frame: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(frame) * (1 - test_size))
    train_df = frame.iloc[:cutoff].copy()
    test_df = frame.iloc[cutoff:].copy()
    return train_df, test_df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, a_min=1.0, a_max=None)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
    }


def build_linear_pipeline() -> TransformedTargetRegressor:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    features = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    ridge = Pipeline(
        steps=[
            ("features", features),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=ridge, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def build_tree_pipeline(estimator) -> TransformedTargetRegressor:
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    features = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    pipe = Pipeline(steps=[("features", features), ("model", estimator)])
    return TransformedTargetRegressor(
        regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def extract_inner_estimator(model: TransformedTargetRegressor):
    return model.regressor_.named_steps["model"]


def extract_feature_names(model: TransformedTargetRegressor) -> np.ndarray:
    transformer = model.regressor_.named_steps["features"]
    return transformer.get_feature_names_out()


def permutation_style_importance(
    model: TransformedTargetRegressor,
    x_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Drop-column-style importance on original engineered inputs.

    More interpretable than OHE dummy importances for the product UI.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    sample_pos = rng.choice(len(x_holdout), size=min(2500, len(x_holdout)), replace=False)
    x_sample = x_holdout.iloc[sample_pos].copy()
    y_sample = y_holdout[sample_pos]
    baseline = regression_metrics(y_sample, model.predict(x_sample))["mae"]

    for column in columns:
        deltas = []
        original = x_sample[column].copy()
        for _ in range(n_repeats):
            shuffled = original.to_numpy().copy()
            rng.shuffle(shuffled)
            x_perm = x_sample.copy()
            x_perm[column] = shuffled
            perm_mae = regression_metrics(y_sample, model.predict(x_perm))["mae"]
            deltas.append(perm_mae - baseline)
        rows.append({"feature": column, "mae_increase": float(np.mean(deltas))})
    return pd.DataFrame(rows).sort_values("mae_increase", ascending=False)


def evaluate_split(
    name: str, model, x_train, y_train, x_test, y_test, cv_mae: float | None = None
) -> dict[str, Any]:
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    result = {
        "model": name,
        "train": regression_metrics(y_train, train_pred),
        "test": regression_metrics(y_test, test_pred),
        "cv_mae": cv_mae,
        "test_predictions": test_pred,
    }
    return result


def train_and_select_models(frame: pd.DataFrame) -> dict[str, Any]:
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_all = frame[feature_columns]
    y_all = frame[TARGET].to_numpy(dtype=float)

    chrono_train, chrono_test = chronological_split(frame)
    x_train = chrono_train[feature_columns]
    y_train = chrono_train[TARGET].to_numpy(dtype=float)
    x_test = chrono_test[feature_columns]
    y_test = chrono_test[TARGET].to_numpy(dtype=float)

    x_rand_train, x_rand_test, y_rand_train, y_rand_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=RANDOM_STATE
    )

    tscv = TimeSeriesSplit(n_splits=3)

    linear_model = build_linear_pipeline()
    linear_model.fit(x_train, y_train)

    rf_base = build_tree_pipeline(
        RandomForestRegressor(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions={
            "regressor__model__n_estimators": [120, 180, 240],
            "regressor__model__max_depth": [12, 16, 20],
            "regressor__model__min_samples_leaf": [2, 4, 8],
            "regressor__model__max_features": [0.5, 0.7, 1.0],
        },
        n_iter=6,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    rf_search.fit(x_train, y_train)
    rf_model = rf_search.best_estimator_

    xgb_base = build_tree_pipeline(
        XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    )
    xgb_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions={
            "regressor__model__n_estimators": [200, 300, 400],
            "regressor__model__max_depth": [3, 5, 6],
            "regressor__model__learning_rate": [0.05, 0.08, 0.12],
            "regressor__model__subsample": [0.7, 0.85],
            "regressor__model__min_child_weight": [1, 3, 5],
        },
        n_iter=8,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    xgb_search.fit(x_train, y_train)
    xgb_model = xgb_search.best_estimator_

    candidates = [
        evaluate_split(
            "Ridge (log-price baseline)", linear_model, x_train, y_train, x_test, y_test
        ),
        evaluate_split(
            "Random Forest",
            rf_model,
            x_train,
            y_train,
            x_test,
            y_test,
            cv_mae=float(-rf_search.best_score_),
        ),
        evaluate_split(
            "XGBoost",
            xgb_model,
            x_train,
            y_train,
            x_test,
            y_test,
            cv_mae=float(-xgb_search.best_score_),
        ),
    ]

    fitted = {
        "Ridge (log-price baseline)": linear_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
    }
    params = {
        "Ridge (log-price baseline)": {"alpha": 1.0},
        "Random Forest": rf_search.best_params_,
        "XGBoost": xgb_search.best_params_,
    }

    best = min(candidates, key=lambda row: row["test"]["mae"])
    best_name = best["model"]
    best_model = fitted[best_name]

    # Investigate random split vs chronological split on the winner only.
    random_clone = fitted[best_name]
    random_metrics = regression_metrics(y_rand_test, random_clone.predict(x_rand_test))
    # The random split above uses a model trained on chronological data, so also
    # fit a diagnostic clone on the random training fold for a fair comparison.
    diagnostic = build_tree_pipeline(
        XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    )
    if best_name == "Ridge (log-price baseline)":
        diagnostic = build_linear_pipeline()
    elif best_name == "Random Forest":
        diagnostic = build_tree_pipeline(
            RandomForestRegressor(
                n_estimators=180,
                max_depth=16,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        )
    diagnostic.fit(x_rand_train, y_rand_train)
    random_metrics = regression_metrics(y_rand_test, diagnostic.predict(x_rand_test))

    importance = permutation_style_importance(best_model, x_test, y_test)
    y_pred_test = best["test_predictions"]
    log_resid = np.log1p(y_test) - np.log1p(np.clip(y_pred_test, 1.0, None))
    conformal_q90 = float(np.quantile(np.abs(log_resid), 0.90))

    rng = np.random.default_rng(RANDOM_STATE)
    sample_n = min(2500, len(y_test))
    sample_idx = np.sort(rng.choice(len(y_test), size=sample_n, replace=False))

    comparison_table = []
    for row in candidates:
        comparison_table.append(
            {
                "model": row["model"],
                "train_mae": row["train"]["mae"],
                "train_rmse": row["train"]["rmse"],
                "train_r2": row["train"]["r2"],
                "test_mae": row["test"]["mae"],
                "test_rmse": row["test"]["rmse"],
                "test_r2": row["test"]["r2"],
                "test_mape": row["test"]["mape"],
                "cv_mae": row["cv_mae"],
            }
        )

    bundle = {
        "pipeline": best_model,
        "model_name": best_name,
        "best_params": params[best_name],
        "comparison": comparison_table,
        "chrono_test_metrics": best["test"],
        "random_split_diagnostic_metrics": random_metrics,
        "feature_importance": importance.to_dict(orient="records"),
        "conformal_q90_log": conformal_q90,
        "valuation_defaults": {
            "sale_year": int(frame["sale_year"].max()),
            "sale_month": int(
                frame.loc[frame["sale_year"] == frame["sale_year"].max(), "sale_month"].max()
            ),
        },
        "feature_medians": {
            col: (x_train[col].median() if col != "zipcode" else str(x_train[col].mode().iat[0]))
            for col in feature_columns
        },
        "zipcodes": sorted(frame["zipcode"].astype(str).unique().tolist()),
        "test_sample": {
            "y_true": y_test[sample_idx].tolist(),
            "y_pred": y_pred_test[sample_idx].tolist(),
            "residual": (y_test[sample_idx] - y_pred_test[sample_idx]).tolist(),
        },
        "split": {
            "strategy": "chronological 80/20 by sale date",
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "train_start": str(chrono_train["date"].min().date()),
            "train_end": str(chrono_train["date"].max().date()),
            "test_start": str(chrono_test["date"].min().date()),
            "test_end": str(chrono_test["date"].max().date()),
        },
        "random_state": RANDOM_STATE,
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "mape_note": (
            "MAPE over-weights inexpensive homes and is unstable near low prices. "
            "MAE and RMSE on USD are the primary selection metrics."
        ),
        "xgboost_justification": (
            "XGBoost is included because King County prices are nonlinear in living "
            "area, grade, waterfront, and location. A linear baseline and a bagged "
            "tree model are retained for comparison."
        ),
    }
    return bundle, comparison_table, best_name


def build_data_profile(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, Any]:
    profile = validate_raw_dataset(raw)
    numeric_cols = [
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "grade",
        "yr_built",
    ]
    profile["cleaned_rows"] = int(len(cleaned))
    profile["rows_removed"] = int(len(raw) - len(cleaned))
    profile["date_min"] = str(cleaned["date"].min().date())
    profile["date_max"] = str(cleaned["date"].max().date())
    profile["n_zipcodes"] = int(cleaned["zipcode"].nunique())
    profile["correlations_with_price"] = (
        cleaned[numeric_cols].corr(numeric_only=True)["price"].sort_values(ascending=False).to_dict()
    )
    q1 = cleaned["price"].quantile(0.25)
    q3 = cleaned["price"].quantile(0.75)
    iqr = q3 - q1
    profile["price_iqr_outlier_count"] = int(
        ((cleaned["price"] < q1 - 1.5 * iqr) | (cleaned["price"] > q3 + 1.5 * iqr)).sum()
    )
    return profile


def save_bundle(bundle: dict[str, Any]) -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return MODEL_PATH


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    model_path = path or MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Run feature_engineering.py first."
        )
    return joblib.load(model_path)


def prepare_inference_frame(payload: dict[str, Any], defaults: dict[str, int]) -> pd.DataFrame:
    row = {column: payload.get(column) for column in RAW_INPUT_COLUMNS}
    row["sale_year"] = int(payload.get("sale_year", defaults["sale_year"]))
    row["sale_month"] = int(payload.get("sale_month", defaults["sale_month"]))
    row["zipcode"] = str(row["zipcode"])
    frame = pd.DataFrame([row])
    return engineer_features(frame)[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def validate_prediction_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    bounds = {
        "bedrooms": (1, 12),
        "bathrooms": (0.5, 10),
        "sqft_living": (250, 15000),
        "sqft_lot": (400, 1500000),
        "floors": (1, 4),
        "waterfront": (0, 1),
        "view": (0, 4),
        "condition": (1, 5),
        "grade": (1, 13),
        "yr_built": (1900, 2015),
        "yr_renovated": (0, 2015),
        "lat": (47.0, 48.0),
        "long": (-123.0, -121.0),
        "sqft_basement": (0, 8000),
    }
    for key, (low, high) in bounds.items():
        if key not in payload or payload[key] is None:
            errors.append(f"{key} is required.")
            continue
        try:
            value = float(payload[key])
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric.")
            continue
        if value < low or value > high:
            errors.append(f"{key} must be between {low} and {high}.")
    if payload.get("sqft_basement", 0) and float(payload["sqft_basement"]) > float(
        payload.get("sqft_living", 0)
    ):
        errors.append("Basement area cannot exceed living area.")
    if int(payload.get("yr_renovated", 0) or 0) not in (0,) and int(payload["yr_renovated"]) < int(
        payload.get("yr_built", 1900)
    ):
        errors.append("Renovation year cannot be earlier than year built.")
    return errors


def run_training() -> dict[str, Any]:
    print("Loading dataset...")
    raw = load_raw_dataset()
    raw_profile = validate_raw_dataset(raw)
    print(json.dumps({k: v for k, v in raw_profile.items() if k != "missing_by_column"}, indent=2))

    print("Cleaning and engineering features...")
    prepared = prepare_training_frame(raw)
    profile = build_data_profile(raw, prepared)

    print(
        f"Training on {len(prepared)} cleaned records "
        f"(removed {profile['rows_removed']} invalid rows)."
    )
    bundle, comparison, best_name = train_and_select_models(prepared)
    bundle["data_profile"] = profile
    path = save_bundle(bundle)

    print("\nModel comparison (chronological holdout):")
    print(pd.DataFrame(comparison).to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nBest model: {best_name}")
    print("Chronological test metrics:", json.dumps(bundle["chrono_test_metrics"], indent=2))
    print("Random-split diagnostic:", json.dumps(bundle["random_split_diagnostic_metrics"], indent=2))
    print(f"Saved pipeline -> {path}")

    loaded = load_bundle()
    sample = prepared.iloc[[0]][NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    pred = float(loaded["pipeline"].predict(sample)[0])
    print(f"Sanity-check prediction on first training-like row: ${pred:,.0f}")
    return bundle


if __name__ == "__main__":
    run_training()
