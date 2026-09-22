"""
ML Model Training Script
========================
Loads the Institute Rankings dataset, engineers features, trains three
regression models (Linear Regression, Random Forest, K-Nearest Neighbors),
evaluates them, and persists the best model + preprocessing artifacts to disk.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "Institute_Rankings.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    for col in ("Name", "City", "State"):
        df[col] = df[col].astype(str).str.strip()
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df = df.dropna(subset=["Score", "Rank"]).copy()
    df = df.sort_values("Rank").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame):
    state_enc = LabelEncoder()
    city_enc = LabelEncoder()
    df["State_enc"] = state_enc.fit_transform(df["State"])
    df["City_enc"] = city_enc.fit_transform(df["City"])
    feature_cols = ["Score", "State_enc", "City_enc"]
    X = df[feature_cols].values
    y = df["Rank"].values
    return X, y, feature_cols, state_enc, city_enc


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        if name == "Random Forest":
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        else:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, preds)

        results[name] = {
            "model": model,
            "mae": float(mae),
            "mse": float(mse),
            "rmse": rmse,
            "r2": float(r2),
            "uses_scaler": name != "Random Forest",
        }
        print(f"  {name}: R2={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

    return results, scaler


def save_artifacts(results, scaler, state_enc, city_enc, feature_cols):
    best_name = max(results, key=lambda n: results[n]["r2"])
    best = results[best_name]

    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "state_encoder.pkl", "wb") as f:
        pickle.dump(state_enc, f)
    with open(MODEL_DIR / "city_encoder.pkl", "wb") as f:
        pickle.dump(city_enc, f)

    meta = {
        "best_model": best_name,
        "uses_scaler": best["uses_scaler"],
        "feature_cols": feature_cols,
        "metrics": {
            name: {
                "r2": v["r2"],
                "mae": v["mae"],
                "mse": v["mse"],
                "rmse": v["rmse"],
            }
            for name, v in results.items()
        },
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest model: {best_name} (R2={best['r2']:.4f})")
    print(f"Artifacts saved to {MODEL_DIR}/")
    return best_name


def main():
    print("=" * 60)
    print("  Institute Rank Prediction - Model Training")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    df = load_data()
    print(f"  Loaded {len(df)} institutes")

    print("\n[2/4] Engineering features...")
    X, y, feature_cols, state_enc, city_enc = engineer_features(df)
    print(f"  Features: {feature_cols}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    print("\n[3/4] Training models...")
    results, scaler = train_and_evaluate(X, y)

    print("\n[4/4] Saving artifacts...")
    save_artifacts(results, scaler, state_enc, city_enc, feature_cols)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
