"""
XGBoost-based demand forecasting engine.
Clean, stable, and safe for synthetic or sparse data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor


class ForecastingEngine:
    def __init__(self, sales_path="../data/sales.csv"):
        # Resolve project root and data path
        root = Path(__file__).resolve().parents[1]
        self.sales_path = root / "data" / "sales.csv"

        self.model = None
        self.df = None

    # ---------------------------------------------------------
    # Load and prepare sales data
    # ---------------------------------------------------------
    def load_sales(self):
        """Load and aggregate weekly demand."""
        df = pd.read_csv(self.sales_path)

        # Aggregate quantity per product per week
        df = df.groupby(["week", "product_id"])["quantity"].sum().reset_index()

        self.df = df
        return df

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------
    def _add_features(self, df):
        """Create lag, rolling, and cyclical features."""
        df = df.sort_values(["product_id", "week"])

        # Lag features
        for lag in [1, 2, 3]:
            df[f"lag_{lag}"] = df.groupby("product_id")["quantity"].shift(lag)

        # Rolling windows
        df["rolling_3"] = (
            df.groupby("product_id")["quantity"].shift(1).rolling(3).mean()
        )
        df["rolling_6"] = (
            df.groupby("product_id")["quantity"].shift(1).rolling(6).mean()
        )

        # Cyclical encoding for week number
        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

        return df

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------
    def train(self):
        """Train a global XGBoost model on all products."""
        df = self._add_features(self.df.copy())

        # Drop rows with NaNs ONLY during training
        df = df.dropna()

        feature_cols = [
            "product_id",
            "lag_1", "lag_2", "lag_3",
            "rolling_3", "rolling_6",
            "week_sin", "week_cos"
        ]

        X = df[feature_cols]
        y = df["quantity"]

        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror"
        )

        self.model.fit(X, y)
        return self.model

    # ---------------------------------------------------------
    # Forecast future demand
    # ---------------------------------------------------------
    def predict(self, product_id, weeks_ahead=12):
        """Forecast future demand for a product."""
        df = self.df.copy()
        last_week = df["week"].max()

        # Build future frame
        future = pd.DataFrame({
            "week": range(last_week + 1, last_week + weeks_ahead + 1),
            "product_id": product_id,
            "quantity": np.nan  # placeholder for feature creation
        })

        # Combine history + future
        hist = df[df["product_id"] == product_id].copy()
        combined = pd.concat([hist, future], ignore_index=True)

        # Add features
        combined = self._add_features(combined)

        # Forward-fill lag/rolling features so future rows are valid
        for col in ["lag_1", "lag_2", "lag_3", "rolling_3", "rolling_6"]:
            combined[col] = combined[col].fillna(method="ffill").fillna(0)

        # Only predict future rows
        future_rows = combined[combined["week"] > last_week]

        feature_cols = [
            "product_id",
            "lag_1", "lag_2", "lag_3",
            "rolling_3", "rolling_6",
            "week_sin", "week_cos"
        ]

        preds = self.model.predict(future_rows[feature_cols])

        return pd.DataFrame({
            "week": future_rows["week"],
            "predicted_quantity": preds
        })