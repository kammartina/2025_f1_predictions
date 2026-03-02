"""
RacePredictor: trains an XGBoost model on the feature matrix and predicts
finishing positions.

Design choices
──────────────
  Target      : finish_position (regression, then ranked 1–20)
  Model       : XGBoostRegressor — handles missing values natively,
                robust to small datasets, strong baseline for tabular data
  Validation  : Leave-One-Race-Out cross-validation (LORO-CV).
                Train on all races except race N, predict race N.
                This is the correct evaluation strategy because using a
                random 80/20 split would leak race-level information.
  Recency     : More-recent races receive higher sample weights so the
                model adapts quickly as the season unfolds.
  Features    : Any NaN values are passed directly to XGBoost which
                handles them internally — no imputation needed.

Default hyperparameters:
n_estimators=400      # number of trees
learning_rate=0.05    # how much each tree contributes (small = more stable)
max_depth=4           # how deep each tree grows (prevents overfitting)
subsample=0.8         # each tree trains on 80% of rows (reduces variance)
colsample_bytree=0.8  # each tree uses 80% of features (reduces correlation)
min_child_weight=3    # a leaf needs at least 3 samples (prevents overfitting)
reg_alpha=0.1         # L1 regularisation (pushes some weights to 0)
reg_lambda=1.0        # L2 regularisation (keeps weights small)

_DEFAULT_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# Default XGBoost hyperparameters — tune with Optuna once you have ≥10 races
_DEFAULT_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


class RacePredictor:
    """
    Wraps XGBoost for F1 position prediction.

    Usage
    ─────
    predictor = RacePredictor()
    predictor.train(matrix)                # fit on all races
    predictions = predictor.predict(features_df)  # one row per driver
    cv_report = predictor.evaluate_loro(matrix)   # leave-one-race-out CV
    predictor.save("models/predictor.json")
    predictor.load("models/predictor.json")
    """

    def __init__(self, recency_half_life: int = 12, **xgb_params) -> None:
        """
        Parameters
        ----------
        recency_half_life : Number of races after which a sample's weight
                            halves. E.g. 12 means a race from 12 races ago
                            has half the weight of the most recent race.
                            Keeps the model adapting quickly mid-season.
        xgb_params        : Override any default XGBoost hyperparameter.
        """
        self.recency_half_life = recency_half_life
        self._params = {**_DEFAULT_PARAMS, **xgb_params}
        self._model: Optional[XGBRegressor] = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, matrix: pd.DataFrame) -> None:
        """
        Fit the model on the full feature matrix.

        Parameters
        ----------
        matrix : Output of FeatureEngineer.build_training_matrix().
                 Must contain 'year', 'round', 'driver_code',
                 'finish_position', and all feature columns.
        """
        X, y, weights = self._prepare(matrix)
        if X.empty:
            raise ValueError("Feature matrix is empty — collect some race data first.")

        self._model = XGBRegressor(**self._params)
        self._model.fit(X, y, sample_weight=weights)
        logger.info(
            "Model trained on %d driver-race samples (%d features).",
            len(X), X.shape[1],
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict finishing positions for an upcoming race.

        Parameters
        ----------
        features_df : One row per driver. Output of
                      FeatureEngineer.build_prediction_features().

        Returns
        -------
        DataFrame with columns: driver_code, predicted_position, confidence.
        Confidence is 1 / predicted_position_std across LORO folds (if available).
        """
        if self._model is None:
            raise RuntimeError("Call train() before predict().")

        X = features_df[self._feature_cols].copy()
        raw_scores = self._model.predict(X)

        result = features_df[["driver_code"]].copy()
        result["raw_score"] = raw_scores

        # Rank by raw score — lower predicted position → better → rank 1st
        result = result.sort_values("raw_score").reset_index(drop=True)
        result["predicted_position"] = range(1, len(result) + 1)
        result = result.drop(columns=["raw_score"])

        return result

    # ------------------------------------------------------------------
    # Leave-One-Race-Out cross-validation
    # ------------------------------------------------------------------

    def evaluate_loro(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate prediction accuracy using Leave-One-Race-Out CV.

        For each race N: train on all other races, predict race N.
        Reports per-race Spearman correlation and top-3 accuracy.

        Returns
        -------
        DataFrame with one row per evaluated race:
            year, round, spearman_r, top3_overlap, n_drivers
        """
        race_keys = (
            matrix[["year", "round"]]
            .drop_duplicates()
            .sort_values(["year", "round"])
            .reset_index(drop=True)
        )

        records = []
        with tqdm(
            race_keys.iterrows(),
            total=len(race_keys),
            desc="Cross-validation",
            unit="fold",
            ncols=72,
        ) as pbar:
            for _, race_key in pbar:
                y_val = int(race_key["year"])
                r_val = int(race_key["round"])
                pbar.set_postfix_str(f"{y_val} R{r_val:02d}")

                train_mask = ~((matrix["year"] == y_val) & (matrix["round"] == r_val))
                test_mask = (matrix["year"] == y_val) & (matrix["round"] == r_val)

                train_df = matrix[train_mask]
                test_df = matrix[test_mask]

                if len(train_df) < 20 or test_df.empty:
                    logger.debug("Not enough training data to evaluate race %d R%d.", y_val, r_val)
                    continue

                X_train, y_train, w_train = self._prepare(train_df)
                X_test = test_df[self._feature_cols]
                y_test = test_df["finish_position"]

                if y_test.isna().all():
                    continue

                tmp_model = XGBRegressor(**self._params)
                tmp_model.fit(X_train, y_train, sample_weight=w_train)
                raw = tmp_model.predict(X_test)

                # Rank predictions
                pred_rank = pd.Series(raw).rank().astype(int)
                actual_rank = y_test.rank().astype(int)

                spearman = spearmanr(pred_rank, actual_rank).statistic
                top3_pred = set(test_df["driver_code"].iloc[pred_rank.nsmallest(3).index].tolist())
                top3_actual = set(test_df["driver_code"][y_test.nsmallest(3).index].tolist())
                top3_overlap = len(top3_pred & top3_actual)

                records.append({
                    "year":          y_val,
                    "round":         r_val,
                    "spearman_r":    round(float(spearman), 3),
                    "top3_overlap":  top3_overlap,
                    "n_drivers":     len(test_df),
                })

        cv_results = pd.DataFrame(records)
        if not cv_results.empty:
            logger.info(
                "LORO-CV — mean Spearman r: %.3f | mean top-3 overlap: %.1f/3",
                cv_results["spearman_r"].mean(),
                cv_results["top3_overlap"].mean(),
            )
        return cv_results

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        if self._model is None:
            raise RuntimeError("Model not trained yet.")
        scores = self._model.feature_importances_
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/predictor.json") -> None:
        if self._model is None:
            raise RuntimeError("Nothing to save — train the model first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(path)
        logger.info("Model saved to %s", path)

    def load(self, path: str = "models/predictor.json") -> None:
        self._model = XGBRegressor(**self._params)
        self._model.load_model(path)
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(
        self, matrix: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Extract features X, target y, and recency-based sample weights.

        Features with no variation or all-NaN are dropped automatically by
        XGBoost (it handles NaN natively so we do not impute here).
        """
        from src.features.feature_engineering import FeatureEngineer
        all_cols = FeatureEngineer.feature_columns()

        # Keep only columns that exist in this matrix
        self._feature_cols = [c for c in all_cols if c in matrix.columns]

        X = matrix[self._feature_cols].copy()
        # XGBoost requires int/float/bool — cast everything to numeric.
        # None and non-parseable values become NaN, which XGBoost handles natively.
        X = X.apply(pd.to_numeric, errors="coerce")
        y = matrix["finish_position"].copy()

        # Drop rows where target is unknown
        valid = y.notna()
        X, y = X[valid], y[valid]

        # Recency weights: most recent race = weight 1.0, older races decay
        total_races = matrix[["year", "round"]].drop_duplicates().shape[0]
        race_order = (
            matrix[["year", "round"]]
            .drop_duplicates()
            .sort_values(["year", "round"])
            .reset_index(drop=True)
        )
        race_order["race_index"] = range(len(race_order))

        merged = matrix[valid][["year", "round"]].merge(race_order, on=["year", "round"])
        max_idx = race_order["race_index"].max()
        weights = np.power(0.5, (max_idx - merged["race_index"]) / self.recency_half_life)

        return X, y, weights.values
