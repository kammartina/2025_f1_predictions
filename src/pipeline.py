"""
F1Pipeline: top-level orchestrator that ties together collection,
feature engineering, and model training/prediction.

Typical workflow
────────────────
  # 1. First run: collect all 2025 data
  pipeline = F1Pipeline()
  pipeline.collect_season(2025)

  # 2. Train the model
  pipeline.train()

  # 3. Before Race 1 of 2026 (qualifying done, race hasn't started)
  pipeline.collect_round(2026, 1)          # collects qualifying, no race yet
  forecast = {"air_temp": 27, "track_temp": 40, "rainfall": 0, ...}
  predictions = pipeline.predict_race(2026, 1, weather_forecast=forecast)
  print(predictions)

  # 4. After Race 1 finishes, collect results and retrain
  pipeline.collect_round(2026, 1, force=True)   # now race results exist
  pipeline.train()
"""

from __future__ import annotations

import logging
from typing import Optional

import fastf1
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.collectors.session_collector import SessionCollector
from src.collectors.weather_collector import WeatherCollector
from src.db.database import F1Database
from src.features.feature_engineering import FeatureEngineer
from src.models.predictor import RacePredictor

logger = logging.getLogger(__name__)


class F1Pipeline:
    """
    High-level interface for the F1 prediction system.

    Parameters
    ----------
    db_path        : Path to the DuckDB database file
    cache_path     : Path for FastF1's local data cache
    model_path     : Where to save / load the trained XGBoost model
    recency_half_life : How quickly historical races are down-weighted
                        (see RacePredictor for details)
    """

    def __init__(
        self,
        db_path: str = "data/f1_data.db",
        cache_path: str = "f1_cache",
        model_path: str = "models/predictor.json",
        recency_half_life: int = 12,
    ) -> None:
        self.db_path = db_path
        self.cache_path = cache_path
        self.model_path = model_path
        self._predictor = RacePredictor(recency_half_life=recency_half_life)

        # Ensure the database and schema exist
        with F1Database(db_path) as db:
            db.create_tables()

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def collect_round(
        self,
        year: int,
        round_num: int,
        include_telemetry: bool = False,
        force: bool = False,
    ) -> None:
        """
        Fetch and store data for one race weekend.

        Set force=True to re-collect a round that is already in the database
        (useful after a race finishes to add actual race results).
        """
        with F1Database(self.db_path) as db:
            # 1. Race + qualifying data from FastF1
            collector = SessionCollector(db, cache_path=self.cache_path)
            collector.collect(year, round_num, include_telemetry=include_telemetry, force=force)

            # 2. Historical weather from Open-Meteo (race metadata must exist first)
            WeatherCollector(db).collect(year, round_num, force=force)

    def collect_season(
        self,
        year: int,
        include_telemetry: bool = False,
        force: bool = False,
    ) -> None:
        """
        Fetch and store data for every race in a season.

        Rounds that fail (e.g. future races not yet available) are logged
        and skipped so the rest of the season still gets collected.
        """
        schedule = self._get_schedule(year)
        if schedule is None:
            logger.error("Cannot retrieve schedule for %d.", year)
            return

        total = len(schedule)
        logger.info("Collecting %d rounds for the %d season …", total, year)

        with F1Database(self.db_path) as db:
            collector = SessionCollector(db, cache_path=self.cache_path)
            with logging_redirect_tqdm():
                with tqdm(
                    total=total,
                    desc=f"{year} season",
                    unit="round",
                    ncols=72,
                    colour="cyan",
                ) as season_bar:
                    for _, event in schedule.iterrows():
                        round_num = int(event["RoundNumber"])
                        name = event.get("EventName", f"Round {round_num}")
                        season_bar.set_postfix_str(name[:28])
                        try:
                            collector.collect(
                                year, round_num,
                                include_telemetry=include_telemetry,
                                force=force,
                            )
                            season_bar.set_postfix_str(f"{name[:20]} · weather")
                            WeatherCollector(db).collect(year, round_num, force=force)
                        except Exception as exc:
                            logger.warning("Skipping %s: %s", name, exc)
                        season_bar.update(1)

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def train(
        self,
        years: Optional[list[int]] = None,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Build the feature matrix and train the model.

        Parameters
        ----------
        years : Restrict training data to these seasons (None = all).
        save  : Whether to persist the model to disk after training.

        Returns
        -------
        Leave-One-Race-Out CV results DataFrame.
        """
        with F1Database(self.db_path) as db:
            engineer = FeatureEngineer(db)
            matrix = engineer.build_training_matrix(years=years)

        if matrix.empty:
            logger.error("No training data available. Run collect_season() first.")
            return pd.DataFrame()

        logger.info("Training on %d driver-race records …", len(matrix))
        cv_results, cv_predictions = self._predictor.evaluate_loro(matrix)
        self._predictor.train(matrix)

        if save:
            self._predictor.save(self.model_path)

        # Persist CV predictions (overwrite any previous run for these years)
        if not cv_predictions.empty:
            with F1Database(self.db_path) as db:
                trained_years = cv_predictions["year"].unique().tolist()
                for y in trained_years:
                    db.execute(
                        "DELETE FROM predictions WHERE year = ? AND source = 'cv'",
                        [int(y)],
                    )
                db.insert_df(cv_predictions, "predictions")

        return cv_results

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_race(
        self,
        year: int,
        round_num: int,
        weather_forecast: Optional[dict] = None,
        auto_weather: bool = True,
        load_model: bool = True,
    ) -> pd.DataFrame:
        """
        Predict the finishing order for an upcoming race.

        Qualifying data for this round must already be in the database.
        Call collect_round() after qualifying to load it.

        Parameters
        ----------
        year             : Race year
        round_num        : Race round number
        weather_forecast : Explicit weather dict to use, e.g.:
                           {"air_temp": 28.0, "track_temp": 42.0,
                            "humidity": 55.0, "pressure": 1013.0,
                            "rainfall": 0, "wind_speed": 3.0,
                            "wind_direction": 180.0}
                           If provided, this takes priority over auto_weather.
        auto_weather     : If True (default) and no weather_forecast is given,
                           automatically fetch a race-day forecast from
                           Open-Meteo. Falls back to stored historical weather
                           median if the forecast cannot be retrieved (e.g. race
                           is more than 16 days away).
        load_model       : Reload the saved model from disk before predicting.

        Returns
        -------
        DataFrame: driver_code | predicted_position, sorted 1 → 20.
        """
        # Auto-fetch Open-Meteo forecast unless the caller supplied weather manually
        if weather_forecast is None and auto_weather:
            with F1Database(self.db_path) as db:
                fetched = WeatherCollector(db).fetch_race_forecast(year, round_num)
            if fetched is not None:
                weather_forecast = fetched
                logger.info(
                    "Auto-fetched Open-Meteo forecast for %d R%d "
                    "(air=%.1f°C, rain=%d).",
                    year, round_num,
                    fetched.get("air_temp", float("nan")),
                    fetched.get("rainfall", 0),
                )
            else:
                logger.warning(
                    "Could not fetch Open-Meteo forecast for %d R%d. "
                    "Falling back to stored historical weather median.",
                    year, round_num,
                )

        if load_model:
            try:
                self._predictor.load(self.model_path)
            except Exception as exc:
                logger.warning("Could not load saved model (%s). Using in-memory model.", exc)

        with F1Database(self.db_path) as db:
            engineer = FeatureEngineer(db)
            features = engineer.build_prediction_features(
                year, round_num, weather_override=weather_forecast
            )

        if features is None or features.empty:
            logger.error(
                "No features built for %d R%d. "
                "Make sure qualifying data is collected first.",
                year, round_num,
            )
            return pd.DataFrame()

        predictions = self._predictor.predict(features)
        logger.info(
            "Predicted winner for %d R%d: %s",
            year, round_num, predictions.iloc[0]["driver_code"],
        )

        # Persist live predictions (overwrite any previous prediction for this race)
        with F1Database(self.db_path) as db:
            db.execute(
                "DELETE FROM predictions WHERE year = ? AND round = ? AND source = 'live'",
                [year, round_num],
            )
            live_df = predictions[["driver_code", "predicted_position"]].copy()
            live_df["year"]            = year
            live_df["round"]           = round_num
            live_df["actual_position"] = None
            live_df["source"]          = "live"
            db.insert_df(live_df, "predictions")

        return predictions

    # ------------------------------------------------------------------
    # Evaluation & reporting
    # ------------------------------------------------------------------

    def evaluate(self, years: Optional[list[int]] = None) -> pd.DataFrame:
        """
        Run Leave-One-Race-Out CV and return detailed accuracy report.
        Does NOT retrain the production model.
        """
        with F1Database(self.db_path) as db:
            engineer = FeatureEngineer(db)
            matrix = engineer.build_training_matrix(years=years)

        if matrix.empty:
            logger.error("No data for evaluation.")
            return pd.DataFrame()

        return self._predictor.evaluate_loro(matrix)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances from the trained model."""
        return self._predictor.feature_importance()

    def database_summary(self) -> None:
        """Print a quick summary of what is in the database."""
        with F1Database(self.db_path) as db:
            races = db.get_races()
            print(f"\n{'─' * 50}")
            print(f"  Database: {self.db_path}")
            print(f"  Races collected: {len(races)}")
            if not races.empty:
                for year, group in races.groupby("year"):
                    print(f"    {year}: {len(group)} rounds")
            lap_count = db.query("SELECT COUNT(*) AS n FROM lap_data")["n"].iloc[0]
            print(f"  Lap rows: {int(lap_count):,}")
            print(f"{'─' * 50}\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_schedule(year: int) -> Optional[pd.DataFrame]:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            return schedule[schedule["RoundNumber"] > 0].reset_index(drop=True)
        except Exception as exc:
            logger.error("Failed to get %d schedule: %s", year, exc)
            return None
