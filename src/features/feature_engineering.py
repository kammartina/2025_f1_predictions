"""
FeatureEngineer: builds one feature-matrix row per (driver, race) from the
data stored in F1Database.

All features use only information that would have been available BEFORE the
race being predicted — this prevents data leakage.

Feature groups
──────────────
  Qualifying      : quali_time, quali_position
  Sector pace     : avg sector times at same circuit type (historical)
  Race pace       : clean-air pace (historical), historical avg lap time
  Tire degradation: slope of lap-time vs tire-age per compound (historical)
  Pit stops       : team avg pit duration (historical)
  Standing        : constructor points, driver points (cumulative at time of race)
  Driver form     : rolling avg finishing position over last 3 races
  Reliability     : historical DNF rate
  Weather         : air/track temp, humidity, pressure, rainfall, wind
  Circuit         : encoded circuit type, historical safety-car probability
  Telemetry       : avg speed, max speed, throttle%, brake%, DRS% (historical, optional)
"""

from __future__ import annotations

import logging
from typing import Optional

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from src.db.database import F1Database
from src.db.schema import CIRCUIT_TYPE_ENCODING

logger = logging.getLogger(__name__)

# Track-status codes that indicate caution (Safety Car, VSC, Yellow, Red)
_CAUTION_STATUSES = {"2", "4", "5", "6", "7"}


class FeatureEngineer:
    """
    Builds the training/prediction feature matrix from the F1Database.

    Parameters
    ----------
    db : F1Database  (must already be connected)
    """

    def __init__(self, db: F1Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_training_matrix(self, years: Optional[list[int]] = None) -> pd.DataFrame:
        """
        Build the complete feature matrix for all collected races.

        Each row = one driver in one race.
        The target column is `finish_position`.

        Parameters
        ----------
        years : If provided, restrict to these seasons (e.g. [2025, 2026]).
                None means use everything in the database.
        """
        races = self.db.get_races()
        if years:
            races = races[races["year"].isin(years)]
        races = races.sort_values(["year", "round"]).reset_index(drop=True)

        all_rows: list[pd.DataFrame] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", FutureWarning)
            with tqdm(
                races.iterrows(),
                total=len(races),
                desc="Building features",
                unit="race",
                ncols=72,
            ) as pbar:
                for _, race in pbar:
                    year, round_num = int(race["year"]), int(race["round"])
                    name = str(race.get("circuit_name", f"R{round_num}"))
                    pbar.set_postfix_str(f"{year} R{round_num:02d} {name[:18]}")
                    features = self._build_race_features(year, round_num, races)
                    if features is not None and not features.empty:
                        all_rows.append(features)

            if not all_rows:
                return pd.DataFrame()

            matrix = pd.concat(all_rows, ignore_index=True)
        logger.info(
            "Feature matrix: %d rows × %d columns", len(matrix), matrix.shape[1]
        )
        return matrix

    def build_prediction_features(
        self,
        year: int,
        round_num: int,
        weather_override: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Build features for an upcoming race (no results exist yet).

        Requires qualifying data to already be stored in the database.
        Pass weather_override to inject pre-race forecast values, e.g.:
            {"air_temp": 28.0, "track_temp": 42.0, "rainfall": 0, ...}

        Returns a DataFrame with one row per driver (no target column).
        """
        all_races = self.db.get_races()
        return self._build_race_features(
            year, round_num, all_races, weather_override=weather_override
        )

    # ------------------------------------------------------------------
    # Per-race feature building
    # ------------------------------------------------------------------

    def _build_race_features(
        self,
        year: int,
        round_num: int,
        all_races: pd.DataFrame,
        weather_override: Optional[dict] = None,
    ) -> Optional[pd.DataFrame]:
        """Assemble features for all drivers in one race."""

        quali = self.db.get_qualifying_results(year, round_num)
        if quali.empty:
            logger.debug("No qualifying data for %d R%d — skipping.", year, round_num)
            return None

        # All races strictly before this one (for historical features)
        prior_mask = (all_races["year"] < year) | (
            (all_races["year"] == year) & (all_races["round"] < round_num)
        )
        prior_races = all_races[prior_mask].copy()

        race_meta = all_races[
            (all_races["year"] == year) & (all_races["round"] == round_num)
        ]
        circuit_type = race_meta["circuit_type"].iloc[0] if not race_meta.empty else None

        rows = []
        for _, q in quali.iterrows():
            driver = q["driver_code"]
            row = self._driver_race_features(
                driver=driver,
                year=year,
                round_num=round_num,
                quali_row=q,
                prior_races=prior_races,
                circuit_type=circuit_type,
                weather_override=weather_override,
            )
            rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)

        # Attach target if the race has already been run
        results = self.db.get_session_results(year, round_num)
        if not results.empty:
            df = df.merge(
                results[["driver_code", "finish_position"]],
                on="driver_code",
                how="left",
            )

        # Normalise championship points within-race
        for col in ["constructor_points", "driver_points"]:
            if col in df.columns and df[col].max() > 0:
                df[f"{col}_norm"] = df[col] / df[col].max()
            else:
                df[f"{col}_norm"] = 0.0

        df["year"] = year
        df["round"] = round_num
        return df

    def _driver_race_features(
        self,
        driver: str,
        year: int,
        round_num: int,
        quali_row: pd.Series,
        prior_races: pd.DataFrame,
        circuit_type: Optional[str],
        weather_override: Optional[dict],
    ) -> dict:
        row: dict = {
            "driver_code": driver,
            "circuit_type_enc": CIRCUIT_TYPE_ENCODING.get(circuit_type or "", -1),
        }

        # --- Qualifying features ---
        row.update(self._qualifying_features(quali_row))

        # --- Historical race-pace features (prior races, same circuit type) ---
        row.update(self._historical_pace_features(driver, prior_races, circuit_type))

        # --- Tire degradation ---
        row.update(self._tire_deg_features(driver, prior_races))

        # --- Pit stop duration (team-level) ---
        team = self._get_driver_team(driver, year, round_num)
        row.update(self._pit_stop_features(team, prior_races))

        # --- Championship standings ---
        row.update(self._standing_features(driver, team, year, round_num))

        # --- Driver form ---
        row.update(self._driver_form_features(driver, prior_races))

        # --- Reliability ---
        row.update(self._reliability_features(driver, prior_races))

        # --- Weather ---
        row.update(self._weather_features(year, round_num, weather_override))

        # --- Circuit safety-car probability ---
        row.update(self._circuit_sc_probability(prior_races, circuit_type))

        # --- Telemetry summary (if collected) ---
        row.update(self._telemetry_features(driver, prior_races, circuit_type))

        return row

    # ------------------------------------------------------------------
    # Feature group methods
    # ------------------------------------------------------------------

    def _qualifying_features(self, quali_row: pd.Series) -> dict:
        q3 = quali_row.get("q3_time")
        q2 = quali_row.get("q2_time")
        q1 = quali_row.get("q1_time")
        best_time = next((t for t in [q3, q2, q1] if t and not pd.isna(t)), None)

        return {
            "quali_time":     best_time,
            "quali_position": quali_row.get("qualifying_position"),
            "q3_time":        q3 if not pd.isna(q3) else None,
        }

    def _historical_pace_features(
        self,
        driver: str,
        prior_races: pd.DataFrame,
        circuit_type: Optional[str],
    ) -> dict:
        if prior_races.empty:
            return {
                "clean_air_pace": None,
                "avg_sector1":    None,
                "avg_sector2":    None,
                "avg_sector3":    None,
                "avg_lap_time":   None,
            }

        # Restrict to same circuit type where possible
        same_type = prior_races[prior_races["circuit_type"] == circuit_type]
        source = same_type if not same_type.empty else prior_races

        lap_dfs = []
        for _, race in source.iterrows():
            laps = self.db.get_lap_data(int(race["year"]), int(race["round"]))
            driver_laps = laps[laps["driver_code"] == driver].copy()
            if not driver_laps.empty:
                lap_dfs.append(driver_laps)

        if not lap_dfs:
            return {
                "clean_air_pace": None,
                "avg_sector1":    None,
                "avg_sector2":    None,
                "avg_sector3":    None,
                "avg_lap_time":   None,
            }

        all_laps = pd.concat(lap_dfs, ignore_index=True)

        # Clean laps: no caution, not pit-in/out, tire_age > 3 (settled rubber)
        clean = all_laps[
            (~all_laps["track_status"].isin(_CAUTION_STATUSES))
            & (~all_laps["is_pit_in_lap"].fillna(False))
            & (~all_laps["is_pit_out_lap"].fillna(False))
            & (all_laps["tire_age"].fillna(0) > 3)
            & (all_laps["lap_time"].notna())
        ]

        return {
            "clean_air_pace": clean["lap_time"].median() if not clean.empty else None,
            "avg_sector1":    all_laps["sector1_time"].median(),
            "avg_sector2":    all_laps["sector2_time"].median(),
            "avg_sector3":    all_laps["sector3_time"].median(),
            "avg_lap_time":   all_laps["lap_time"].median(),
        }

    def _tire_deg_features(
        self, driver: str, prior_races: pd.DataFrame
    ) -> dict:
        """
        Estimate lap-time degradation rate (seconds/lap) per compound by
        fitting a linear regression of lap_time ~ tire_age for each stint.
        """
        result = {
            "tire_deg_soft":   None,
            "tire_deg_medium": None,
            "tire_deg_hard":   None,
        }
        if prior_races.empty:
            return result

        for _, race in prior_races.iterrows():
            laps = self.db.get_lap_data(int(race["year"]), int(race["round"]))
            driver_laps = laps[laps["driver_code"] == driver].copy()
            if driver_laps.empty:
                continue

            clean = driver_laps[
                (~driver_laps["track_status"].isin(_CAUTION_STATUSES))
                & (~driver_laps["is_pit_in_lap"].fillna(False))
                & (~driver_laps["is_pit_out_lap"].fillna(False))
                & driver_laps["lap_time"].notna()
                & driver_laps["tire_age"].notna()
                & driver_laps["compound"].notna()
            ]

            for compound, key in [
                ("SOFT",   "tire_deg_soft"),
                ("MEDIUM", "tire_deg_medium"),
                ("HARD",   "tire_deg_hard"),
            ]:
                stint_data = clean[clean["compound"] == compound]
                if len(stint_data) < 4:
                    continue
                slope, _, _, _, _ = stats.linregress(
                    stint_data["tire_age"], stint_data["lap_time"]
                )
                # Average across races (last value wins if multiple races)
                result[key] = slope

        return result

    def _pit_stop_features(
        self, team: Optional[str], prior_races: pd.DataFrame
    ) -> dict:
        if team is None or prior_races.empty:
            return {"avg_pit_duration": None}

        durations = []
        for _, race in prior_races.iterrows():
            results = self.db.get_session_results(int(race["year"]), int(race["round"]))
            team_drivers = results[results["team_name"] == team]["driver_code"].tolist()
            if not team_drivers:
                continue
            pits = self.db.get_pit_stops(int(race["year"]), int(race["round"]))
            team_pits = pits[pits["driver_code"].isin(team_drivers)]
            if not team_pits.empty and team_pits["pit_duration"].notna().any():
                durations.extend(team_pits["pit_duration"].dropna().tolist())

        return {"avg_pit_duration": float(np.median(durations)) if durations else None}

    def _standing_features(
        self,
        driver: str,
        team: Optional[str],
        year: int,
        round_num: int,
    ) -> dict:
        """Sum of points scored in races before this one in the same season."""
        results = self.db.get_session_results(year)

        # Only races that have already happened (lower round number)
        prior = results[results["round"] < round_num]

        driver_pts = 0.0
        if not prior.empty:
            d = prior[prior["driver_code"] == driver]["points"]
            driver_pts = float(d.sum()) if not d.empty else 0.0

        team_pts = 0.0
        if team and not prior.empty:
            t = prior[prior["team_name"] == team]["points"]
            team_pts = float(t.sum()) if not t.empty else 0.0

        return {
            "driver_points":      driver_pts,
            "constructor_points": team_pts,
        }

    def _driver_form_features(
        self, driver: str, prior_races: pd.DataFrame
    ) -> dict:
        """Average finishing position in the last 3 races."""
        if prior_races.empty:
            return {"driver_form_3": None, "dnf_rate": None}

        recent = prior_races.tail(3)
        positions, dnf_count, total = [], 0, 0

        for _, race in recent.iterrows():
            results = self.db.get_session_results(int(race["year"]), int(race["round"]))
            driver_result = results[results["driver_code"] == driver]
            if driver_result.empty:
                continue
            total += 1
            pos = driver_result.iloc[0]["finish_position"]
            status = str(driver_result.iloc[0].get("status", ""))
            if pd.isna(pos) or "DNF" in status or "Accident" in status:
                dnf_count += 1
            else:
                positions.append(float(pos))

        form = float(np.mean(positions)) if positions else None
        dnf_rate = dnf_count / total if total > 0 else None
        return {"driver_form_3": form, "dnf_rate": dnf_rate}

    def _reliability_features(
        self, driver: str, prior_races: pd.DataFrame
    ) -> dict:
        """Season-wide DNF rate from prior races."""
        if prior_races.empty:
            return {"season_dnf_rate": None}

        dnf_count, total = 0, 0
        for _, race in prior_races.iterrows():
            results = self.db.get_session_results(int(race["year"]), int(race["round"]))
            driver_result = results[results["driver_code"] == driver]
            if driver_result.empty:
                continue
            total += 1
            status = str(driver_result.iloc[0].get("status", ""))
            if any(kw in status for kw in ("DNF", "Accident", "Engine", "Gearbox", "Mechanical")):
                dnf_count += 1

        return {"season_dnf_rate": dnf_count / total if total > 0 else None}

    def _weather_features(
        self,
        year: int,
        round_num: int,
        override: Optional[dict],
    ) -> dict:
        """
        Weather at race time.

        Source priority (ensures training and prediction use the same data source):
          1. override dict — caller-supplied values (e.g. from Open-Meteo forecast
             fetched by WeatherCollector.fetch_race_forecast() in predict_race())
          2. OM_HIST — Open-Meteo historical archive data collected alongside
             the race session. Same API as the forecast, so training and
             prediction features come from consistent distributions.
          3. FastF1 "R" — in-circuit sensor data. Falls back to this for races
             collected before WeatherCollector was added to the pipeline.
        """
        defaults = {
            "air_temp":       None,
            "track_temp":     None,
            "humidity":       None,
            "pressure":       None,
            "rainfall":       None,
            "wind_speed":     None,
            "wind_direction": None,
        }

        if override:
            defaults.update(override)
            return defaults

        # Try Open-Meteo historical first (consistent with forecast source)
        weather = self.db.get_weather(year, round_num, session_type="OM_HIST")

        # Fall back to FastF1 circuit sensor data
        if weather.empty:
            weather = self.db.get_weather(year, round_num, session_type="R")

        if weather.empty:
            return defaults

        # Aggregate all available readings with median (robust to outliers)
        for col in defaults:
            if col in weather.columns:
                val = weather[col].dropna()
                defaults[col] = float(val.median()) if not val.empty else None

        return defaults

    def _circuit_sc_probability(
        self, prior_races: pd.DataFrame, circuit_type: Optional[str]
    ) -> dict:
        """
        Historical fraction of races at circuits of the same type that had
        at least one safety car or VSC period.
        """
        if prior_races.empty or circuit_type is None:
            return {"sc_probability": None}

        same_type = prior_races[prior_races["circuit_type"] == circuit_type]
        if same_type.empty:
            return {"sc_probability": None}

        sc_count, total = 0, 0
        for _, race in same_type.iterrows():
            laps = self.db.get_lap_data(int(race["year"]), int(race["round"]))
            total += 1
            if laps["track_status"].isin({"4", "6"}).any():
                sc_count += 1

        return {"sc_probability": sc_count / total if total > 0 else None}

    def _telemetry_features(
        self,
        driver: str,
        prior_races: pd.DataFrame,
        circuit_type: Optional[str],
    ) -> dict:
        defaults = {
            "tel_mean_speed": None,
            "tel_max_speed":  None,
            "tel_brake_pct":  None,
            "tel_drs_pct":    None,
        }

        if prior_races.empty:
            return defaults

        same_type = prior_races[prior_races["circuit_type"] == circuit_type]
        source = same_type if not same_type.empty else prior_races

        tel_dfs = []
        for _, race in source.iterrows():
            tel = self.db.get_telemetry_summary(int(race["year"]), int(race["round"]))
            driver_tel = tel[tel["driver_code"] == driver]
            if not driver_tel.empty:
                tel_dfs.append(driver_tel)

        if not tel_dfs:
            return defaults

        combined = pd.concat(tel_dfs, ignore_index=True)
        return {
            "tel_mean_speed": float(combined["mean_speed"].median()) if combined["mean_speed"].notna().any() else None,
            "tel_max_speed":  float(combined["max_speed"].median()) if combined["max_speed"].notna().any() else None,
            "tel_brake_pct":  float(combined["brake_pct"].median()) if combined["brake_pct"].notna().any() else None,
            "tel_drs_pct":    float(combined["drs_pct"].median()) if combined["drs_pct"].notna().any() else None,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _get_driver_team(
        self, driver: str, year: int, round_num: int
    ) -> Optional[str]:
        results = self.db.get_session_results(year, round_num)
        row = results[results["driver_code"] == driver]
        if row.empty:
            # Fall back to most recent known result
            all_results = self.db.get_session_results(year)
            recent = all_results[all_results["driver_code"] == driver]
            if not recent.empty:
                return str(recent.sort_values("round").iloc[-1]["team_name"])
            return None
        return str(row.iloc[0]["team_name"])

    @staticmethod
    def feature_columns() -> list[str]:
        """Returns the ordered list of feature column names used by the model."""
        return [
            "quali_time",
            "quali_position",
            "q3_time",
            "clean_air_pace",
            "avg_sector1",
            "avg_sector2",
            "avg_sector3",
            "avg_lap_time",
            "tire_deg_soft",
            "tire_deg_medium",
            "tire_deg_hard",
            "avg_pit_duration",
            "driver_points_norm",
            "constructor_points_norm",
            "driver_form_3",
            "dnf_rate",
            "season_dnf_rate",
            "air_temp",
            "track_temp",
            "humidity",
            "pressure",
            "rainfall",
            "wind_speed",
            "wind_direction",
            "sc_probability",
            "circuit_type_enc",
            "tel_mean_speed",
            "tel_max_speed",
            "tel_brake_pct",
            "tel_drs_pct",
        ]
