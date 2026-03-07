"""
SessionCollector: fetches one race weekend's data from FastF1 and stores
it in the F1Database.

Sessions collected per round
─────────────────────────────
  Race (R)       → session_results, lap_data, pit_stops, weather[R],
                   telemetry_summary (optional)
  Qualifying (Q) → qualifying_results, weather[Q]
"""

from __future__ import annotations

import logging
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.collectors.base_collector import BaseCollector
from src.db.database import F1Database
from src.db.schema import CIRCUIT_INFO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _td_to_sec(td) -> Optional[float]:
    """Convert a pandas Timedelta (or NaT) to float seconds."""
    if pd.isna(td):
        return None
    return float(td.total_seconds())


def _track_status_label(raw) -> Optional[str]:
    """
    FastF1 track status is a string of digit flags set simultaneously.
    We keep only the highest-priority flag for simplicity:
      '1' green  '2' yellow  '4' SC  '5' red  '6' VSC  '7' VSC ending
    """
    if pd.isna(raw):
        return None
    raw = str(raw).strip()
    for flag in ("5", "4", "6", "2", "7", "1"):
        if flag in raw:
            return flag
    return raw


# ---------------------------------------------------------------------------
# SessionCollector
# ---------------------------------------------------------------------------

class SessionCollector(BaseCollector):
    """
    Collects race-weekend data via FastF1 and persists it to DuckDB.

    Usage
    -----
    with F1Database() as db:
        db.create_tables()
        collector = SessionCollector(db, cache_path="f1_cache")
        collector.collect(2025, 1, include_telemetry=False)
    """

    def __init__(self, db: F1Database, cache_path: str = "f1_cache") -> None:
        super().__init__(db)
        fastf1.Cache.enable_cache(cache_path)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def collect(
        self,
        year: int,
        round_num: int,
        include_telemetry: bool = False,
        force: bool = False,
    ) -> None:
        """
        Collect all data for one race weekend.

        Parameters
        ----------
        year              : Season year
        round_num         : Round number (1-based)
        include_telemetry : Whether to collect and store aggregated telemetry
                            (slow on first run; all data is then cached)
        force             : Re-collect even if data already exists in DB
        """
        if self.db.session_results_exist(year, round_num) and not force:
            logger.info("Round %d/%d already in database — skipping.", year, round_num)
            return

        steps = 4 + (1 if include_telemetry else 0)
        with tqdm(
            total=steps,
            desc=f"    R{round_num:02d}",
            unit="step",
            leave=False,
            ncols=72,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}",
        ) as pbar:
            # ── Step 1: load race session from FastF1 ──────────────────────
            pbar.set_postfix_str("loading race session…")
            try:
                race_session = fastf1.get_session(year, round_num, "R")
                race_session.load(telemetry=include_telemetry, weather=True, laps=True)
                if len(race_session.drivers) == 0:
                    logger.warning("No data available for %d R%d (session not yet published)", year, round_num)
                    return
            except Exception as exc:
                logger.error("Failed to load race session %d R%d: %s", year, round_num, exc)
                return
            pbar.update(1)

            # ── Step 2: store race data ────────────────────────────────────
            pbar.set_postfix_str("storing race data…")
            self._store_race_metadata(race_session, year, round_num)
            self._store_drivers_and_teams(race_session, year)
            self._store_session_results(race_session, year, round_num)
            self._store_lap_data(race_session, year, round_num)
            self._store_pit_stops(race_session, year, round_num)
            self._store_weather(race_session, year, round_num, session_type="R")
            pbar.update(1)

            # ── Step 3 (optional): telemetry ──────────────────────────────
            if include_telemetry:
                pbar.set_postfix_str("storing telemetry…")
                self._store_telemetry_summary(race_session, year, round_num)
                pbar.update(1)

            # ── Step 4: load qualifying session ───────────────────────────
            pbar.set_postfix_str("loading qualifying session…")
            try:
                quali_session = fastf1.get_session(year, round_num, "Q")
                quali_session.load(telemetry=False, weather=True, laps=False)
            except Exception as exc:
                logger.warning("Could not load qualifying for %d R%d: %s", year, round_num, exc)
                pbar.update(2)  # skip both remaining qualifying steps
                return
            pbar.update(1)

            # ── Step 5: store qualifying data ─────────────────────────────
            pbar.set_postfix_str("storing qualifying data…")
            self._store_qualifying_results(quali_session, year, round_num)
            self._store_weather(quali_session, year, round_num, session_type="Q")
            pbar.update(1)
            pbar.set_postfix_str("done")

        logger.info("Round %d/%d collected successfully.", year, round_num)

    # ------------------------------------------------------------------
    # Private storage methods
    # ------------------------------------------------------------------

    def _store_race_metadata(
        self, session: fastf1.core.Session, year: int, round_num: int
    ) -> None:
        event = session.event
        circuit_name = str(event.get("EventName", ""))
        info = CIRCUIT_INFO.get(circuit_name, {})

        race_date = None
        try:
            race_date = pd.Timestamp(event.get("EventDate")).date().isoformat()
        except Exception:
            pass

        row = pd.DataFrame([{
            "year":         year,
            "round":        round_num,
            "circuit_name": circuit_name,
            "country":      str(event.get("Country", "")),
            "city":         str(event.get("Location", "")),
            "race_date":    race_date,
            "circuit_type": info.get("type"),
            "lat":          info.get("lat"),
            "lon":          info.get("lon"),
        }])
        self.db.insert_df(row, "races")

    def _store_drivers_and_teams(
        self, session: fastf1.core.Session, year: int
    ) -> None:
        results = session.results
        if results is None or results.empty:
            return

        drivers, teams = [], []
        for _, row in results.iterrows():
            drivers.append({
                "driver_code":    str(row.get("Abbreviation", "")),
                "full_name":      str(row.get("FullName", "")),
                "broadcast_name": str(row.get("BroadcastName", "")),
                "headshot_url":   str(row.get("HeadshotUrl", "")),
                "nationality":    str(row.get("CountryCode", "")),
            })
            teams.append({
                "team_name": str(row.get("TeamName", "")),
                "year":      year,
                "color":     str(row.get("TeamColor", "")),
            })

        self.db.insert_df(pd.DataFrame(drivers).drop_duplicates("driver_code"), "drivers")
        self.db.insert_df(pd.DataFrame(teams).drop_duplicates(["team_name", "year"]), "teams")

    def _store_session_results(
        self, session: fastf1.core.Session, year: int, round_num: int
    ) -> None:
        results = session.results
        if results is None or results.empty:
            return

        rows = []
        for _, row in results.iterrows():
            pos = row.get("Position")
            rows.append({
                "year":            year,
                "round":           round_num,
                "driver_code":     str(row.get("Abbreviation", "")),
                "team_name":       str(row.get("TeamName", "")),
                "finish_position": int(pos) if not pd.isna(pos) else None,
                "grid_position":   int(row["GridPosition"]) if not pd.isna(row.get("GridPosition")) else None,
                "points":          float(row["Points"]) if not pd.isna(row.get("Points")) else 0.0,
                "status":          str(row.get("Status", "")),
                "total_laps":      int(row["NumberOfLaps"]) if not pd.isna(row.get("NumberOfLaps")) else None,
            })

        self.db.insert_df(pd.DataFrame(rows), "session_results")

    def _store_qualifying_results(
        self, session: fastf1.core.Session, year: int, round_num: int
    ) -> None:
        results = session.results
        if results is None or results.empty:
            return

        rows = []
        for _, row in results.iterrows():
            pos = row.get("Position")
            rows.append({
                "year":                 year,
                "round":                round_num,
                "driver_code":          str(row.get("Abbreviation", "")),
                "q1_time":              _td_to_sec(row.get("Q1")),
                "q2_time":              _td_to_sec(row.get("Q2")),
                "q3_time":              _td_to_sec(row.get("Q3")),
                "qualifying_position":  int(pos) if not pd.isna(pos) else None,
            })

        self.db.insert_df(pd.DataFrame(rows), "qualifying_results")

    def _store_lap_data(
        self, session: fastf1.core.Session, year: int, round_num: int
    ) -> None:
        laps = session.laps
        if laps is None or laps.empty:
            return

        rows = []
        for _, lap in laps.iterrows():
            rows.append({
                "year":          year,
                "round":         round_num,
                "driver_code":   str(lap.get("Driver", "")),
                "lap_number":    int(lap["LapNumber"]) if not pd.isna(lap.get("LapNumber")) else None,
                "lap_time":      _td_to_sec(lap.get("LapTime")),
                "sector1_time":  _td_to_sec(lap.get("Sector1Time")),
                "sector2_time":  _td_to_sec(lap.get("Sector2Time")),
                "sector3_time":  _td_to_sec(lap.get("Sector3Time")),
                "compound":      str(lap.get("Compound", "")) if not pd.isna(lap.get("Compound")) else None,
                "tire_age":      int(lap["TyreLife"]) if not pd.isna(lap.get("TyreLife")) else None,
                "stint_number":  int(lap["Stint"]) if not pd.isna(lap.get("Stint")) else None,
                "track_status":  _track_status_label(lap.get("TrackStatus")),
                "is_pit_in_lap": bool(not pd.isna(lap.get("PitInTime"))),
                "is_pit_out_lap": bool(not pd.isna(lap.get("PitOutTime"))),
                "position":      int(lap["Position"]) if not pd.isna(lap.get("Position")) else None,
            })

        df = pd.DataFrame(rows).dropna(subset=["lap_number"])
        df["lap_number"] = df["lap_number"].astype(int)
        self.db.insert_df(df, "lap_data")

    def _store_pit_stops(
        self, session: fastf1.core.Session, year: int, round_num: int
    ) -> None:
        laps = session.laps
        if laps is None or laps.empty:
            return

        rows = []
        for driver_code in laps["Driver"].unique():
            driver_laps = laps.pick_driver(driver_code).reset_index(drop=True)

            pit_in_laps = driver_laps[driver_laps["PitInTime"].notna()]
            for _, pit_in_lap in pit_in_laps.iterrows():
                lap_num = int(pit_in_lap["LapNumber"])

                # Pit-out is on the following lap
                next_laps = driver_laps[driver_laps["LapNumber"] == lap_num + 1]
                pit_out_time = None
                if not next_laps.empty and not pd.isna(next_laps.iloc[0]["PitOutTime"]):
                    pit_out_time = next_laps.iloc[0]["PitOutTime"]

                duration = None
                if pit_out_time is not None:
                    duration = _td_to_sec(pit_out_time - pit_in_lap["PitInTime"])

                compound_before = (
                    str(pit_in_lap["Compound"])
                    if not pd.isna(pit_in_lap.get("Compound")) else None
                )
                compound_after = None
                if not next_laps.empty and not pd.isna(next_laps.iloc[0].get("Compound")):
                    compound_after = str(next_laps.iloc[0]["Compound"])

                rows.append({
                    "year":            year,
                    "round":           round_num,
                    "driver_code":     driver_code,
                    "lap_number":      lap_num,
                    "pit_duration":    duration,
                    "compound_before": compound_before,
                    "compound_after":  compound_after,
                })

        if rows:
            self.db.insert_df(pd.DataFrame(rows), "pit_stops")

    def _store_weather(
        self,
        session: fastf1.core.Session,
        year: int,
        round_num: int,
        session_type: str,
    ) -> None:
        weather = session.weather_data
        if weather is None or weather.empty:
            return

        rows = []
        for _, w in weather.iterrows():
            time_offset = _td_to_sec(w.get("Time"))
            if time_offset is None:
                continue
            rows.append({
                "year":          year,
                "round":         round_num,
                "session_type":  session_type,
                "time_offset":   time_offset,
                "air_temp":      float(w["AirTemp"]) if not pd.isna(w.get("AirTemp")) else None,
                "track_temp":    float(w["TrackTemp"]) if not pd.isna(w.get("TrackTemp")) else None,
                "humidity":      float(w["Humidity"]) if not pd.isna(w.get("Humidity")) else None,
                "pressure":      float(w["Pressure"]) if not pd.isna(w.get("Pressure")) else None,
                "rainfall":      int(bool(w["Rainfall"])) if not pd.isna(w.get("Rainfall")) else 0,
                "wind_speed":    float(w["WindSpeed"]) if not pd.isna(w.get("WindSpeed")) else None,
                "wind_direction": float(w["WindDirection"]) if not pd.isna(w.get("WindDirection")) else None,
            })

        if rows:
            self.db.insert_df(pd.DataFrame(rows), "weather")

    def _store_telemetry_summary(
        self, session: fastf1.core.Session, year: int, round_num: int
    ) -> None:
        """
        Aggregate 10Hz telemetry to per-lap statistics per driver.
        This is optional (slow on first run) but cached afterwards.
        """
        laps = session.laps
        if laps is None or laps.empty:
            return

        rows = []
        for driver_code in laps["Driver"].unique():
            driver_laps = laps.pick_driver(driver_code)
            for _, lap in driver_laps.iterrows():
                lap_num = lap.get("LapNumber")
                if pd.isna(lap_num):
                    continue
                try:
                    tel = lap.get_telemetry()
                    if tel is None or tel.empty:
                        continue
                    rows.append({
                        "year":          year,
                        "round":         round_num,
                        "driver_code":   driver_code,
                        "lap_number":    int(lap_num),
                        "mean_speed":    float(tel["Speed"].mean()),
                        "max_speed":     float(tel["Speed"].max()),
                        "mean_throttle": float(tel["Throttle"].mean()),
                        "brake_pct":     float(tel["Brake"].astype(float).mean()),
                        "drs_pct":       float((tel["DRS"] >= 10).mean()),
                        "mean_rpm":      float(tel["RPM"].mean()),
                    })
                except Exception as exc:
                    logger.debug(
                        "Telemetry unavailable for %s lap %d: %s",
                        driver_code, int(lap_num), exc,
                    )

        if rows:
            self.db.insert_df(pd.DataFrame(rows), "telemetry_summary")
