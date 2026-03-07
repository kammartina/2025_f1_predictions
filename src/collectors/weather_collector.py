"""
WeatherCollector: fetches weather data from Open-Meteo (free, no API key required).

Two modes
─────────
  Historical (archive-api.open-meteo.com):
      Called during collect_season / collect_round for completed races.
      Provides actual measured hourly weather for the race date.
      Stored in the weather table with session_type = "OM_HIST".

  Forecast (api.open-meteo.com):
      Called automatically inside predict_race() for upcoming races.
      Provides an hourly forecast for the race date.
      Returned as a dict — NOT stored in the database.

Why Open-Meteo alongside FastF1 weather?
─────────────────────────────────────────
  FastF1 provides highly accurate in-circuit sensor readings every ~30s during
  sessions (historical only). Open-Meteo provides consistent API data for both
  historical dates AND future forecasts.

  Using the same data source (Open-Meteo) for both training and prediction
  avoids a feature distribution mismatch — the model learns weather patterns
  from the same kind of numbers it will receive at prediction time.

  FastF1 sensor data remains available as a fallback in FeatureEngineer if
  Open-Meteo data has not been collected for older races.

Track temperature note
──────────────────────
  Open-Meteo does not measure asphalt temperature directly. We use
  `surface_temperature` (radiative skin temperature of the land surface) as
  the closest available proxy. Real F1 track temperatures can be 10–20°C
  higher than this on sunny days because asphalt absorbs more heat than
  average land surface. This proxy is still useful for distinguishing wet/cool
  races from hot dry races, which is what the model needs.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd
import requests

from src.collectors.base_collector import BaseCollector
from src.db.database import F1Database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Open-Meteo endpoint URLs
# ---------------------------------------------------------------------------
_ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Variables to request from the archive (historical) endpoint
_ARCHIVE_VARS = [
    "temperature_2m",       # air temperature at 2m height (°C)
    "surface_temperature",  # radiative skin temperature (°C) — track temp proxy
    "relativehumidity_2m",  # relative humidity at 2m (%)
    "surface_pressure",     # atmospheric pressure (hPa = mbar)
    "precipitation",        # accumulated precipitation per hour (mm) — > 0.1 mm → rain
    "windspeed_10m",        # wind speed at 10m (m/s when wind_speed_unit=ms)
    "winddirection_10m",    # wind direction at 10m (degrees, 0–360)
]

# Variables to request from the forecast endpoint
# (precipitation_probability replaces precipitation for future dates)
_FORECAST_VARS = [
    "temperature_2m",
    "surface_temperature",
    "relativehumidity_2m",
    "surface_pressure",
    "precipitation_probability",  # % chance of rain per hour — > 40% → expect rain
    "windspeed_10m",
    "winddirection_10m",
]

# Local-time hour window that covers any F1 race start time worldwide
# (earliest: 08:00 local for some Asian markets; latest: 17:00 for night races)
_RACE_HOUR_MIN = 8
_RACE_HOUR_MAX = 17


# ---------------------------------------------------------------------------
# WeatherCollector
# ---------------------------------------------------------------------------

class WeatherCollector(BaseCollector):
    """
    Fetches Open-Meteo weather and persists historical data to the DB.

    Usage — historical (completed races)
    ─────────────────────────────────────
    with F1Database() as db:
        wc = WeatherCollector(db)
        wc.collect(2025, 1)          # stores OM_HIST rows for 2025 Round 1

    Usage — forecast (upcoming race, returns dict only)
    ────────────────────────────────────────────────────
    with F1Database() as db:
        wc = WeatherCollector(db)
        forecast = wc.fetch_race_forecast(2026, 1)
    pipeline.predict_race(2026, 1, weather_forecast=forecast)
    """

    def __init__(self, db: F1Database) -> None:
        super().__init__(db)

    # ------------------------------------------------------------------
    # Public: historical collection
    # ------------------------------------------------------------------

    def collect(
        self,
        year: int,
        round_num: int,
        force: bool = False,
        **_kwargs,
    ) -> None:
        """
        Fetch and store Open-Meteo historical weather for a completed race.

        The race entry must already exist in the `races` table (run
        SessionCollector first) so that coordinates and date are available.
        """
        meta = self._get_race_meta(year, round_num)
        if meta is None:
            logger.warning(
                "No race metadata for %d R%d — run SessionCollector first.", year, round_num
            )
            return

        if meta["lat"] is None or meta["lon"] is None:
            logger.warning(
                "No coordinates for %d R%d (%s). Add this circuit to CIRCUIT_INFO in schema.py.",
                year, round_num, meta["circuit_name"],
            )
            return

        if meta["race_date"] is None:
            logger.warning("No race_date stored for %d R%d.", year, round_num)
            return

        # Skip if already collected (unless force=True)
        if not force:
            existing = self.db.query(
                "SELECT COUNT(*) AS n FROM weather "
                "WHERE year = ? AND round = ? AND session_type = 'OM_HIST'",
                [year, round_num],
            )
            if int(existing["n"].iloc[0]) > 0:
                logger.info(
                    "Open-Meteo historical weather already stored for %d R%d — skipping.",
                    year, round_num,
                )
                return

        race_date = _to_date(meta["race_date"])
        if race_date >= date.today():
            logger.info(
                "Skipping Open-Meteo for %d R%d — race date %s is in the future.",
                year, round_num, race_date,
            )
            return
        df = self._fetch_archive(meta["lat"], meta["lon"], race_date)

        if df is None or df.empty:
            logger.warning("Open-Meteo returned no usable data for %d R%d.", year, round_num)
            return

        df["year"]         = year
        df["round"]        = round_num
        df["session_type"] = "OM_HIST"
        self.db.insert_df(df, "weather")

        logger.info(
            "Stored Open-Meteo historical weather for %d R%d (%d hourly rows).",
            year, round_num, len(df),
        )

    # ------------------------------------------------------------------
    # Public: forecast (does not write to DB)
    # ------------------------------------------------------------------

    def fetch_race_forecast(self, year: int, round_num: int) -> Optional[dict]:
        """
        Fetch an Open-Meteo weather forecast for an upcoming race.

        Returns a dict compatible with pipeline.predict_race(weather_forecast=...):
            {air_temp, track_temp, humidity, pressure, rainfall,
             wind_speed, wind_direction}

        Returns None if the forecast cannot be retrieved (e.g. race is more
        than 16 days away, which is Open-Meteo's maximum forecast horizon).
        """
        meta = self._get_race_meta(year, round_num)
        if meta is None or meta["lat"] is None or meta["race_date"] is None:
            logger.warning(
                "Cannot fetch forecast for %d R%d: missing race metadata.", year, round_num
            )
            return None

        race_date = _to_date(meta["race_date"])
        df = self._fetch_forecast(meta["lat"], meta["lon"], race_date)

        if df is None or df.empty:
            logger.warning(
                "Open-Meteo forecast unavailable for %d R%d "
                "(race may be more than 16 days away or API unavailable).",
                year, round_num,
            )
            return None

        # Aggregate race-window hours into a single representative value
        return {
            "air_temp":       _median(df, "air_temp"),
            "track_temp":     _median(df, "track_temp"),
            "humidity":       _median(df, "humidity"),
            "pressure":       _median(df, "pressure"),
            "rainfall":       int(df["rainfall"].max()),   # 1 if ANY hour expects rain
            "wind_speed":     _median(df, "wind_speed"),
            "wind_direction": _median(df, "wind_direction"),
        }

    # ------------------------------------------------------------------
    # Private: API calls
    # ------------------------------------------------------------------

    def _fetch_archive(
        self, lat: float, lon: float, target_date: date
    ) -> Optional[pd.DataFrame]:
        """Call archive-api.open-meteo.com for one date."""
        date_str = target_date.isoformat()
        params = {
            "latitude":        lat,
            "longitude":       lon,
            "start_date":      date_str,
            "end_date":        date_str,
            "hourly":          ",".join(_ARCHIVE_VARS),
            "timezone":        "auto",   # returns local-time timestamps
            "wind_speed_unit": "ms",     # m/s — same unit as FastF1 weather
        }
        try:
            resp = requests.get(_ARCHIVE_URL, params=params, timeout=15)
            resp.raise_for_status()
            return self._parse_response(resp.json(), forecast=False, target_date=target_date)
        except requests.RequestException as exc:
            logger.error("Open-Meteo archive request failed: %s", exc)
            return None

    def _fetch_forecast(
        self, lat: float, lon: float, target_date: date
    ) -> Optional[pd.DataFrame]:
        """Call api.open-meteo.com for a future date."""
        days_ahead = (target_date - date.today()).days
        # Need enough days in the forecast to reach race_date; API max is 16
        forecast_days = max(1, min(days_ahead + 2, 16))

        params = {
            "latitude":        lat,
            "longitude":       lon,
            "hourly":          ",".join(_FORECAST_VARS),
            "timezone":        "auto",
            "wind_speed_unit": "ms",
            "forecast_days":   forecast_days,
        }
        try:
            resp = requests.get(_FORECAST_URL, params=params, timeout=15)
            resp.raise_for_status()
            return self._parse_response(resp.json(), forecast=True, target_date=target_date)
        except requests.RequestException as exc:
            logger.error("Open-Meteo forecast request failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Private: JSON → DataFrame
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        data: dict,
        forecast: bool,
        target_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Parse an Open-Meteo hourly JSON response.

        Filters to:
          - rows matching target_date (local time)
          - hours between _RACE_HOUR_MIN and _RACE_HOUR_MAX

        Maps API field names to the weather table column names and handles
        the different precipitation representations (mm vs probability %).
        """
        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])
        if not times:
            return None

        n = len(times)

        def _col(key: str) -> list:
            return hourly.get(key, [None] * n)

        air_temps  = _col("temperature_2m")
        surf_temps = _col("surface_temperature")
        humidity   = _col("relativehumidity_2m")
        pressure   = _col("surface_pressure")
        wind_spd   = _col("windspeed_10m")
        wind_dir   = _col("winddirection_10m")
        precip_key = "precipitation_probability" if forecast else "precipitation"
        precip     = _col(precip_key)

        records = []
        for i, t in enumerate(times):
            dt = pd.Timestamp(t)

            # Keep only the target date and the race-window hours
            if dt.date() != target_date:
                continue
            if not (_RACE_HOUR_MIN <= dt.hour <= _RACE_HOUR_MAX):
                continue

            # time_offset = seconds from midnight (matching pattern of other weather rows)
            time_offset = float(dt.hour * 3600)

            # Rainfall: archive gives mm/h; forecast gives probability %
            p = precip[i]
            if p is None:
                rainfall = 0
            elif forecast:
                rainfall = 1 if p > 40 else 0   # > 40% probability = expect rain
            else:
                rainfall = 1 if p > 0.1 else 0  # > 0.1 mm/h = measurable precipitation

            records.append({
                "time_offset":    time_offset,
                "air_temp":       air_temps[i],
                "track_temp":     surf_temps[i],   # surface temp as track-temp proxy
                "humidity":       humidity[i],
                "pressure":       pressure[i],
                "rainfall":       rainfall,
                "wind_speed":     wind_spd[i],
                "wind_direction": wind_dir[i],
            })

        return pd.DataFrame(records) if records else None

    # ------------------------------------------------------------------
    # Private: database helpers
    # ------------------------------------------------------------------

    def _get_race_meta(self, year: int, round_num: int) -> Optional[dict]:
        """Fetch circuit name, coordinates, and race date from the races table."""
        df = self.db.query(
            "SELECT circuit_name, lat, lon, race_date "
            "FROM races WHERE year = ? AND round = ?",
            [year, round_num],
        )
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "circuit_name": row.get("circuit_name"),
            "lat":          row.get("lat"),
            "lon":          row.get("lon"),
            "race_date":    row.get("race_date"),
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_date(value) -> date:
    """Coerce a string, pandas Timestamp, or date object to datetime.date."""
    if isinstance(value, date) and not isinstance(value, type(pd.Timestamp.now())):
        return value
    return pd.Timestamp(value).date()


def _median(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the median of a column, or None if all values are null."""
    s = df[col].dropna()
    return float(s.median()) if not s.empty else None
