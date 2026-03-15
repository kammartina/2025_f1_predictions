# F1 Race Predictions — 2025/2026 Season

A machine-learning pipeline that predicts F1 race finishing order using **XGBoost**, **FastF1**, and **Open-Meteo** weather data. Trained on lap-level historical data and evaluated with Leave-One-Race-Out cross-validation.

---

## How It Works

```
collect → train → predict
```

1. **Collect** — Pulls race and qualifying sessions from the FastF1 API and historical weather from Open-Meteo. Stores everything in a local DuckDB database (`data/f1_data.db`).
2. **Train** — Builds a 30-feature matrix (one row per driver per race) and trains an XGBoost regressor. Runs Leave-One-Race-Out CV to report Spearman correlation and top-3 overlap accuracy.
3. **Predict** — After qualifying is collected for an upcoming race, generates a ranked finishing-order prediction. Automatically fetches an Open-Meteo race-day forecast (falls back to historical median if the race is too far out).

---

## Model

| | |
|---|---|
| **Algorithm** | XGBoost (regression → ranked 1–20) |
| **Validation** | Leave-One-Race-Out CV |
| **Weighting** | Recency — recent races weighted higher (half-life = 12 races) |
| **Missing values** | Passed directly to XGBoost (no imputation needed) |

### Feature Groups (30 features)

| Group | Features |
|---|---|
| Qualifying | `quali_time`, `quali_position`, `q3_time` |
| Race pace | `clean_air_pace`, `avg_lap_time`, `avg_sector1/2/3` |
| Tire degradation | `tire_deg_soft/medium/hard` (slope of lap time vs tire age) |
| Pit stops | `avg_pit_duration` (team historical median) |
| Championship standings | `driver_points_norm`, `constructor_points_norm` |
| Driver form | `driver_form_3` (avg finish pos last 3 races), `dnf_rate`, `season_dnf_rate` |
| Weather | `air_temp`, `track_temp`, `humidity`, `pressure`, `rainfall`, `wind_speed`, `wind_direction` |
| Circuit | `circuit_type_enc` (street / technical / high_speed / mixed), `sc_probability` |
| Telemetry (optional) | `tel_mean_speed`, `tel_max_speed`, `tel_brake_pct`, `tel_drs_pct` |

### Latest Training Run (2026-03-11)

| Metric | Value |
|---|---|
| Trained through | 2026 R01 (25 races) |
| Mean Spearman r | 0.593 |
| Mean top-3 overlap | 2.04 / 3 |
| Top feature | `quali_position` (importance 0.238) |

---

## Project Structure

```
├── main.py                        # CLI entry point
├── requirements.txt
├── data/
│   └── f1_data.db                 # DuckDB database (auto-created)
├── models/
│   ├── predictor.json             # Saved XGBoost model
│   └── training_log.csv           # Per-run accuracy metrics + top features
├── src/
│   ├── pipeline.py                # F1Pipeline orchestrator
│   ├── collectors/
│   │   ├── session_collector.py   # FastF1 race + qualifying data
│   │   └── weather_collector.py   # Open-Meteo historical + forecast
│   ├── features/
│   │   └── feature_engineering.py # Builds the training/prediction matrix
│   ├── models/
│   │   └── predictor.py           # XGBoost wrapper + LORO-CV
│   └── db/
│       ├── database.py            # DuckDB query helpers
│       └── schema.py              # Table definitions + circuit metadata
├── original_prediction_files/     # Earlier per-race script versions (archived)
└── f1_cache/                      # FastF1 local cache
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** `fastf1`, `xgboost`, `duckdb`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `requests`, `matplotlib`, `tqdm`

---

## Usage

### Collect data

The `--session` flag controls what gets collected:

| Flag | When to use |
|---|---|
| `--session all` (default) | Full season backfill — collects qualifying + race for each round |
| `--session quali` | After qualifying, before the race |
| `--session race` | After the race finishes |

```bash
# Collect an entire past season (qualifying + race for every round)
python main.py collect --year 2025

# After qualifying for an upcoming race
python main.py collect --year 2026 --round 3 --session quali

# After the race finishes
python main.py collect --year 2026 --round 3 --session race --force

# Also collect telemetry (slow first run, then cached)
python main.py collect --year 2025 --telemetry
```

### Train the model

```bash
# Train on all collected data
python main.py train

# Train on specific seasons only
python main.py train --years 2025 2026
```

Output includes Leave-One-Race-Out CV results and a backtested grid for the most recent race.

### Predict a race

```bash
# Auto-fetches Open-Meteo race-day forecast
python main.py predict --year 2026 --round 2

# Override weather manually
python main.py predict --year 2026 --round 2 --air-temp 28 --track-temp 42 --rainfall 0

# Use historical weather median instead of fetching a forecast
python main.py predict --year 2026 --round 2 --no-auto-weather
```

Qualifying for the target round must be collected first.

### Other commands

```bash
# Run LORO-CV without retraining the production model
python main.py evaluate

# Show feature importances from the trained model
python main.py features

# Print a summary of what is in the database
python main.py summary
```

### Global options

```
--db     PATH    DuckDB database path  (default: data/f1_data.db)
--cache  PATH    FastF1 cache dir      (default: f1_cache)
--model  PATH    XGBoost model path    (default: models/predictor.json)
```

---

## Typical Workflow Each Race Weekend

### After qualifying (before the race)

```bash
# 1. Collect qualifying session + circuit metadata
python main.py collect --year 2026 --round 3 --session quali

# 2. Predict the race (auto-fetches Open-Meteo weather forecast)
python main.py predict --year 2026 --round 3
```

`train` is not needed here — the model already learned the qualifying→race relationship
from all historical rounds. The new qualifying data feeds directly into the prediction features.

### After the race

```bash
# 1. Collect race results + weather
python main.py collect --year 2026 --round 3 --session race --force

# 2. Retrain on all data including the new race result
python main.py train

# 3. Predict next round (after its qualifying is collected)
python main.py collect --year 2026 --round 4 --session quali
python main.py predict --year 2026 --round 4
```

---

## License

MIT