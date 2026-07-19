"""
SQL schema definitions and static circuit metadata for the F1 predictions database.

All tables use natural composite primary keys (no surrogate IDs) so that
INSERT ... ON CONFLICT DO NOTHING is simple and predictable.
"""

# ---------------------------------------------------------------------------
# Circuit metadata: name as returned by FastF1 -> type + coordinates
# ---------------------------------------------------------------------------
CIRCUIT_INFO: dict[str, dict] = {
    "Australian Grand Prix":      {"type": "street_adjacent", "lat": -37.8497, "lon": 144.9680},
    "Chinese Grand Prix":         {"type": "technical",       "lat":  31.3389, "lon": 121.2198},
    "Japanese Grand Prix":        {"type": "technical",       "lat":  34.8431, "lon": 136.5407},
    "Bahrain Grand Prix":         {"type": "technical",       "lat":  26.0325, "lon":  50.5106},
    "Saudi Arabian Grand Prix":   {"type": "street",          "lat":  21.6319, "lon":  39.1044},
    "Miami Grand Prix":           {"type": "street",          "lat":  25.9581, "lon": -80.2389},
    "Emilia Romagna Grand Prix":  {"type": "technical",       "lat":  44.3439, "lon":  11.7167},
    "Monaco Grand Prix":          {"type": "street",          "lat":  43.7347, "lon":   7.4205},
    "Spanish Grand Prix":         {"type": "mixed",           "lat":  41.5700, "lon":   2.2611},
    "Canadian Grand Prix":        {"type": "street_adjacent", "lat":  45.5017, "lon": -73.5229},
    "Austrian Grand Prix":        {"type": "high_speed",      "lat":  47.2197, "lon":  14.7647},
    "British Grand Prix":         {"type": "high_speed",      "lat":  52.0786, "lon":  -1.0169},
    "Belgian Grand Prix":         {"type": "high_speed",      "lat":  50.4372, "lon":   5.9714},
    "Hungarian Grand Prix":       {"type": "technical",       "lat":  47.5789, "lon":  19.2486},
    "Dutch Grand Prix":           {"type": "technical",       "lat":  52.3888, "lon":   4.5409},
    "Italian Grand Prix":         {"type": "high_speed",      "lat":  45.6156, "lon":   9.2811},
    "Singapore Grand Prix":       {"type": "street",          "lat":   1.2914, "lon": 103.8640},
    "Azerbaijan Grand Prix":      {"type": "street",          "lat":  40.3725, "lon":  49.8533},
    "United States Grand Prix":   {"type": "technical",       "lat":  30.1328, "lon": -97.6411},
    "Mexico City Grand Prix":     {"type": "technical",       "lat":  19.4042, "lon": -99.0907},
    "São Paulo Grand Prix":       {"type": "mixed",           "lat": -23.7036, "lon": -46.6997},
    "Las Vegas Grand Prix":       {"type": "street",          "lat":  36.1147, "lon":-115.1728},
    "Qatar Grand Prix":           {"type": "technical",       "lat":  25.4890, "lon":  51.4536},
    "Abu Dhabi Grand Prix":       {"type": "mixed",           "lat":  24.4672, "lon":  54.6031},
}

# Ordinal encoding for circuit_type used as a model feature
CIRCUIT_TYPE_ENCODING: dict[str, int] = {
    "street":          0,
    "street_adjacent": 1,
    "technical":       2,
    "mixed":           3,
    "high_speed":      4,
}

# ---------------------------------------------------------------------------
# CREATE TABLE statements
# ---------------------------------------------------------------------------

CREATE_RACES = """
CREATE TABLE IF NOT EXISTS races (
    year          INTEGER NOT NULL,
    round         INTEGER NOT NULL,
    circuit_name  TEXT,
    country       TEXT,
    city          TEXT,
    race_date     DATE,
    circuit_type  TEXT,   -- 'street', 'street_adjacent', 'technical', 'mixed', 'high_speed'
    lat           REAL,
    lon           REAL,
    PRIMARY KEY (year, round)
)
"""

CREATE_DRIVERS = """
CREATE TABLE IF NOT EXISTS drivers (
    driver_code     TEXT PRIMARY KEY,   -- e.g. 'VER', 'HAM'
    full_name       TEXT,
    broadcast_name  TEXT,
    headshot_url    TEXT,
    nationality     TEXT
)
"""

CREATE_TEAMS = """
CREATE TABLE IF NOT EXISTS teams (
    team_name  TEXT    NOT NULL,
    year       INTEGER NOT NULL,
    color      TEXT,
    PRIMARY KEY (team_name, year)
)
"""

CREATE_SESSION_RESULTS = """
CREATE TABLE IF NOT EXISTS session_results (
    year             INTEGER NOT NULL,
    round            INTEGER NOT NULL,
    driver_code      TEXT    NOT NULL,
    team_name        TEXT,
    finish_position  INTEGER,   -- classified finishing position (NULL for DNF classified)
    grid_position    INTEGER,
    points           REAL,
    status           TEXT,      -- 'Finished', '+1 Lap', 'DNF', etc.
    total_laps       INTEGER,
    PRIMARY KEY (year, round, driver_code)
)
"""

CREATE_QUALIFYING_RESULTS = """
CREATE TABLE IF NOT EXISTS qualifying_results (
    year                  INTEGER NOT NULL,
    round                 INTEGER NOT NULL,
    driver_code           TEXT    NOT NULL,
    q1_time               REAL,   -- seconds; NULL if not set
    q2_time               REAL,
    q3_time               REAL,
    qualifying_position   INTEGER,
    grid_position         INTEGER,  -- actual starting slot; differs from
                                     -- qualifying_position when a grid penalty
                                     -- applies. Defaults to qualifying_position
                                     -- until overridden (set-grid) or synced
                                     -- from the official race-day grid.
    PRIMARY KEY (year, round, driver_code)
)
"""

CREATE_LAP_DATA = """
CREATE TABLE IF NOT EXISTS lap_data (
    year           INTEGER NOT NULL,
    round          INTEGER NOT NULL,
    driver_code    TEXT    NOT NULL,
    lap_number     INTEGER NOT NULL,
    lap_time       REAL,           -- seconds; NULL for deleted / inaccurate laps
    sector1_time   REAL,
    sector2_time   REAL,
    sector3_time   REAL,
    compound       TEXT,           -- SOFT, MEDIUM, HARD, INTERMEDIATE, WET
    tire_age       INTEGER,        -- laps completed on this tyre set
    stint_number   INTEGER,
    track_status   TEXT,           -- '1'=green '2'=yellow '4'=SC '5'=red '6'=VSC
    is_pit_in_lap  BOOLEAN DEFAULT FALSE,
    is_pit_out_lap BOOLEAN DEFAULT FALSE,
    position       INTEGER,        -- race position at end of lap
    PRIMARY KEY (year, round, driver_code, lap_number)
)
"""

CREATE_PIT_STOPS = """
CREATE TABLE IF NOT EXISTS pit_stops (
    year             INTEGER NOT NULL,
    round            INTEGER NOT NULL,
    driver_code      TEXT    NOT NULL,
    lap_number       INTEGER NOT NULL,  -- lap on which the driver pitted
    pit_duration     REAL,              -- stationary time in pit box (seconds)
    compound_before  TEXT,
    compound_after   TEXT,
    PRIMARY KEY (year, round, driver_code, lap_number)
)
"""

CREATE_TELEMETRY_SUMMARY = """
CREATE TABLE IF NOT EXISTS telemetry_summary (
    year          INTEGER NOT NULL,
    round         INTEGER NOT NULL,
    driver_code   TEXT    NOT NULL,
    lap_number    INTEGER NOT NULL,
    mean_speed    REAL,   -- km/h
    max_speed     REAL,
    mean_throttle REAL,   -- 0–100
    brake_pct     REAL,   -- fraction of distance with brake applied
    drs_pct       REAL,   -- fraction of distance with DRS open
    mean_rpm      REAL,
    PRIMARY KEY (year, round, driver_code, lap_number)
)
"""

CREATE_WEATHER = """
CREATE TABLE IF NOT EXISTS weather (
    year           INTEGER NOT NULL,
    round          INTEGER NOT NULL,
    session_type   TEXT    NOT NULL,   -- 'R', 'Q', 'P1', 'P2', 'P3'
    time_offset    REAL    NOT NULL,   -- seconds from session start
    air_temp       REAL,               -- °C
    track_temp     REAL,               -- °C
    humidity       REAL,               -- %
    pressure       REAL,               -- mbar
    rainfall       INTEGER,            -- 0 or 1
    wind_speed     REAL,               -- m/s
    wind_direction REAL,               -- degrees (0–360)
    PRIMARY KEY (year, round, session_type, time_offset)
)
"""

CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    year               INTEGER NOT NULL,
    round              INTEGER NOT NULL,
    driver_code        TEXT    NOT NULL,
    predicted_position INTEGER,
    actual_position    INTEGER,         -- NULL for future (live) predictions
    source             TEXT    NOT NULL, -- 'cv' (backtested) or 'live' (pre-race)
    PRIMARY KEY (year, round, driver_code, source)
)
"""

# Ordered list used by F1Database.create_tables()
ALL_TABLES = [
    CREATE_RACES,
    CREATE_DRIVERS,
    CREATE_TEAMS,
    CREATE_SESSION_RESULTS,
    CREATE_QUALIFYING_RESULTS,
    CREATE_LAP_DATA,
    CREATE_PIT_STOPS,
    CREATE_TELEMETRY_SUMMARY,
    CREATE_WEATHER,
    CREATE_PREDICTIONS,
]
