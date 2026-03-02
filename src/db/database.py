"""
F1Database: thin wrapper around DuckDB for all read/write operations.

Why DuckDB over SQLite:
  - Identical SQL syntax you already know
  - Single file, no server required (same dev experience)
  - Columnar storage → GROUP BY / window function queries are 10-100x faster,
    which matters a lot for feature engineering aggregations
  - Native .df() method returns pandas DataFrames directly
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

from src.db.schema import ALL_TABLES


class F1Database:
    """Manages the DuckDB connection and all CRUD operations."""

    def __init__(self, db_path: str = "data/f1_data.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    # ------------------------------------------------------------------
    # Connection management (supports both 'with' and manual open/close)
    # ------------------------------------------------------------------

    def connect(self) -> "F1Database":
        self._conn = duckdb.connect(self.db_path)
        return self

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "F1Database":
        return self.connect()

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError(
                "Database is not connected. Use 'with F1Database() as db:' "
                "or call db.connect() first."
            )
        return self._conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Create all tables if they do not already exist."""
        for ddl in ALL_TABLES:
            self.conn.execute(ddl)

    # ------------------------------------------------------------------
    # Generic query helpers
    # ------------------------------------------------------------------

    def query(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        """Run a SELECT statement and return results as a DataFrame."""
        if params:
            return self.conn.execute(sql, params).df()
        return self.conn.execute(sql).df()

    def execute(self, sql: str, params: Optional[list] = None) -> None:
        """Run a non-SELECT statement (INSERT, UPDATE, DELETE, DDL)."""
        if params:
            self.conn.execute(sql, params)
        else:
            self.conn.execute(sql)

    def insert_df(self, df: pd.DataFrame, table: str, on_conflict: str = "DO NOTHING") -> None:
        """
        Bulk-insert a DataFrame into *table*.

        DuckDB lets us register a DataFrame as a virtual table and then
        INSERT ... SELECT from it, which is much faster than row-by-row inserts.

        on_conflict: 'DO NOTHING'  – skip duplicate rows (default)
                     'DO UPDATE …' – you must provide the full SET clause
        """
        if df.empty:
            return
        view = f"_tmp_{table}"
        self.conn.register(view, df)
        try:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {table} SELECT * FROM {view}"
            )
        finally:
            self.conn.unregister(view)

    # ------------------------------------------------------------------
    # Existence / lookup helpers
    # ------------------------------------------------------------------

    def race_exists(self, year: int, round_num: int) -> bool:
        result = self.query(
            "SELECT COUNT(*) AS cnt FROM races WHERE year = ? AND round = ?",
            [year, round_num],
        )
        return int(result["cnt"].iloc[0]) > 0

    def session_results_exist(self, year: int, round_num: int) -> bool:
        result = self.query(
            "SELECT COUNT(*) AS cnt FROM session_results WHERE year = ? AND round = ?",
            [year, round_num],
        )
        return int(result["cnt"].iloc[0]) > 0

    def lap_data_exists(self, year: int, round_num: int) -> bool:
        result = self.query(
            "SELECT COUNT(*) AS cnt FROM lap_data WHERE year = ? AND round = ?",
            [year, round_num],
        )
        return int(result["cnt"].iloc[0]) > 0

    # ------------------------------------------------------------------
    # Convenience read methods
    # ------------------------------------------------------------------

    def get_races(self, year: Optional[int] = None) -> pd.DataFrame:
        if year is not None:
            return self.query(
                "SELECT * FROM races WHERE year = ? ORDER BY round", [year]
            )
        return self.query("SELECT * FROM races ORDER BY year, round")

    def get_lap_data(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> pd.DataFrame:
        if year is not None and round_num is not None:
            return self.query(
                "SELECT * FROM lap_data WHERE year = ? AND round = ? ORDER BY driver_code, lap_number",
                [year, round_num],
            )
        if year is not None:
            return self.query(
                "SELECT * FROM lap_data WHERE year = ? ORDER BY round, driver_code, lap_number",
                [year],
            )
        return self.query("SELECT * FROM lap_data ORDER BY year, round, driver_code, lap_number")

    def get_session_results(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> pd.DataFrame:
        if year is not None and round_num is not None:
            return self.query(
                "SELECT * FROM session_results WHERE year = ? AND round = ?",
                [year, round_num],
            )
        if year is not None:
            return self.query(
                "SELECT * FROM session_results WHERE year = ?", [year]
            )
        return self.query("SELECT * FROM session_results ORDER BY year, round")

    def get_qualifying_results(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> pd.DataFrame:
        if year is not None and round_num is not None:
            return self.query(
                "SELECT * FROM qualifying_results WHERE year = ? AND round = ?",
                [year, round_num],
            )
        if year is not None:
            return self.query(
                "SELECT * FROM qualifying_results WHERE year = ?", [year]
            )
        return self.query("SELECT * FROM qualifying_results ORDER BY year, round")

    def get_weather(
        self,
        year: int,
        round_num: int,
        session_type: str = "R",
    ) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM weather WHERE year = ? AND round = ? AND session_type = ? ORDER BY time_offset",
            [year, round_num, session_type],
        )

    def get_pit_stops(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> pd.DataFrame:
        if year is not None and round_num is not None:
            return self.query(
                "SELECT * FROM pit_stops WHERE year = ? AND round = ?",
                [year, round_num],
            )
        return self.query("SELECT * FROM pit_stops ORDER BY year, round")

    def get_telemetry_summary(
        self,
        year: int,
        round_num: int,
    ) -> pd.DataFrame:
        return self.query(
            "SELECT * FROM telemetry_summary WHERE year = ? AND round = ?",
            [year, round_num],
        )

    def get_predictions(
        self,
        year: Optional[int] = None,
        round_num: Optional[int] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        filters, params = [], []
        if year is not None:
            filters.append("year = ?");  params.append(year)
        if round_num is not None:
            filters.append("round = ?"); params.append(round_num)
        if source is not None:
            filters.append("source = ?"); params.append(source)
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        return self.query(
            f"SELECT * FROM predictions {where} ORDER BY year, round, predicted_position",
            params or None,
        )
