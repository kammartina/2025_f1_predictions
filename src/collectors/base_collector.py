"""Abstract base class for all data collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.db.database import F1Database


class BaseCollector(ABC):
    """
    All collectors receive a shared F1Database instance and implement collect().
    The database connection is managed by the caller (F1Pipeline), not here.
    """

    def __init__(self, db: F1Database) -> None:
        self.db = db

    @abstractmethod
    def collect(self, year: int, round_num: int, **kwargs: bool) -> None:
        """
        Fetch data for one race weekend and persist it to the database.

        Parameters
        ----------
        year      : Calendar year (e.g. 2025)
        round_num : Race round number on the F1 calendar (1-based)
        **kwargs  : Collector-specific flags (e.g. include_telemetry=True)
        """
        ...
