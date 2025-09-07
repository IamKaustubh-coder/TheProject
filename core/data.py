# core/data.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Iterator
import pandas as pd

from .events import MarketEvent

class DataHandler(ABC):
    """Abstract interface for any market data feed."""

    @abstractmethod
    def update_bars(self):
        """Advance the data feed by one step and emit MarketEvent(s)."""
        raise NotImplementedError

    @abstractmethod
    def has_data(self) -> bool:
        """Return True if any symbol has more data to stream."""
        raise NotImplementedError


class CSVDataHandler(DataHandler):
    """
    Streams OHLCV rows from one or more CSVs.
    Expected columns: datetime, open, high, low, close, volume
    datetime must parse to a timezone-aware or naive datetime usable for ordering.
    """
    def __init__(self, event_queue, symbol_to_csv: Dict[str, str], datetime_col: str = "datetime"):
        self.event_queue = event_queue
        self.datetime_col = datetime_col
        self.symbols = list(symbol_to_csv.keys())
        self._frames: Dict[str, pd.DataFrame] = {}
        self._iters: Dict[str, Iterator] = {}

        # Load and sort each CSV by datetime, set index if desired
        for sym, path in symbol_to_csv.items():
            df = pd.read_csv(path)
            df[self.datetime_col] = pd.to_datetime(df[self.datetime_col])
            df = df.sort_values(self.datetime_col).reset_index(drop=True)
            self._frames[sym] = df
            self._iters[sym] = df.iterrows()

        # Track the next row per symbol
        self._next_cache: Dict[str, Dict] = {}
        for sym in self.symbols:
            try:
                _, row = next(self._iters[sym])
                self._next_cache[sym] = row
            except StopIteration:
                self._next_cache[sym] = None

    def has_data(self) -> bool:
        return any(self._next_cache[sym] is not None for sym in self.symbols)

    def update_bars(self):
        """
        Find the earliest next timestamp across all symbols, emit MarketEvent(s) for that timestamp,
        and advance those symbols by one step.
        """
        # Determine the earliest timestamp available
        times = []
        for sym in self.symbols:
            row = self._next_cache[sym]
            if row is not None:
                times.append(row[self.datetime_col])
        if not times:
            return  # No data left

        current_time = min(times)

        # Emit MarketEvent for all symbols that share this current_time
        for sym in self.symbols:
            row = self._next_cache[sym]
            if row is not None and row[self.datetime_col] == current_time:
                ohlcv = {
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low":  float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                }
                me = MarketEvent(timestamp=current_time.to_pydatetime(), symbol=sym, ohlcv=ohlcv)
                self.event_queue.put(me)

                # Advance iterator for this symbol
                try:
                    _, nxt = next(self._iters[sym])
                    self._next_cache[sym] = nxt
                except StopIteration:
                    self._next_cache[sym] = None
