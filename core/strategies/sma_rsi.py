# core/strategies/sma_rsi.py
from collections import deque
from typing import Dict, List
import numpy as np

from ..events import MarketEvent, SignalEvent
from ..strategy import Strategy

class SmaRsiStrategy(Strategy):
    """
    SMA crossover filtered by RSI momentum.
    - LONG when SMA_short > SMA_long and RSI >= rsi_long_threshold
    - SHORT when SMA_short < SMA_long and RSI <= rsi_short_threshold
    Neutral otherwise.
    """
    def __init__(
        self,
        symbols: List[str],
        short_window: int = 20,
        long_window: int = 50,
        rsi_period: int = 14,
        rsi_long_threshold: float = 55.0,
        rsi_short_threshold: float = 45.0,
        max_history: int = 1000
    ):
        assert short_window < long_window, "short_window must be < long_window"
        self.symbols = symbols
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_long_th = rsi_long_threshold
        self.rsi_short_th = rsi_short_threshold

        # Per-symbol rolling close history
        self.closes: Dict[str, deque] = {sym: deque(maxlen=max_history) for sym in symbols}
        # Track last crossover direction to avoid excessive signals
        self.last_state: Dict[str, str] = {sym: "NEUTRAL" for sym in symbols}

    def _sma(self, arr: np.ndarray, window: int) -> float:
        if arr.size < window:
            return np.nan
        return float(np.mean(arr[-window:]))

    def _rsi(self, arr: np.ndarray, period: int) -> float:
        """
        Wilder's RSI calculation.
        """
        if arr.size < period + 1:
            return np.nan
        delta = np.diff(arr[-(period + 1):])
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def on_market(self, event: MarketEvent) -> List[SignalEvent]:
        sym = event.symbol
        close = float(event.ohlcv["close"])
        self.closes[sym].append(close)

        arr = np.array(self.closes[sym], dtype=float)
        sma_s = self._sma(arr, self.short_window)
        sma_l = self._sma(arr, self.long_window)
        rsi = self._rsi(arr, self.rsi_period)

        signals: List[SignalEvent] = []
        if not np.isnan([sma_s, sma_l, rsi]).any():
            # Determine desired state
            desired = "NEUTRAL"
            if sma_s > sma_l and rsi >= self.rsi_long_th:
                desired = "LONG"
            elif sma_s < sma_l and rsi <= self.rsi_short_th:
                desired = "SHORT"

            # Emit signal only on state change
            if desired != self.last_state[sym]:
                if desired in ("LONG", "SHORT"):
                    signals.append(
                        SignalEvent(
                            timestamp=event.timestamp,
                            symbol=sym,
                            direction=desired,
                            strength=1.0
                        )
                    )
                else:
                    signals.append(
                        SignalEvent(
                            timestamp=event.timestamp,
                            symbol=sym,
                            direction="EXIT",
                            strength=1.0
                        )
                    )
                self.last_state[sym] = desired

        return signals
