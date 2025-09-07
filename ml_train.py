# core/predict_adapter.py
from typing import List
import numpy as np
import pandas as pd
from .events import SignalEvent

class ProbaToSignals:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def to_signals(self, timestamp: pd.Timestamp, symbol: str, proba_up: float) -> List[SignalEvent]:
        out: List[SignalEvent] = []
        if proba_up >= self.threshold:
            out.append(SignalEvent(timestamp=timestamp.to_pydatetime(), symbol=symbol, direction="LONG", strength=proba_up))
        # Optional: symmetric short if proba_down >= threshold; requires calibrated proba_down
        return out
