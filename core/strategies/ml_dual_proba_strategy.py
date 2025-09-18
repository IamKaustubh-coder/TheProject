# core/strategies/ml_dual_proba_strategy.py
from typing import Dict, List
import pandas as pd
from ..events import MarketEvent, SignalEvent

class MLDualProbaStrategy:
    def __init__(self, symbol_to_df: Dict[str, pd.DataFrame], thr_up: Dict[str, float], thr_dn: Dict[str, float]):
        """
        symbol_to_df[sym]: DataFrame indexed by timestamp with ['proba_up','proba_dn']
        """
        self.pfeeds = symbol_to_df
        self.tu = thr_up
        self.td = thr_dn

    def on_market(self, event: MarketEvent) -> List[SignalEvent]:
        sym = event.symbol
        df = self.pfeeds.get(sym)
        if df is None or event.timestamp not in df.index:
            return []
        pu = float(df.loc[event.timestamp, "proba_up"])
        pdn = float(df.loc[event.timestamp, "proba_dn"])
        tu = float(self.tu.get(sym, 0.6))
        td = float(self.td.get(sym, 0.6))

        cand = []
        if pu >= tu:
            cand.append(("LONG", pu - tu))
        if pdn >= td:
            cand.append(("SHORT", pdn - td))

        sigs: List[SignalEvent] = []
        if cand:
            best_signal = max(cand, key=lambda x: x[1])
            direction = best_signal[0]
            strength = pu if direction == "LONG" else pdn
            sigs.append(SignalEvent(timestamp=event.timestamp, symbol=sym, direction=direction, strength=strength))
        return sigs
