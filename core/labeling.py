
import pandas as pd
import numpy as np

def get_triple_barrier_labels(
    prices: pd.Series,
    events: pd.DatetimeIndex,
    profit_take_pct: float,
    stop_loss_pct: float,
    time_limit_periods: int
) -> pd.DataFrame:
    """
    For each event timestamp, determines the outcome based on three barriers.

    Args:
        prices: Series of prices indexed by datetime.
        events: Timestamps for which to generate labels (e.g., every bar).
        profit_take_pct: Percentage for the upper barrier.
        stop_loss_pct: Percentage for the lower barrier.
        time_limit_periods: Maximum holding period (vertical barrier).

    Returns:
        A DataFrame with columns:
        - 'label': The outcome (+1 for profit_take, -1 for stop_loss, 0 for time_limit).
        - 't_final': The timestamp when a barrier was touched.
        - 'ret': The return over the holding period.
    """
    results = []
    
    # Align events with the prices index
    events = events.intersection(prices.index)

    for t_event in events:
        t_start_loc = prices.index.get_loc(t_event)
        price_start = prices.iloc[t_start_loc]
        
        upper_barrier = price_start * (1 + profit_take_pct)
        lower_barrier = price_start * (1 - stop_loss_pct)
        
        t_end_loc = min(t_start_loc + time_limit_periods, len(prices) - 1)
        
        path = prices.iloc[t_start_loc:t_end_loc + 1]
        
        hits_up = path[path >= upper_barrier]
        hits_dn = path[path <= lower_barrier]
        
        t_up = hits_up.index[0] if not hits_up.empty else pd.NaT
        t_dn = hits_dn.index[0] if not hits_dn.empty else pd.NaT
        
        # Determine which barrier was hit first
        if pd.notna(t_up) and pd.notna(t_dn):
            t_final = min(t_up, t_dn)
        elif pd.notna(t_up):
            t_final = t_up
        elif pd.notna(t_dn):
            t_final = t_dn
        else:
            t_final = path.index[-1]

        # Assign label
        if t_final == t_up:
            label = 1
        elif t_final == t_dn:
            label = -1
        else:
            label = 0 # Time barrier hit
            
        ret = (prices.loc[t_final] / price_start) - 1
            
        results.append({
            't_event': t_event,
            't_final': t_final,
            'label': label,
            'ret': ret
        })

    if not results:
        return None

    return pd.DataFrame(results).set_index('t_event')
