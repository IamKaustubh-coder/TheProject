# core/labeling.py
import pandas as pd

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
    """
    # 1. For each event in `events`:
    # 2.   Calculate the upper barrier price (price * (1 + profit_take_pct)).
    # 3.   Calculate the lower barrier price (price * (1 - stop_loss_pct)).
    # 4.   Look forward in the `prices` series for `time_limit_periods`.
    # 5.   Determine which of the three barriers was touched first.
    # 6.   Assign the label {-1, 0, 1} and record the barrier touch timestamp.
    # 7. Return the resulting DataFrame.
    pass