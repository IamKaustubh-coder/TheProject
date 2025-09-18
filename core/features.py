# core/features.py
from __future__ import annotations
import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame, price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    """
    Input: dataframe indexed by datetime with columns: open, high, low, close, volume
    Output: feature dataframe aligned to df.index (NaNs retained for caller to drop after alignment with labels)
    """
    f = pd.DataFrame(index=df.index.copy())
    px = df[price_col].astype(float)
    vol = df[vol_col].astype(float)

    # Simple returns and log returns
    f["ret_1"] = px.pct_change()  # one-period simple return
    f["log_ret_1"] = np.log(px).diff()

    # Rolling realized volatility (sum of squared intraperiod returns over window) as HF volatility proxy
    # For 1-minute bars, use trailing windows (e.g., 5, 15, 60) to capture short- and medium-horizon risk
    for w in [5, 15, 60]:
        r = f["log_ret_1"]
        f[f"rv_{w}"] = np.sqrt((r.rolling(w).apply(lambda x: np.sum(x**2), raw=True)))  # realized vol
    # [Andersen & Benzoni overview of realized volatility] [23]

    # Range-based volatility proxies (Parkinson-like) using high/low
    hl_range = (df["high"] - df["low"]).astype(float)
    for w in [5, 15, 60]:
        f[f"range_mean_{w}"] = hl_range.rolling(w).mean()
        f[f"range_std_{w}"] = hl_range.rolling(w).std(ddof=1)

    # Momentum / mean-reversion summaries
    for w in [5, 15, 30, 60]:
        f[f"mom_{w}"] = px.pct_change(w)
        f[f"zscore_{w}"] = (px - px.rolling(w).mean()) / (px.rolling(w).std(ddof=1) + 1e-8)

    # Volume normalization and shocks
    for w in [20, 60, 120]:
        f[f"vol_z_{w}"] = (vol - vol.rolling(w).mean()) / (vol.rolling(w).std(ddof=1) + 1e-8)
        f[f"val_traded_{w}"] = (px * vol).rolling(w).mean()  # proxy for dollar activity

    # Technical summaries (keep implementation-light; aligns with ta/pandas-ta availability if swapped later)
    # SMA differentials, RSI-lite (can be replaced with ta library later)
    for s, l in [(10, 30), (20, 50)]:
        sma_s = px.rolling(s).mean()
        sma_l = px.rolling(l).mean()
        f[f"sma_diff_{s}_{l}"] = (sma_s - sma_l) / (sma_l + 1e-8)

    # RSI (Wilder-like) minimal
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs))
    f["rsi_14"] = rsi(px, 14)

    return f
