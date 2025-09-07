# core/metrics.py
import math
import numpy as np
import pandas as pd

def equity_to_returns(equity_df: pd.DataFrame) -> pd.Series:
    """
    Expects columns ['timestamp','equity'] and returns pct_change indexed by timestamp.
    """
    ser = pd.Series(equity_df["equity"].values, index=pd.to_datetime(equity_df["timestamp"]))
    rets = ser.pct_change().dropna()
    return rets

def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Compound returns to annualized rate.
    """
    cum = (1.0 + returns).prod()
    years = len(returns) / float(periods_per_year)
    if years <= 0:
        return 0.0
    return cum ** (1.0 / years) - 1.0

def sharpe_ratio(returns: pd.Series, rf_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sharpe = (mean excess return per period / std per period) * sqrt(periods_per_year).
    rf_rate is the ANNUAL risk-free rate; converted to per-period assuming simple division.
    """
    if returns.empty:
        return float("nan")
    rf_per_period = rf_rate / periods_per_year
    excess = returns - rf_per_period
    std = excess.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return (excess.mean() / std) * math.sqrt(periods_per_year)

def sortino_ratio(returns: pd.Series, rf_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sortino = (mean excess return per period / downside std per period) * sqrt(periods_per_year).
    """
    if returns.empty:
        return float("nan")
    rf_per_period = rf_rate / periods_per_year
    excess = returns - rf_per_period
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        return float("inf")
    dd = downside.std(ddof=1)
    if dd == 0 or math.isnan(dd):
        return float("nan")
    return (excess.mean() / dd) * math.sqrt(periods_per_year)

def drawdown_stats(equity_df: pd.DataFrame):
    """
    Returns (drawdown_series, max_drawdown, max_drawdown_start, max_drawdown_end)
    """
    eq = pd.Series(equity_df["equity"].values, index=pd.to_datetime(equity_df["timestamp"]))
    running_max = eq.cummax()
    dd = (eq - running_max) / running_max
    mdd = dd.min()
    # Approximate start/end
    end = dd.idxmin()
    start = eq.loc[:end].idxmax()
    return dd, float(mdd), start, end

def calmar_ratio(returns: pd.Series, equity_df: pd.DataFrame, periods_per_year: int = 252) -> float:
    """
    Calmar = annualized return / abs(max drawdown).
    """
    ann = annualize_return(returns, periods_per_year)
    _, mdd, _, _ = drawdown_stats(equity_df)
    if mdd == 0:
        return float("inf")
    return ann / abs(mdd)

def summarize_performance(equity_df: pd.DataFrame, rf_rate: float = 0.0, periods_per_year: int = 252) -> dict:
    """
    Computes key KPIs from an equity curve DataFrame with columns ['timestamp','equity'].
    """
    rets = equity_to_returns(equity_df)
    ann = annualize_return(rets, periods_per_year)
    shrp = sharpe_ratio(rets, rf_rate, periods_per_year)
    srtn = sortino_ratio(rets, rf_rate, periods_per_year)
    dd_series, mdd, dd_start, dd_end = drawdown_stats(equity_df)
    calmar = calmar_ratio(rets, equity_df, periods_per_year)
    return {
        "annualized_return": ann,
        "sharpe": shrp,
        "sortino": srtn,
        "max_drawdown": mdd,
        "max_drawdown_start": dd_start,
        "max_drawdown_end": dd_end,
        "calmar": calmar,
    }
