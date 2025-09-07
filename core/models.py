# core/models.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Optional: XGBoost if available in environment
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

@dataclass
class TrainResult:
    model: Any
    oof_pred: np.ndarray
    oof_proba: np.ndarray
    folds: List[Tuple[np.ndarray, np.ndarray]]
    metrics: Dict[str, float]
    threshold: float

def cost_aware_threshold(y_true: np.ndarray, proba: np.ndarray, gain_per_win: float, loss_per_lose: float, cost_per_trade: float) -> float:
    """
    Grid-search a probability threshold for entering trades to maximize expected net return per decision:
    E[pi] = p*gain - (1-p)*loss - cost, applied to candidates across [0.5..0.9].
    """
    best_t, best_val = 0.5, -1e18
    for t in np.linspace(0.5, 0.9, 41):
        y_hat = (proba >= t).astype(int)
        p = ((y_true == 1) & (y_hat == 1)).sum() / max((y_hat == 1).sum(), 1)
        # approximate expected per-trade payoff
        exp_val = p * gain_per_win - (1 - p) * loss_per_lose - cost_per_trade
        if exp_val > best_val:
            best_val, best_t = exp_val, t
    return best_t

def train_random_forest_cpcv(
    X: pd.DataFrame, y: pd.Series, cpcv_splitter, class_weight: Dict[int, float] | str = "balanced",
    rf_params: Dict[str, Any] | None = None,
    cost_per_trade: float = 0.0015,  # example: 15 bps round-trip
    gain_per_win: float = 0.002,     # example payoff per hit (tuned to PT/SL)
    loss_per_lose: float = 0.002
) -> TrainResult:
    rf_params = rf_params or {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42}
    oof_pred = np.zeros(len(X))
    oof_proba = np.zeros(len(X))
    folds = []
    models = []

    for tr_idx, te_idx in cpcv_splitter.split(X, y, label_info=None):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        clf = RandomForestClassifier(class_weight=class_weight, **rf_params)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_proba[te_idx] = proba
        oof_pred[te_idx] = pred
        models.append(clf)
        folds.append((tr_idx, te_idx))

    auc = roc_auc_score(y, oof_proba) if len(np.unique(y)) > 1 else np.nan
    thr = cost_aware_threshold(y.values, oof_proba, gain_per_win, loss_per_lose, cost_per_trade)

    return TrainResult(model=models[-1], oof_pred=oof_pred, oof_proba=oof_proba, folds=folds, metrics={"auc": auc}, threshold=thr)

def train_xgboost_cpcv(
    X: pd.DataFrame, y: pd.Series, cpcv_splitter, xgb_params: Dict[str, Any] | None = None,
    cost_per_trade: float = 0.0015, gain_per_win: float = 0.002, loss_per_lose: float = 0.002
) -> TrainResult:
    assert HAS_XGB, "XGBoost not available"
    xgb_params = xgb_params or {
        "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0,
        "random_state": 42, "tree_method": "hist", "n_jobs": -1, "objective": "binary:logistic"
    }
    oof_pred = np.zeros(len(X))
    oof_proba = np.zeros(len(X))
    folds = []
    models = []

    for tr_idx, te_idx in cpcv_splitter.split(X, y, label_info=None):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        clf = XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_proba[te_idx] = proba
        oof_pred[te_idx] = pred
        models.append(clf)
        folds.append((tr_idx, te_idx))

    auc = roc_auc_score(y, oof_proba) if len(np.unique(y)) > 1 else np.nan
    thr = cost_aware_threshold(y.values, oof_proba, gain_per_win, loss_per_lose, cost_per_trade)

    return TrainResult(model=models[-1], oof_pred=oof_pred, oof_proba=oof_proba, folds=folds, metrics={"auc": auc}, threshold=thr)
