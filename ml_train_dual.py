# ml_train_dual.py
import os
import numpy as np
import pandas as pd
from core.features import make_features
from core.labeling import get_triple_barrier_labels
from core.model_selection import CombinatorialPurgedCV
from core.models import train_random_forest_cpcv
from core.calibration import fit_isotonic, apply_calibrator, fit_platt, apply_platt

def train_dual_side(
    df: pd.DataFrame,
    profit_take=0.004, stop_loss=0.004, tmax=60,
    cost_per_trade=0.0015,
    out_dir="artifacts", symbol="AAPL",
    calibration="isotonic"  # "isotonic" | "platt"
):
    X = make_features(df)
    labels = get_triple_barrier_labels(
        prices=df["close"], events=df.index,
        profit_take_pct=profit_take, stop_loss_pct=stop_loss, time_limit_periods=tmax
    )
    Z = X.join(labels[["label","t_final","ret"]], how="inner").dropna()
    Xz = Z[X.columns]
    y_up = (Z["label"] == 1).astype(int)
    y_dn = (Z["label"] == -1).astype(int)

    cpcv = CombinatorialPurgedCV(n_splits=3, embargo_pct=0.01)

    tr_up = train_random_forest_cpcv(
        Xz, y_up, cpcv, labels,
        class_weight="balanced",
        rf_params={"n_estimators": 500, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42},
        cost_per_trade=cost_per_trade, gain_per_win=profit_take, loss_per_lose=stop_loss
    )
    tr_dn = train_random_forest_cpcv(
        Xz, y_dn, cpcv, labels,
        class_weight="balanced",
        rf_params={"n_estimators": 500, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42},
        cost_per_trade=cost_per_trade, gain_per_win=profit_take, loss_per_lose=stop_loss
    )

    # Calibrate OOS probabilities for each side
    if calibration == "isotonic":
        cal_up = fit_isotonic(y_up.values, tr_up.oof_proba)
        cal_dn = fit_isotonic(y_dn.values, tr_dn.oof_proba)
        p_up_cal = apply_calibrator(cal_up, tr_up.oof_proba)
        p_dn_cal = apply_calibrator(cal_dn, tr_dn.oof_proba)
    else:
        cal_up = fit_platt(y_up.values, tr_up.oof_proba)
        cal_dn = fit_platt(y_dn.values, tr_dn.oof_proba)
        p_up_cal = apply_platt(cal_up, tr_up.oof_proba)
        p_dn_cal = apply_platt(cal_dn, tr_dn.oof_proba)

    os.makedirs(out_dir, exist_ok=True)
    out = pd.DataFrame({
        "timestamp": Z.index,
        "proba_up_raw": tr_up.oof_proba,
        "proba_dn_raw": tr_dn.oof_proba,
        "proba_up": p_up_cal,
        "proba_dn": p_dn_cal
    }).set_index("timestamp")
    out.to_csv(os.path.join(out_dir, f"{symbol}_oos_dual.csv"))
    with open(os.path.join(out_dir, f"{symbol}_thr_up.txt"), "w") as f: f.write(str(tr_up.threshold))
    with open(os.path.join(out_dir, f"{symbol}_thr_dn.txt"), "w") as f: f.write(str(tr_dn.threshold))

    return tr_up, tr_dn, out
