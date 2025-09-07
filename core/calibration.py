# core/calibration.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def fit_isotonic(y_true: np.ndarray, proba: np.ndarray):
    """
    Fits isotonic regression to map raw proba -> calibrated proba.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(proba, y_true)
    return ir

def apply_calibrator(calibrator, proba: np.ndarray) -> np.ndarray:
    return calibrator.predict(proba)

def fit_platt(y_true: np.ndarray, proba: np.ndarray):
    """
    Platt scaling via logistic regression on raw probabilities.
    """
    x = proba.reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x, y_true)
    return lr

def apply_platt(calibrator, proba: np.ndarray) -> np.ndarray:
    x = proba.reshape(-1, 1)
    return calibrator.predict_proba(x)[:, 1]
