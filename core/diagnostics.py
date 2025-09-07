# core/diagnostics.py
import numpy as np
import pandas as pd

def compute_shap_summary(model, X: pd.DataFrame):
    """
    Returns SHAP values for global feature importance visualization.
    """
    import shap
    explainer = shap.Explainer(model if not isinstance(model, list) else model[-1], X)
    shap_values = explainer(X)
    # Caller can plot: shap.plots.beeswarm(shap_values)
    return shap_values
