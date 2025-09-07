# shap_local.py
import pandas as pd
import shap

def local_explanations(model, X_sample, y_outcome, n=200):
    """
    X_sample: OOS rows with realized outcome labels for case-control review.
    y_outcome: realized barrier outcome (e.g., 1 = hit PT, 0/-1 otherwise).
    """
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X_sample)
    if isinstance(sv, list):
        sv = sv[1]
    # Split winners/losers for qualitative inspection
    winners = X_sample[y_outcome == 1].head(n)
    losers  = X_sample[y_outcome != 1].head(n)
    return winners, losers, sv
