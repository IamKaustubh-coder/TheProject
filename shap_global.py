# shap_global.py
import numpy as np
import pandas as pd
import shap

def shap_global_folds(models, X_folds, feature_names):
    """
    models: list of trained tree models (one per fold in CPCV or WFO)
    X_folds: list of DataFrames for each fold's validation/OOS X (aligned with the model)
    Returns: DataFrame mean|SHAP| per feature averaged across folds and fold-wise table.
    """
    fold_tables = []
    for m, Xoos in zip(models, X_folds):
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(Xoos)  # for binary clf, may return array or list
        if isinstance(sv, list):  # xgboost-style [neg_class, pos_class]
            sv = sv[1]
        abs_mean = np.abs(sv).mean(axis=0)
        fold_tables.append(pd.DataFrame({"feature": feature_names, "mean_abs_shap": abs_mean}))
    all_tbl = pd.concat(fold_tables, keys=range(len(fold_tables)), names=["fold"]).reset_index(level=0)
    agg = all_tbl.groupby("feature")["mean_abs_shap"].agg(["mean","std"]).sort_values("mean", ascending=False)
    return agg, all_tbl

# Example usage:
# agg_up, all_up = shap_global_folds(models_up, X_oos_folds_up, X_cols)
# agg_dn, all_dn = shap_global_folds(models_dn, X_oos_folds_dn, X_cols)
# agg_up.head(25), agg_dn.head(25)
