# shap_stability.py
import pandas as pd
from scipy.stats import spearmanr

def rank_stability(all_tbl, top_k=20):
    """
    all_tbl: output of shap_global_folds (per-fold table)
    Returns Spearman rank correlation matrix across folds for top_k features.
    """
    piv = all_tbl.pivot_table(index="feature", columns="fold", values="mean_abs_shap", aggfunc="mean").fillna(0.0)
    top = piv.mean(axis=1).sort_values(ascending=False).head(top_k).index
    mat = piv.loc[top]
    ranks = mat.rank(ascending=False)
    folds = ranks.columns
    corr = pd.DataFrame(index=folds, columns=folds, dtype=float)
    for i in folds:
        for j in folds:
            corr.loc[i, j] = spearmanr(ranks[i], ranks[j]).correlation
    return corr
