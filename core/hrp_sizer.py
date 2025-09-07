# core/hrp_sizer.py
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Dict, List

class HRPSizer:
    """
    Implements Hierarchical Risk Parity allocation.
    Consumes directional signals and computes target portfolio weights
    based on asset correlations and variances.
    """
    def _get_corr_dist(self, corr: pd.DataFrame) -> np.ndarray:
        # Distance matrix from the correlation matrix
        dist = np.sqrt((1 - corr) / 2)
        return squareform(dist)

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        # Sort items based on hierarchical clustering
        link = link.astype(int)
        sort_ix = pd.Series([link[0, 0], link[0, 1]])
        for i in range(1, link.shape[0]):
            a, b = link[i, 0], link[i, 1]
            if a in sort_ix.index:
                pos = sort_ix.index.get_loc(a)
                sort_ix = pd.concat([sort_ix.iloc[:pos+1], pd.Series([b]), sort_ix.iloc[pos+1:]])
            else:
                pos = sort_ix.index.get_loc(b)
                sort_ix = pd.concat([sort_ix.iloc[:pos], pd.Series([a]), sort_ix.iloc[:pos:]])
        return sort_ix.tolist()

    def _get_cluster_var(self, cov: pd.DataFrame, cluster_items: List[int]) -> float:
        # Compute variance of a cluster
        cov_ = cov.iloc[cluster_items, cluster_items]
        w_ = (1. / np.diag(cov_)).reshape(-1, 1)
        w_ /= w_.sum()
        c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
        return c_var

    def _get_rec_bisection(self, cov: pd.DataFrame, sort_ix: List[int]) -> pd.Series:
        # Recursive bisection to compute weights
        w = pd.Series(1, index=sort_ix, dtype=float)
        c_items = [sort_ix]
        while len(c_items) > 0:
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(c_items), 2):
                c1 = c_items[i]
                c2 = c_items[i+1]
                v1 = self._get_cluster_var(cov, c1)
                v2 = self._get_cluster_var(cov, c2)
                alpha = 1 - v1 / (v1 + v2)
                w[c1] *= alpha
                w[c2] *= (1 - alpha)
        return w

    def get_target_weights(
        self,
        sides: Dict[str, int],
        returns_window: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Inputs conform to the HRPSizerInterface contract.
        """
        active_symbols = list(sides.keys())
        if not active_symbols:
            return {}

        # Filter returns for active symbols and compute covariance
        returns = returns_window[active_symbols]
        cov = returns.cov()
        corr = returns.corr()

        # HRP algorithm
        dist = self._get_corr_dist(corr)
        link = linkage(dist, 'single')
        sort_ix = self._get_quasi_diag(link)
        
        # Reorder columns for sorted access
        sorted_symbols = corr.index[sort_ix].tolist()
        sorted_cov = cov.loc[sorted_symbols, sorted_symbols]
        
        # Get risk-parity weights
        hrp_weights = self._get_rec_bisection(sorted_cov, range(len(sorted_symbols)))
        hrp_weights.index = sorted_symbols
        
        # Apply sides and normalize
        final_weights = {sym: hrp_weights.get(sym, 0.0) * sides[sym] for sym in active_symbols}
        
        return final_weights