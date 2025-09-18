# core/model_selection.py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Generator, Tuple

class CombinatorialPurgedCV:
    """
    Implements Combinatorial Purged Cross-Validation (CPCV).
    Ensures train/test splits are leakage-free by:
    1. Purging training samples whose labels overlap with the test set.
    2. Embargoing a small number of samples after the test set.
    """
    def __init__(self, n_splits: int = 10, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        label_info: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates train/test indices.

        Args:
            X: Feature set, indexed by datetime.
            y: Labels (not directly used for splitting, but for API consistency).
            label_info: DataFrame from get_triple_barrier_labels, must contain 't_final'.
        """
        if not isinstance(X.index, pd.DatetimeIndex) or not isinstance(label_info.index, pd.DatetimeIndex):
            raise ValueError("X and label_info must be indexed by a DatetimeIndex.")
        
        # Ensure label_info is aligned with X's index
        label_info = label_info.loc[X.index]
        
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        all_indices = np.arange(len(X))
        
        # Calculate embargo size in terms of number of data points
        embargo_size = int(len(X) * self.embargo_pct)

        for train_indices_initial, test_indices in kf.split(all_indices):
            test_start_time = X.index[test_indices].min()
            test_end_time = X.index[test_indices].max()

            train_indices_before = train_indices_initial[X.index[train_indices_initial] < test_start_time]
            train_indices_after = train_indices_initial[X.index[train_indices_initial] > test_end_time]

            # Purge the "before" part
            train_label_ends_before = label_info.iloc[train_indices_before]['t_final']
            purged_train_indices_before = train_indices_before[train_label_ends_before < test_start_time]

            # Embargo the "after" part
            embargo_start_time = test_end_time
            embargo_period = X.index[X.index > embargo_start_time]
            if not embargo_period.empty:
                embargo_end_time = embargo_period[min(embargo_size, len(embargo_period)-1)]
                embargoed_train_indices_after = train_indices_after[X.index[train_indices_after] > embargo_end_time]
            else:
                embargoed_train_indices_after = train_indices_after

            final_train_indices = np.concatenate([purged_train_indices_before, embargoed_train_indices_after])

            yield final_train_indices, test_indices