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
            # 1. Get the start and end times of the test set
            test_start_time = X.index[test_indices].min()
            test_end_time = X.index[test_indices].max()

            # 2. Get label end times for the initial training set
            train_label_ends = label_info.iloc[train_indices_initial]['t_final']

            # 3. Purge: Find training samples whose labels end AFTER the test set begins.
            # These labels overlap with the test period and must be removed.
            purged_train_indices = train_indices_initial[train_label_ends < test_start_time]

            # 4. Embargo: Remove training samples that are within the embargo period after the test set.
            embargo_start_time = test_end_time
            embargo_period = X.index[X.index > embargo_start_time]
            if not embargo_period.empty:
                embargo_end_time = embargo_period[min(embargo_size, len(embargo_period)-1)]
                embargo_indices = all_indices[(X.index >= embargo_start_time) & (X.index <= embargo_end_time)]
                final_train_indices = np.setdiff1d(purged_train_indices, embargo_indices)
            else:
                final_train_indices = purged_train_indices

            yield final_train_indices, test_indices