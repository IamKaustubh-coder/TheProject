# ml_train_meta.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from core.models import cost_aware_threshold  # Reuse your threshold logic

def train_meta_model(
    primary_oos: pd.DataFrame,   # OOS df from train_dual_side
    labels: pd.DataFrame,        # Original triple-barrier labels
    primary_threshold: float,
    pt: float, sl: float, cost: float
):
    """
    Trains a meta-model to predict if the primary model will be correct.
    """
    # Align primary model predictions with the original labels
    df = primary_oos.join(labels, how="inner")
    
    # Generate meta-labels: 1 if primary signal was correct, 0 otherwise
    # A "correct" long signal is one that hit the profit-take barrier.
    is_primary_long_signal = (df['proba_up'] >= primary_threshold)
    is_actual_win = (df['label'] == 1)
    
    df['meta_label'] = 0
    df.loc[is_primary_long_signal, 'meta_label'] = is_actual_win[is_primary_long_signal].astype(int)
    
    # Feature for meta-model is the primary model's probability
    X_meta = df[['proba_up']]
    y_meta = df['meta_label']
    
    # Train a simple classifier (Logistic Regression is a good choice)
    meta_model = LogisticRegression(class_weight='balanced')
    meta_model.fit(X_meta, y_meta)
    
    # Calibrate a threshold for the meta-model
    meta_proba = meta_model.predict_proba(X_meta)[:, 1]
    meta_threshold = cost_aware_threshold(y_meta, meta_proba, pt, sl, cost)
    
    return meta_model, meta_threshold