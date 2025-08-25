from src.utils.helper_functions import *
from src.preprocess import *
from tabpfn import TabPFNClassifier

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import class_weight
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
import sys
from src.utils.analysis import FeatureSelector
import matplotlib.pyplot as plt
import pickle
import numpy as np
import joblib
from tqdm import tqdm
import pandas as pd
import json


def hypertrain_nested_tabpfn(X, y, df_cols, shap_selected, n_repeats, scaling=False, imputing=False, corr_thresh=0.85, 
                             target_corr_thresh=0.1,device='cpu', n_ensemble=4):
    """
    device: 'cpu' or 'cuda' 
    n_ensemble: TabPFN ensemble size (smaller -> faster)
    """

    model_name = 'tabpfn'
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    all_results = []
    models = []
    xs_test_list = []
    ys_test_list = []

    outer_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=42)
    splits_iter = outer_cv.split(X, y)
    total_outer_folds = outer_cv.get_n_splits()

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits_iter, total=total_outer_folds, desc="Outer CV folds (TabPFN)"), 1):
        # split and reset indices (keep DataFrame for FS)
        X_train_df, X_test_df = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]

        # 1) Preprocessing: impute + scale on train, apply on test
        X_train_df, scaler, imputer = preprocess_nested(X_train_df, scaling=scaling, imputing=imputing, training=True)
        X_test_df, _, _ = preprocess_nested(X_test_df, scaler=scaler, imputer=imputer, scaling=scaling, imputing=imputing, training=False)

        # 2) Feature selection on training fold only
        selector = FeatureSelector(corr_thresh=corr_thresh, target_corr_thresh=target_corr_thresh)
        X_train, y_train = selector.fit(X_train_df, y_train)
        X_test = selector.transform(X_test_df)

        '''
        # Convert to numpy arrays for TabPFN
        X_train_np = X_train_selected_df.to_numpy(dtype=float)
        X_test_np = X_test_selected_df.to_numpy(dtype=float)
        y_train_np = np.asarray(y_train_sel)
        y_test_np = np.asarray(y_test)
        '''

        # 3) Instantiate TabPFN classifier
        # TabPFNClassifier API: TabPFNClassifier( N_ensemble=..., device='cpu'/'cuda' )
        clf = TabPFNClassifier(N_ensemble=n_ensemble, device=device)

        # NOTE: TabPFN uses an internal "fit" that actually performs its in-context inference; it's fast.
        try:
            clf.fit(X_train, y_train)
        except TypeError:
            # older/newer versions might have slightly different signatures
            clf.fit(X_train, y_train, sample_weight=None)

        # Save the fitted classifier and test data for later evaluation
        models.append(clf)
        xs_test_list.append(X_test)  # keep both df and np for convenience
        ys_test_list.append(y_test)

        # Optionally save per-fold predictions/probabilities right away
        try:
            probs = clf.predict_proba(X_test)
            preds = clf.predict(X_test)
        except Exception:
            # fallback: if predict_proba not available, use predict only
            preds = clf.predict(X_test)
            probs = None

        # Save fold outputs to disk
        fold_out = {
            'fold': fold_idx,
            'y_test': y_test.tolist(),
            'y_pred': preds.tolist()
        }
        if probs is not None:
            fold_out['y_proba'] = probs.tolist()
        with open(f'./outputs/{model_name}/fold_{fold_idx}_results.json', 'w') as f:
            json.dump(fold_out, f)

    return models, xs_test_list, ys_test_list
