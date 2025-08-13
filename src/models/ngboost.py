import numpy as np
import pandas as pd
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import f1_score
from src.preprocess import preprocess_nested
from src.utils.analysis import FeatureSelector
import json
import os
from tqdm import tqdm
import random

def sample_params(param_space):
    """Randomly sample one parameter set from param_space dict"""
    return {k: random.choice(v) for k, v in param_space.items()}

def create_ngboost_model(params):
    """Create an NGBClassifier with base learner configured according to params dict"""
    base_learner = DecisionTreeRegressor(
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    )
    
    model = NGBClassifier(
        Dist=Bernoulli,
        Score=LogScore,
        Base=base_learner,
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        minibatch_frac=params['minibatch_frac'],
        col_sample=params['col_sample'],
        natural_gradient=params['natural_gradient'],
        verbose=False,
        random_state=42
    )
    return model

def hypertrain_nested_ngboost(X, y, df_cols, shap_selected, n_repeats, scaling=False, imputing=False, n_iter=20, corr_thresh=0.85, target_corr_thresh=0.1):
    model_name = 'ngboost'
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    param_space = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'minibatch_frac': [1.0],  
        'col_sample': [0.5, 0.8, 1.0],
        'natural_gradient': [True, False],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    all_hp_results = []
    models = []
    xs_test_list = []
    ys_test_list = []

    outer_cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=42)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    splits_iter = outer_cv.split(X, y)
    total_outer_folds = outer_cv.get_n_splits()

    for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits_iter, total=total_outer_folds, desc="Outer CV folds"), 1):
        X_train_full, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train_full, y_test = y[train_idx], y[test_idx]

        # Preprocess full training set first
        # 1. Preprocessing: impute + scale on train, apply on test
        X_train, scaler, imputer = preprocess_nested(X_train, scaling=scaling, imputing=imputing, training=True)
        X_test, _, _ = preprocess_nested(X_test, scaler=scaler, imputer=imputer, scaling=scaling, imputing=imputing, training=False)

        # 2. Feature selection on training fold only
        selector = FeatureSelector(corr_thresh, target_corr_thresh)
        X_train, y_train = selector.fit(X_train, y_train)
        X_test = selector.transform(X_test)

        best_inner_score = -np.inf
        best_params = None
        best_model = None

        inner_splits = list(inner_cv.split(X_train_full, y_train_full))

        for i in range(n_iter):
            params = sample_params(param_space)
            f1_scores = []

            for inner_train_idx, inner_val_idx in inner_splits:
                X_inner_train = X_train_full.iloc[inner_train_idx].reset_index(drop=True)
                y_inner_train = y_train_full[inner_train_idx]
                X_inner_val = X_train_full.iloc[inner_val_idx].reset_index(drop=True)
                y_inner_val = y_train_full[inner_val_idx]

                model = create_ngboost_model(params)

                try:
                    model.fit(X_inner_train, y_inner_train)
                except np.linalg.LinAlgError:
                    # Skip this trial if singular matrix encountered
                    f1_scores.append(0)
                    continue

                preds = model.predict(X_inner_val)
                score = f1_score(y_inner_val, preds)
                f1_scores.append(score)

            if len(f1_scores) == 0:
                mean_f1 = 0
            else:
                mean_f1 = np.mean(f1_scores)

            all_hp_results.append({
                **params,
                'fold': fold_idx,
                'inner_mean_f1': mean_f1
            })

            if mean_f1 > best_inner_score:
                best_inner_score = mean_f1
                best_params = params

        final_model = create_ngboost_model(best_params)
        final_model.fit(X_train_full, y_train_full)

        models.append(final_model)
        xs_test_list.append(X_test)
        ys_test_list.append(y_test)

        df_cv = pd.DataFrame([r for r in all_hp_results if r['fold'] == fold_idx])
        df_cv.to_csv(f'./models/{model_name}/cv_results_fold_{fold_idx}.csv', index=False)

    df_all_hp = pd.DataFrame(all_hp_results)
    df_all_hp.to_csv(f'./models/{model_name}/cv_results_all_folds.csv', index=False)

    return models, xs_test_list, ys_test_list