import os

from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret
from src.preprocess import preprocess_nested
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.utils import class_weight
from src.utils.analysis import FeatureSelector
import xgboost as xgb
import te2rules
from te2rules.explainer import ModelExplainer
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier
import joblib
import json
import os
import pickle

def hypertrain_nested_xgb(X, y, df_cols, shap_selected, n_repeats, scaling=False, imputing=False, corr_thresh=0.85, target_corr_thresh=0.1):
    model_name = 'xgb'
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )

    hp_space = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 1.5, 2.0]
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
        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y[train_idx], y[test_idx]

        # 1. Preprocessing: impute + scale on train, apply on test
        X_train, scaler, imputer = preprocess_nested(X_train, scaling=scaling, imputing=imputing, training=True)
        X_test, _, _ = preprocess_nested(X_test, scaler=scaler, imputer=imputer, scaling=scaling, imputing=imputing, training=False)

        # 2. Feature selection on training fold only
        selector = FeatureSelector(corr_thresh, target_corr_thresh)
        X_train, y_train = selector.fit(X_train, y_train)
        X_test = selector.transform(X_test)

        clf = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=hp_space,
            cv=inner_cv,
            scoring='f1',
            n_iter=100,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        # Save cross-validation results
        df_cv = pd.DataFrame(clf.cv_results_)
        df_cv['params_str'] = df_cv['params'].apply(lambda x: json.dumps(x))
        df_cv['fold'] = fold_idx
        df_cv.to_csv(f'./models/{model_name}/cv_results_fold_{fold_idx}.csv', index=False)
        all_hp_results.append(df_cv)

        models.append(best_model)
        xs_test_list.append(X_test)
        ys_test_list.append(y_test)

    # Save all CV results
    df_all_hp = pd.concat(all_hp_results, ignore_index=True)
    df_all_hp.to_csv(f'./models/{model_name}/cv_results_all_folds.csv', index=False)

    return models, xs_test_list, ys_test_list




def hypertrain_ensemble_xgboost(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                shap_selected, interpret_model=True, testing=True):
    models = []
    model_name = 'xgb_shap_selected' if shap_selected else 'xgb'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    # Train models
    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training model {idx}')
        models.append(train_xgb_model_from_paper(X_train, y_train, X_val, y_val))

    # Save models
    model_path = f'models/{model_name}/model.pickle'
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    # Optionally interpret models
    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols, model_name=model_name)

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_xgboost(xs_test, ys_test, df_cols, shap_selected=shap_selected, model_name=model_name)


def evaluate_ensemble_xgboost(xs_test, ys_test, df_cols, shap_selected, model_name='xgb'):
    if 'shap_selected' not in model_name and shap_selected:
        model_name = f'{model_name}_shap_selected'

    checkpoint_file = \
    [f"./models/{model_name}/{f}" for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pickle')][0]

    # Load the models from the saved pickle file
    with open(checkpoint_file, "rb") as f:
        models = pickle.load(f)

    # Test evaluation
    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, prospective)


def finetune_ensemble_xgb(xs_finetune, ys_finetune, xs_val, ys_val, xs_test, ys_test, xs_umm, ys_umm, df_cols,
                          shap_selected, interpret_model=False, testing=True):
    """
    Fine-tunes a saved XGBoost model on new data.

    Parameters:
    xs_finetune (list of np.array): List of feature arrays for fine-tuning.
    ys_finetune (list of np.array): List of target arrays for fine-tuning.
    xs_val (list of np.array): List of feature arrays for validation.
    ys_val (list of np.array): List of target arrays for validation.
    xs_test (list of np.array): List of feature arrays for testing.
    ys_test (list of np.array): List of target arrays for testing.
    xs_umm (list of np.array): List of feature arrays for prospective evaluation.
    ys_umm (list of np.array): List of target arrays for prospective evaluation.
    df_cols (list): List of column names used for feature interpretation.
    shap_selected (bool): Whether SHAP-selected features were used.
    interpret_model (bool): Whether to interpret the model using SHAP.
    testing (bool): Whether to evaluate the model after fine-tuning.
    """
    model_name = 'xgb_shap_selected' if shap_selected else 'xgb'

    # Load the model
    model_path = f'./models/{model_name}/model.pickle'
    with open(model_path, "rb") as f:
        models = pickle.load(f)

    # Fine-tune models
    for idx, (model, X_finetune, y_finetune, X_val, y_val) in enumerate(zip(models, xs_finetune, ys_finetune, xs_val, ys_val)):
        print(f'Fine-tuning model {idx}')
        model.fit(X_finetune, y_finetune, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True,
                  xgb_model=model)
        models[idx] = model

    # Save fine-tuned models
    finetuned_model_dir = f'./models/{model_name}_finetuned/'
    os.makedirs(finetuned_model_dir, exist_ok=True)
    finetuned_model_path = f'{finetuned_model_dir}/model_finetuned.pickle'

    with open(finetuned_model_path, "wb") as f:
        pickle.dump(models, f)

    # Optionally interpret models
    if interpret_model:
        interpret(xs_finetune[0], xs_test[0], df_cols=df_cols, model_name=model_name + '_finetuned')

    # Optionally evaluate models
    if testing:
        evaluate_ensemble_xgboost(xs_test, ys_test, df_cols, shap_selected, model_name=model_name + '_finetuned')


# the xgb model they used in the paper, standart xgb, no hyperparameters.
def train_xgb_model_from_paper(x_train, y_train, x_val, y_val):
    """
    XGBoost model using the same settings described in the referenced paper.
    PCA has been applied
    """

    model = xgb.XGBClassifier(
        n_estimators=100,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss'  
    )
    
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False 
    )

    return model


'''
#the xgb model from marcus

def hypertrain_xgb_model(x_train, y_train, x_val, y_val, early_stopping_rounds=10):
    """
    Trains a model on the provided features (x_train) and labels (y_train) with early stopping based on validation loss.

    Args:
        x_train (pandas.DataFrame or numpy.ndarray): The features used for training.
        y_train (pandas.Series or numpy.ndarray): The labels used for training.
        x_val (pandas.DataFrame or numpy.ndarray): The features used for validation.
        y_val (pandas.Series or numpy.ndarray): The labels used for validation.
        early_stopping_rounds (int): Number of rounds to wait for validation loss to improve before stopping.

    Returns:
        sklearn model: The trained model.
    """

    """Te2Rules Explainer of a single trained XGB"""

    # # ---------------------------------------------------------
    #
    # # model = xgb.XGBClassifier(tree_method='hist', max_depth=1, learning_rate=0.3775, subsample=0.3)
    # model = xgb.XGBClassifier(tree_method='hist', max_depth=2, learning_rate=0.01, subsample=0.3)
    # model.fit(x_train, y_train)
    #
    # accuracy = model.score(x_val, y_val)
    #
    # print("Accuracy")
    # print(accuracy)
    #
    # model_explainer = ModelExplainer(
    #     model=model,
    #     feature_names=['Thrombozyten', 'MCV', 'INR']
    # )
    #
    # rules = model_explainer.explain(
    #     X=x_train, y=y_train,
    #     num_stages=1,  # stages can be between 1 and max_depth
    #     min_precision=0.80,  # higher min_precision can result in rules with more terms overfit on training data
    #     jaccard_threshold=0.8  # lower jaccard_threshold speeds up the rule exploration, but can miss some good rules
    # )
    #
    # print(str(len(rules)) + " rules found:")
    # print()
    # for i in range(len(rules)):
    #     print("Rule " + str(i) + ": " + str(rules[i]))
    #
    # fidelity, positive_fidelity, negative_fidelity = model_explainer.get_fidelity()
    #
    # print("The rules explain " + str(round(fidelity * 100, 2)) + "% of the overall predictions of the model")
    # print("The rules explain " + str(round(positive_fidelity * 100, 2)) + "% of the positive predictions of the model")
    # print("The rules explain " + str(round(negative_fidelity * 100, 2)) + "% of the negative predictions of the model")
    #
    # breakpoint()
    # # ---------------------------------------------------------

    model = xgb.XGBClassifier(tree_method='hist', early_stopping_rounds=early_stopping_rounds, eval_metric='logloss')
    # original hp_space
    #hp_space = {
    #    'max_depth': np.arange(1, 40),
    #    'learning_rate': np.linspace(0.5, 0.01, 5),
    #    'subsample': np.linspace(1, 0.3, 5)
    #}
    
    hp_space = {
    'max_depth': np.arange(3, 21, 1),  # Lower max_depth to prevent overfitting
    'learning_rate': np.linspace(0.1, 0.01, 10),  # More fine-grained search for learning rate
    'subsample': np.linspace(0.6, 1.0, 5), 
    'colsample_bytree': np.linspace(0.6, 1.0, 5),  # Add colsample_bytree for additional regularization
    'n_estimators': np.arange(50, 201, 50)  # Add n_estimators to control tree count
    }

    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

    clf = RandomizedSearchCV(
        estimator=model,
        param_distributions=hp_space,
        scoring='neg_log_loss',
        cv=5,
        random_state=42,
    )

    clf.fit(x_train, y_train, sample_weight=classes_weights, eval_set=[(x_val, y_val)], verbose=0)

    print()
    print(f'max_depth: {clf.best_estimator_.max_depth}')
    print(f'learning_rate: {clf.best_estimator_.learning_rate}')
    print(f'subsample: {clf.best_estimator_.subsample}')
    print()

    return clf.best_estimator_
'''