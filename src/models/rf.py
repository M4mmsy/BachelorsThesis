from src.utils.helper_functions import *
from src.preprocess import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils import class_weight
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from src.utils.analysis import FeatureSelector
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pickle
import numpy as np
import joblib
from tqdm import tqdm
import pandas as pd
import json

# best hp space right now:
'''
    hp_space = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 4, 5, 10],
        'min_samples_leaf': [2, 4, 5, 10],
        'max_features': ['sqrt', 'log2', 0.3],
        'max_samples': [0.7, 0.8, 0.9, None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'class_weight': [
            'balanced', 
            None,
            {0: 1, 1: 2},
            {0: 1, 1: 3}
        ]
    }


    hp_space = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [10, 20, 25],
        'min_samples_leaf': [10, 20, 25, 30],
        'max_features': ['sqrt', 0.1, 0.2],
        'max_samples': [None],
        'bootstrap': [False],
        'criterion': ['gini'],
        'class_weight': ['balanced', None, {0:1, 1:2}]
    }

    HP SPACE FOR FAST TESTING

    hp_space = {
        'n_estimators': [50, 100],   
        'max_depth': [2, 4, 6, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': [None, 'balanced']
    }
    
    '''




def hypertrain_nested_rf(X, y, df_cols, shap_selected, n_repeats, scaling=False, imputing=False, corr_thresh=0.85, target_corr_thresh=0.1):
    model_name = 'rf'
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    rf = RandomForestClassifier(random_state=42)
    
    
    hp_space = {
        'n_estimators': [50, 75, 100, 150, 200],     # try larger ensembles
        'max_depth': [2, 3, 4],                # slightly deeper trees
        'min_samples_split': [3, 4, 5, 6],           # allow small splits but not too small
        'min_samples_leaf': [1, 2, 3, 4],            # allow smaller leaves
        'max_features': ['sqrt', 0.3, 0.5],          # try more feature fractions
        'bootstrap': [True]                  
    }
    

    all_hp_results = []
    models = []
    xs_test_list = []
    ys_test_list = []

    outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=n_repeats, random_state=42)
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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

        # 3. Hyperparameter tuning with inner CV on reduced features
        clf = RandomizedSearchCV(
            estimator=rf,
            param_distributions=hp_space,
            cv=inner_cv,
            scoring='roc_auc',
            n_iter=65,
            random_state=42,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        # Save CV results etc.
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

'''
def hypertrain_ensemble_rf(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                           shap_selected, interpret_model=True, testing=True):
    models = []

    model_name = 'rf_shap_selected' if shap_selected else 'rf'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    # Train an ensemble — loop over all imputations / datasets
    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training ensemble model {idx+1} / {len(xs_train)}')
        model = hypertrain_rf_model(X_train, y_train, X_val, y_val)
        print_rf_model_info(model)
        models.append(model)

    print('------ Finished Training Ensemble ------')

    # Save all models in the ensemble
    model_path = f'models/{model_name}/model.pickle'
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    # Interpret the first model in the ensemble if requested
    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols=df_cols, model_name=model_name)

    # Evaluate ensemble performance immediately after training if testing=True
    if testing:
        evaluate_ensemble_rf(xs_test, ys_test, df_cols, shap_selected,
                             model_name=model_name)

    return models
'''
'''
def hypertrain_ensemble_rf(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                           shap_selected, interpret_model=True, testing=True):
    models = []

    model_name = 'rf_shap_selected' if shap_selected else 'rf'

    # Create directories if they don't exist
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training model {idx}')
        models.append(hypertrain_rf_model(X_train, y_train, X_val, y_val))

    print('------ Finished Training Ensemble ------')

    model_path = f'models/{model_name}/model.pickle'
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols=df_cols, model_name=model_name)

    if testing:
        # Optionally test immediately after training
        evaluate_ensemble_rf(xs_test, ys_test, df_cols, shap_selected,
                             model_name=model_name)
        '''
        
def hypertrain_nested_cv(xs_train, ys_train, xs_val, ys_val, df_cols,
                         shap_selected, interpret_model=True, testing=True):
    """
    Trains 500x10-fold nested CV Random Forest models as per the paper method.

    Args:
        xs_train (list of np.ndarray): List of training feature sets for each fold/repeat.
        ys_train (list of np.ndarray): List of training labels.
        xs_val (list of np.ndarray): List of validation feature sets.
        ys_val (list of np.ndarray): List of validation labels.
        df_cols (list): Feature column names.
        shap_selected (bool): Whether SHAP-based feature selection was used.
        interpret_model (bool): Whether to interpret the model after training.
        testing (bool): Ignored in nested CV (kept for interface consistency).
    """
    model_name = 'rf_shap_selected' if shap_selected else 'rf'
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    models = []

    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training nested CV model {idx + 1}/{len(xs_train)}')
        model = hypertrain_rf_model(X_train, y_train, X_val, y_val)
        print_rf_model_info(model)
        models.append(model)

    # Save all 500×10 models
    model_path = f'models/{model_name}/nested_models.pickle'
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    print('------ Finished 500×10-Fold Nested CV Training ------')

    # Optionally: interpret one representative model (e.g., from first repetition)
    if interpret_model:
        interpret_nested_cv(xs_train[0], xs_val[0], df_cols=df_cols, model=models[0], model_name=model_name)

    # Note: `testing` is ignored here; no test set used in paper method
  
def print_rf_model_info(rf_model):
    print("\nRandom Forest Model Info:")
    print(f"Number of trees (n_estimators): {rf_model.n_estimators}")
    print(f"Max depth (max_depth): {rf_model.max_depth}")
    print(f"Min samples split (min_samples_split): {rf_model.min_samples_split}")
    print(f"Min samples leaf (min_samples_leaf): {rf_model.min_samples_leaf}")
    print(f"Bootstrap samples (bootstrap): {rf_model.bootstrap}")
    print(f"Random state: {rf_model.random_state}")

def evaluate_ensemble_rf(xs_test, ys_test, df_cols, shap_selected,
                         model_name='rf'):
    if 'shap_selected' not in model_name and shap_selected:
        model_name = f'{model_name}_shap_selected'

    checkpoint_file = [f"./models/{model_name}/{f}" for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pickle')][0]

    # Load models
    with open(checkpoint_file, "rb") as f:
        models = pickle.load(f)

    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, prospective)

    # -----------------------------------
    # Step 1: Collect ensemble probabilities
    # -----------------------------------
    from sklearn.metrics import precision_score, recall_score, f1_score

    all_probs = []
    all_labels = []
    
    for model, X, y in zip(models, xs_test, ys_test):
        prob = model.predict_proba(X)[:, 1]
        all_probs.extend(prob)
        all_labels.extend(y)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # -----------------------------------
    # Step 2: ROC Curve + AUC
    # -----------------------------------
    auc_score = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve for Ensemble RF Model")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_roc_curve.png")
    plt.show()

    # -----------------------------------
    # Step 3: Threshold tuning example
    # -----------------------------------
    best_f1 = 0
    best_thresh = 0.5
    for t in np.linspace(0.1, 0.9, 50):
        preds = (all_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    print(f"\nBest threshold for F1: {best_thresh:.2f} (F1 = {best_f1:.3f})")

    # Evaluate at optimal threshold
    tuned_preds = (all_probs >= best_thresh).astype(int)
    prec = precision_score(all_labels, tuned_preds)
    rec = recall_score(all_labels, tuned_preds)
    f1 = f1_score(all_labels, tuned_preds)

    print(f"\n[Threshold-tuned performance @ {best_thresh:.2f}]")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"AUC:       {auc_score:.3f}")

''' ORIGINAL FROM MARCUS
def evaluate_ensemble_rf(xs_test, ys_test, df_cols, shap_selected,
                         model_name='rf'):
    if 'shap_selected' not in model_name and shap_selected:
        model_name = f'{model_name}_shap_selected'

    checkpoint_file = [f"./models/{model_name}/{f}" for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pickle')
                       ][0]

    # Load the models from the saved pickle file
    with open(checkpoint_file, "rb") as f:
        models = pickle.load(f)

    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, prospective)
'''


def finetune_ensemble_rf(xs_finetune, ys_finetune, xs_val, ys_val, xs_test, ys_test, xs_pro, ys_pro, df_cols,
                         shap_selected, interpret_model=True, testing=True):
    """
    Fine-tunes a saved ensemble of Random Forest models on new data.

    Parameters:
    xs_finetune (list of np.array): List of feature arrays for fine-tuning.
    ys_finetune (list of np.array): List of target arrays for fine-tuning.
    xs_val (list of np.array): List of feature arrays for validation.
    ys_val (list of np.array): List of target arrays for validation.
    xs_test (list of np.array): List of feature arrays for testing.
    ys_test (list of np.array): List of target arrays for testing.
    xs_pro (list of np.array): List of feature arrays for prospective evaluation.
    ys_pro (list of np.array): List of target arrays for prospective evaluation.
    df_cols (list): List of column names used for feature interpretation.
    shap_selected (bool): Whether SHAP-selected features were used.
    interpret_model (bool): Whether to interpret the model using SHAP.
    testing (bool): Whether to evaluate the model after fine-tuning.
    """
    model_name = 'rf_shap_selected' if shap_selected else 'rf'

    # Load the model
    model_path = f'./models/{model_name}/model.pickle'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        models = pickle.load(f)

    # Ensure models is a list of RandomForest instances
    if not all(isinstance(model, RandomForestClassifier) for model in models):
        raise TypeError("Loaded models are not of type RandomForestClassifier or RandomForestRegressor")

    # Fine-tune models
    for idx, (model, X_finetune, y_finetune) in enumerate(zip(models, xs_finetune, ys_finetune)):
        print(f'Fine-tuning model {idx}')
        model.fit(X_finetune, y_finetune)
        models[idx] = model

    print('------ Finished Fine-Tuning Ensemble ------')

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
        evaluate_ensemble_rf(xs_test, ys_test, xs_pro, ys_pro, df_cols, shap_selected, model_name=model_name + '_finetuned')


# def interpret_rf(x_train, x_test, df_cols,  model_name='rf'):
#     """
#     Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.
#
#     Parameters:
#         x_train (array-like): Training data features.
#         x_test (array-like): Test data features.
#         df_cols (list): List of column names (features).
#         model_name (str): name of output model
#
#     Returns:
#         None
#     """
#     explainer = shap.KernelExplainer(model=lambda x: predict_rf_model(x, 
#                                                                       model_name=model_name),
#                                      data=shap.sample(x_train, 50), feature_names=df_cols)
#
#     shap_values = explainer.shap_values(x_test)
#
#     f = shap.force_plot(
#         explainer.expected_value,
#         shap_values,
#         x_test,
#         feature_names=df_cols,
#         show=False)
#     shap.save_html(f'outputs/xgb/force_plot.htm', f)
#     plt.close()
#
#     fig, ax = plt.subplots()
#     shap_values2 = explainer(x_test)
#     shap.plots.bar(shap_values2, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/{model_name}/summary_bar.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#     fig, ax = plt.subplots()
#
#     shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/{model_name}/beeswarm.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#
# def predict_rf_model(data,  model_name='rf'):
#     """
#     Predictions of the model for certain data. Model is saved in output/models.pickle
#
#     Args:
#         data: A numpy array to predict on.
#         model_name (str): name of output model
#
#     Returns:
#         A numpy array of class predictions
#     """
#
#     with open(f'models/{model_name}/model.pickle', "rb") as f:
#         models = pickle.load(f)
#
#     y_preds = []
#     for model in models:
#         y_pred = model.predict_proba(data)
#         y_preds.append(y_pred)
#
#     maj_preds = majority_vote(y_preds, rule='soft')
#     indices, _ = get_index_and_proba(maj_preds)
#
#     return np.array(indices)

'''
def hypertrain_rf_model(x_train, y_train, x_val, y_val):
    """
    Trains a RandomForestClassifier using a PredefinedSplit for nested CV folds.

    Args:
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        x_val (np.ndarray): Validation features.
        y_val (np.ndarray): Validation labels.

    Returns:
        sklearn.base.BaseEstimator: Best trained RandomForest model for the fold.
    """
    # Combine train and val sets for PredefinedSplit
    X = np.concatenate((x_train, x_val))
    y = np.concatenate((y_train, y_val))

    split_index = [-1] * len(x_train) + [0] * len(x_val)
    pds = PredefinedSplit(test_fold=split_index)

    model = RandomForestClassifier()

    hp_space = {
        'n_estimators': np.arange(10, 101, 10),
        'max_depth': [None] + list(np.arange(2, 11)),
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
    }

    sample_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)

    clf = RandomizedSearchCV(
        estimator=model,
        param_distributions=hp_space,
        scoring='neg_log_loss',
        cv=pds,
        n_iter=20,  # Reasonable default for speed; increase if needed
        random_state=42,
        n_jobs=-1  # Speed up if multiple cores are available
    )

    clf.fit(X, y, sample_weight=sample_weights)

    return clf.best_estimator_
'''

def hypertrain_rf_model(x_train, y_train, x_val, y_val):
    """
    Trains a Random Forest model with hyperparameter tuning using a predefined train/val split.

    Args:
        x_train (np.ndarray or pd.DataFrame): Training features.
        y_train (np.ndarray or pd.Series): Training labels.
        x_val (np.ndarray or pd.DataFrame): Validation features.
        y_val (np.ndarray or pd.Series): Validation labels.

    Returns:
        sklearn RandomForestClassifier: The best estimator found by RandomizedSearchCV.
    """

    # Concatenate train and validation data for PredefinedSplit
    X = np.concatenate((x_train, x_val))
    y = np.concatenate((y_train, y_val))

    # Create PredefinedSplit: train indices = -1, val indices = 0
    split_index = [-1] * len(x_train) + [0] * len(x_val)
    pds = PredefinedSplit(test_fold=split_index)

    # Define hyperparameter search space
    hp_space = {
        'n_estimators': np.arange(50, 301, 5),
        'max_depth': [8, 9, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None, 0.5, 0.75],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # Compute sample weights to handle class imbalance
    sample_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)

    # Initialize Random Forest classifier
    rf = RandomForestClassifier(random_state=42)

    # Setup RandomizedSearchCV with PredefinedSplit and negative log loss scoring
    clf = RandomizedSearchCV(
        estimator=rf,
        param_distributions=hp_space,
        scoring='neg_log_loss',
        cv= RepeatedStratifiedKFold(n_splits=10, n_repeats=50),
        n_iter=100,             # number of parameter settings that are sampled
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    # Fit the model with sample weights
    clf.fit(X, y, sample_weight=sample_weights)

    print(f"Best RF parameters: {clf.best_params_}")

    # Return the best estimator found
    return clf.best_estimator_

'''def hypertrain_rf_model(x_train, y_train, x_val, y_val):
    """
    Trains a model on the provided features (x_train) and labels (y_train)

    Args:
        x_train (pandas.DataFrame or numpy.ndarray): The features used for training.
        y_train (pandas.Series or numpy.ndarray): The labels used for training.
        x_val (pandas.DataFrame or numpy.ndarray): The features used for validation.
        y_val (pandas.Series or numpy.ndarray): The labels used for validation.

    Returns:
        sklearn model: The trained model.
    """
    # Concatenate x_train and x_val to X to use for PredefinedSplit. Reason: RF does not take eval_set as input
    X = np.concatenate((x_train, x_val))
    y = np.concatenate((y_train, y_val))

    # Create a list where train data indices are -1 and validation data indices are 0
    split_index = [-1] * x_train.shape[0]
    val_index_list = [0] * x_val.shape[0]
    split_index.extend(val_index_list)

    # Use the list to create PredefinedSplit
    pds = PredefinedSplit(test_fold=split_index)

    model = RandomForestClassifier()

    # Define hyperparameter space for RandomForestClassifier
    hp_space = {
        'n_estimators': np.arange(10, 101, 10),  # Number of trees in the forest
        'max_depth': [None] + list(np.arange(2, 11)),  # Maximum depth of the trees
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
    }

    classes_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)

    clf = RandomizedSearchCV(
        estimator=model,
        param_distributions=hp_space,
        scoring='neg_log_loss',
        cv=pds,
        random_state=42,
    )

    clf.fit(X, y, sample_weight=classes_weights)

    # print()
    # print(f'max_depth: {clf.best_estimator_.max_depth}')
    # print(f'n_estimators: {clf.best_estimator_.n_estimators}')
    # print(f'min_samples_leaf: {clf.best_estimator_.min_samples_leaf}')
    # print()

    return clf.best_estimator_
'''