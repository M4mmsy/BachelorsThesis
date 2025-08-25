from src.utils.helper_functions import *
from src.utils.validation_tools import evaluate_performance, interpret
from src.preprocess import preprocess_nested
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, RandomizedSearchCV
from src.preprocess import preprocess_nested
from sklearn.metrics import make_scorer, fbeta_score
from src.utils.analysis import FeatureSelector
from tqdm import tqdm
import json
from sklearn.utils import class_weight
from sklearn.svm import SVC
from sklearn.model_selection import PredefinedSplit

def hypertrain_nested_svm(X, y, df_cols, shap_selected, n_repeats, scaling=False, imputing=False, corr_thresh=0.85, target_corr_thresh=0.1):
    model_name = 'svm'
    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

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

        # Valid debug-fast hyperparameter space
        hp_space = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],  # only used if kernel == 'rbf'
            'class_weight': [None, 'balanced']
        }

        clf = RandomizedSearchCV(
            estimator=SVC(probability=True, random_state=42),
            param_distributions=hp_space,
            cv=inner_cv,
            scoring=make_scorer(fbeta_score, beta=2),
            n_iter=10,  # fast debugging
            random_state=42,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        # Save CV results
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

def hypertrain_ensemble_svm(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                            shap_selected, interpret_model=True, testing=True):
    models = []

    # Directory setup
    model_name = 'svm_shap_selected' if shap_selected else 'svm'

    os.makedirs(f'./models/{model_name}', exist_ok=True)
    os.makedirs(f'./outputs/{model_name}', exist_ok=True)

    for idx, (X_train, y_train, X_val, y_val) in enumerate(zip(xs_train, ys_train, xs_val, ys_val)):
        print(f'Training model {idx}')
        models.append(hypertrain_svm_model(X_train, y_train, X_val, y_val))

    model_path = f'models/{model_name}/model.pickle'
    with open(model_path, "wb") as f:
        pickle.dump(models, f)

    print('------ Finished Training Ensemble ------')

    if interpret_model:
        interpret(xs_train[0], xs_test[0], df_cols=model_name)

    if testing:
        # Optionally test immediately after training
        evaluate_ensemble_svm(xs_test, ys_test, df_cols, shap_selected)


def evaluate_ensemble_svm(xs_test, ys_test, df_cols, shap_selected,
                          model_name='svm'):
    model_name = f'{model_name}_shap_selected' if shap_selected else model_name

    checkpoint_file = [f"./models/{model_name}/{f}" for f in os.listdir(f"./models/{model_name}/") if f.endswith('.pickle')
                       ][0]

    # Load the models from the saved pickle file
    with open(checkpoint_file, "rb") as f:
        models = pickle.load(f)

    prospective = False
    print('----- Test Evaluation ------')
    evaluate_performance(models, xs_test, ys_test, df_cols, model_name, prospective)

#
# def interpret_svm(x_train, x_test, df_cols):
#     """
#     Generate SHAP (SHapley Additive exPlanations) plots to interpret model predictions.
#
#     Parameters:
#         x_train (array-like): Training data features.
#         x_test (array-like): Test data features.
#         df_cols (list): List of column names (features).
#
#     Returns:
#         None
#     """
#     explainer = shap.KernelExplainer(model=lambda x: predict_svm_model(x), data=shap.sample(x_train, 50), feature_names=df_cols)
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
#     f.savefig(f'outputs/svm/summary_bar.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#     fig, ax = plt.subplots()
#
#     shap.summary_plot(shap_values, x_test, plot_type='violin', feature_names=df_cols, show=False)
#
#     f = plt.gcf()
#     f.savefig(f'outputs/svm/beeswarm.png', bbox_inches='tight', dpi=300)
#     plt.close()
#
#
# def predict_svm_model(data):
#     """
#     Predictions of the model for certain data. Model is saved in output/models.pickle
#
#     Args:
#         data: A numpy array to predict on.
#
#     Returns:
#         A numpy array of class predictions
#     """
#
#     with open(f'models/svm/model.pickle', "rb") as f:
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


def hypertrain_svm_model(x_train, y_train, x_val, y_val):
    """
    Trains a model on the provided features (x_train) and labels (y_train).

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

    model = SVC(random_state=42, probability=True)

    hp_space = {
        'degree': np.arange(2, 5),  # Degree of polynomial kernel (if applicable)
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

    return clf.best_estimator_
