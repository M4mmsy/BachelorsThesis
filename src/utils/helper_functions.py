from sklearn.metrics import classification_report, confusion_matrix
import pickle
from collections import Counter
from src.utils.plots import *
from sklearn.model_selection import StratifiedKFold


def test_sklearn_models(models, xs_test, ys_test):
    """
    Trains m models individually on m data sets
    Args:
        models (list): List of XBoost.classifier.
        xs_test (list): list of Test matrices.
        ys_test (list): List of Test labels.

    Returns:
        list: A list containing dictionaries of classifiction reports.
        list: A list of m confusion matrices.
    """

    reports = []
    cms = []
    y_preds = []
    for (model, X, y) in zip(models, xs_test, ys_test):
        y_pred = model.predict(X)
        report, cm = test(y, y_pred)
        reports.append(report)
        cms.append(cm)
        y_preds.append(y_pred)

    return reports, cms


def test(y, y_pred):
    """
    Evaluates a classification prediction.

    Args:
        y (pandas.Series or numpy.ndarray): The true labels.
        y_pred (pandas.Series or numpy.ndarray): The predicted labels.

    Returns:
        dict: A dictionary containing the classification report.
        numpy.ndarray: The confusion matrix.
    """
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    return report, cm


def make_individual_preds(xs_test, ys_test, models, model_name='xgb'):

    cms = []
    reports = []
    y_preds = []  # stores the index of predicted class
    probas = []  # stores the probability of the predicted class
    pred_probas = []  # stores the probability of both classes

    for (X, y, model) in zip(xs_test, ys_test, models):
        if model_name == 'xgb' or model_name == 'rf' or model_name == 'svm':
            probas = model.predict_proba(X)
            preds = model.predict(X)
        else:
            probas = model(torch.tensor(X, dtype=torch.float32, device=get_device(i=0)))
        y_preds.append(preds)

        maj_report, cm = test(y=y, y_pred=y_preds)

        reports.append(maj_report)
        cms.append(cm)
        preds.append(y_preds)
        probas.append(probas)

    return reports, cms, preds, probas, pred_probas

def make_ensemble_preds(xs_test, ys_test, models, intra_model_preds=False, threshold=None):
    """
    Evaluates multiple models on test sets from outer CV.

    Args:
        xs_test (list): List of X_test arrays (one per outer fold).
        ys_test (list): List of y_test arrays (one per outer fold).
        models (list): List of trained models (one per outer fold).
        intra_model_preds (bool): If True, evaluates all models on the first test set only.
        threshold (float or None): Optional threshold for converting probabilities to binary labels.

    Returns:
        ensemble_reports (list): Classification reports (including ROC AUC if binary).
        ensemble_cms (list): Confusion matrices.
        ensemble_preds (list): Predicted class indices per fold.
        ensemble_probas (list): Predicted class 1 probabilities per fold.
        ensemble_pred_probas (list): Full class probability arrays per fold.
    """
    if intra_model_preds:
        xs_test = [xs_test[0]] * len(models)
        ys_test = [ys_test[0]] * len(models)

    ensemble_reports = []
    ensemble_cms = []
    ensemble_preds = []
    ensemble_probas = []
    ensemble_pred_probas = []

    for X_test, y_test, model in zip(xs_test, ys_test, models):
        probas = model.predict_proba(X_test)
        ensemble_pred_probas.append(probas)

        proba_class_1 = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
        ensemble_probas.append(proba_class_1)

        if threshold is not None and probas.shape[1] == 2:
            y_pred = (proba_class_1 >= threshold).astype(int)
        else:
            y_pred = np.argmax(probas, axis=1)

        ensemble_preds.append(y_pred)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Add ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                auc = roc_auc_score(y_test, proba_class_1)
                report['mean_auc'] = {'roc_auc': auc}
            except ValueError:
                report['mean_auc'] = {'roc_auc': np.nan}  # fallback in rare edge cases

        cm = confusion_matrix(y_test, y_pred)

        ensemble_reports.append(report)
        ensemble_cms.append(cm)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas


# Markus sein make ensemble, brauche ein eigenes weil ich nested loops habe und jedes einzeln predicted werden muss
'''
def make_ensemble_preds(xs_test, ys_test, models, intra_model_preds=False):
    """
    Trains m models individually on m data sets
    Args:
        xs_test (list): List of Test matrices.
        ys_test (list): List of Test labels.
        models (list): e.g. m xgboost.XGBClassifier.
        intra_model_preds (bool): If True, then redefine the ensemble preds to intra-model preds on xs[0] instead of
        majority-model vote for intra-dataset preds as xs[0] == xs[1] == xs[2] (set True for shap_selected)
    Returns:
        list: A list containing dictionaries of classification reports.
        list: A list of m confusion matrices.
        list: A list of predictions of m data sets.
        list: A list of probabilities of m data sets.
    """

    ensemble_cms = []
    ensemble_reports = []
    ensemble_preds = []  # stores the index of predicted class
    ensemble_probas = []  # stores the probability of the predicted class
    ensemble_pred_probas = []  # stores the probability of both classes

    # Take first imputed dataset
    X_all, y_all = xs_test[0], ys_test[0]
    # Align test set columns to model's expected features
    if hasattr(models[0], "n_features_in_") and isinstance(X_all, pd.DataFrame):
        shap_feature_names = X_all.columns[:models[0].n_features_in_]
        X_all = X_all[shap_feature_names]
    # Count samples per class
    class_counts = Counter(y_all)
    min_class_size = min(class_counts.values())

    # Choose the number of splits based on smallest class
    n_splits = min(5, len(y_all), min_class_size)

    if n_splits < 2:
        print(f"[WARN] Skipping cross-validation: not enough samples per class (min_class_size={min_class_size})")
        return [], [], [], [], []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Perform cross-validation
    # Use for stratified_split the train_index (length is 80% of all samples) to calculate the CMs
    for split_index, _ in skf.split(X_all, y_all):
        X = X_all[split_index]
        y = y_all[split_index]
        
        y_preds = []
        y_preds_all = []

        for model in models:
            probas = model.predict_proba(X)
            probas_all = model.predict_proba(X_all)

            # Convert to list of lists containing class proababilities
            y_preds.append(probas)
            y_preds_all.append(probas_all)

            """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
            instead ensemble prediction on each xs_test"""
            if intra_model_preds:
                # ensemble_pred is now model probability instead of majority vote of models
                ensemble_pred = probas.tolist()
                ensemble_pred_probas.append(probas)

                ensemble_pred, probas = get_index_and_proba(ensemble_pred)
                maj_report, cm = test(y=y, y_pred=ensemble_pred)

                ensemble_reports.append(maj_report)
                ensemble_cms.append(cm)
                ensemble_preds.append(ensemble_pred)
                ensemble_probas.append(probas)

        """If intra_model_preds, then redefine the ensemble reports to each model prediction on the first xs_test 
        instead ensemble prediction on each xs_test"""
        if intra_model_preds:
            return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas

        """get ensemble_preds_probas on stratisfied preds for ROC curve and test"""
        # Take majority vote or soft voting
        ensemble_pred = majority_vote(y_preds, rule='soft')
        ensemble_pred, probas = get_index_and_proba(ensemble_pred)
        # Convert predicted labels to numpy array
        ensemble_pred = np.array(ensemble_pred)

        """get ensemble_preds_probas and ensemble_preds on all preds for ROC curve and test"""
        ensemble_pred_all_probas = majority_vote(y_preds_all, rule='soft')

        # Then convert to np.array
        ensemble_pred_all_probas = np.array(ensemble_pred_all_probas)
        ensemble_pred_probas.append(ensemble_pred_all_probas)
        ensemble_pred_all, probas_all = get_index_and_proba(ensemble_pred_all_probas)
        # Convert predicted labels to numpy array
        ensemble_pred_all = np.array(ensemble_pred_all)

        # Calculate classification report and confusion matrix
        maj_report = classification_report(y, ensemble_pred, output_dict=True)
        cm = confusion_matrix(y, ensemble_pred)

        ensemble_reports.append(maj_report)
        ensemble_cms.append(cm)
        ensemble_preds.append(ensemble_pred_all)
        ensemble_probas.append(probas_all)

    return ensemble_reports, ensemble_cms, ensemble_preds, ensemble_probas, ensemble_pred_probas
'''

def get_device(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def predict_pickle_model(data, model_name='xgb'):
    """
    Predictions of the model for certain data. Model is saved in output/models.pickle

    Args:
        data: A numpy array to predict on.
    Returns:
        A numpy array of class predictions
    """

    with open(f'models/{model_name}/model.pickle', "rb") as f:
        models = pickle.load(f)

    y_preds = []
    for model in models:
        y_pred = model.predict_proba(data)
        y_preds.append(y_pred)

    maj_preds = majority_vote(y_preds, rule='soft')
    indices, _ = get_index_and_proba(maj_preds)

    return np.array(indices)


def majority_vote(predictions, rule="hard"):
    """
    Performs majority voting on a list of predictions.

    Args:
        predictions: A list of lists, where each inner list contains predictions (classes) from one model.
        rule: "hard" for hard voting, "soft" for soft voting (default).

    Returns:
        A list of size equal to the number of inner lists, where each element is the majority class for the corresponding prediction across all models.
    """
    majority_classes = []
    for i in range(
            len(predictions[0])):

        if rule == 'hard':
            class_counts = Counter([prediction[i]
                                   for prediction in predictions])
            majority_class = class_counts.most_common(1)[0][0]
        elif rule == 'soft':
            majority_class = [0] * len(predictions[0][0])

            for model_predictions in predictions:
                majority_class = [
                    a + b for a,
                    b in zip(
                        majority_class,
                        model_predictions[i])]

            if sum(majority_class) > 0:
                majority_class = [p / sum(majority_class)
                                  for p in majority_class]

        else:
            raise ValueError("Invalid rule. Choose 'hard' or 'soft'.")
        majority_classes.append(majority_class)

    return majority_classes


def get_index_and_proba(data):
    """
    Finds the index and value of the highest element in each sublist.

    Args:
        data: A list of lists, where each inner list contains numerical values.

    Returns:
        A tuple containing two lists:
            - indices: A list containing the index of the highest element in each sublist.
            - values: A list containing the corresponding highest elements.
    """
    indices = []
    values = []

    for _, sublist in enumerate(data):
        # Find the index of the maximum value
        max_index = np.argmax(sublist)
        # Append the index and corresponding value
        indices.append(max_index)
        values.append(max(sublist))

    return indices, values


def remove_files_in_directory(directory):
    # empty all files in the mcmc_bnn
    if not len(os.listdir(directory)) == 0:
        print(f'deleting files in {directory}')
        files = os.listdir(directory)

        # Iterate over each file and remove it
        for file in files:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    else:
        print(f'{directory} already empty')
