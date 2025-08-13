import seaborn as sns
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc


def plot_cm(cm, name='xgb', iteration=None, prospective=False):
    """
    Plot the confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix array.
        name (str): Name of the model or context to be included in the saved plot filename.

    Returns:
        None
    """

    if not os.path.isdir(f'./outputs/{name}/'):
        os.mkdir(f'./outputs/{name}/')

    df_cm = pd.DataFrame(
            cm,
            index=[
                i for i in [
                    'Stage 0',
                    'Stage 1'
                ]],
            columns=[
                i for i in [
                    'Stage 0',
                    'Stage 1'
                ]])
    sns.heatmap(df_cm, annot=True, cmap='coolwarm')
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    if prospective:
        plt.savefig(f"outputs/{name}/prospective/{name}_cm_{iteration}.png")
    else:
        plt.savefig(f"outputs/{name}/{name}_cm_{iteration}.png")

    plt.close()


def plot_auc(y_true, y_pred_proba):
    """
    Plot the ROC curve and calculate the AUC.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred_proba : array-like of shape (n_samples, 2)
        Predicted probabilities of each class.
    """
    y_pred = np.array(y_pred_proba)[:, 1].tolist()  # Select the probabilities for the positive class

    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.fill_between(fpr, tpr, color='orange', alpha=0.3)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_max_correlations(features, corr_thresh=None):
    # Compute absolute correlation matrix
    corr_matrix = features.corr().abs()

    # Set self-correlations to 0 to exclude them
    np.fill_diagonal(corr_matrix.values, 0)

    # For each feature, find max absolute correlation to any other feature
    max_corr_per_feature = corr_matrix.max(axis=1)

    # Sort features by their max correlation
    sorted_max_corr = max_corr_per_feature.sort_values()

    plt.figure(figsize=(12, 10))
    plt.plot(sorted_max_corr.values, marker='o', linestyle='-', color='tab:blue')
    plt.xlabel('Features sorted by max correlation with others', fontsize=12)
    plt.ylabel('Max absolute correlation', fontsize=12)
    plt.title('Max absolute correlation per feature (sorted)', fontsize=14)

    # Add vertical lines and annotation for correlation thresholds
    plt.axhline(corr_thresh, color='red', linestyle='--', label=f'Threshold = {corr_thresh}')

    # Count how many features have max correlation < threshold
    num_selected = (sorted_max_corr < corr_thresh).sum()
    plt.axvline(num_selected, color='green', linestyle='--', label=f'Features below threshold: {num_selected}')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return sorted_max_corr


def plot_combined_roc_curve(y_true_list, y_pred_probas_list, model_name='xgb', prospective=False):
    """
    Plot ROC curves for each fold and a combined ROC curve over all folds.

    Args:
    - y_true_list: list of 1D arrays, each with true labels for a fold
    - y_pred_probas_list: list of 2D arrays, each with predicted probabilities shape (n_samples, 2)
    - model_name: string for saving output files
    - prospective: bool, whether to save in prospective folder

    Returns:
    - None, saves the ROC plot(s) and prints mean AUC
    """
    fpr_list = []
    tpr_list = []
    auc_list = []

    # Compute ROC and AUC for each fold
    for i, (y_true, y_pred_probas) in enumerate(zip(y_true_list, y_pred_probas_list)):
        fpr, tpr, _ = roc_curve(y_true, y_pred_probas[:, 1], pos_label=1)
        auc_val = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc_val)

    plt.figure(figsize=(10, 8))

    # Plot every 10th fold to reduce clutter
    for i in range(len(fpr_list)):
        if i % 10 == 0:
            plt.plot(fpr_list[i], tpr_list[i], label=f'Fold {i+1} (AUC = {auc_list[i]:.3f})', alpha=0.6)

    # Combined ROC from all predictions
    y_true_all = np.concatenate(y_true_list)
    y_probas_all = np.concatenate([probas[:, 1] for probas in y_pred_probas_list])
    fpr_all, tpr_all, _ = roc_curve(y_true_all, y_probas_all, pos_label=1)
    auc_all = auc(fpr_all, tpr_all)

    # Plot combined ROC
    plt.plot(fpr_all, tpr_all, color='black', linestyle='--',
             linewidth=2.5, label=f'Overall ROC (AUC = {auc_all:.3f})')

    # Diagonal random-guessing line
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray', linewidth=2)

    # Styling
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve per Fold and Overall', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save plot
    output_path = f'outputs/{model_name}/prospective/' if prospective else f'outputs/{model_name}/'
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f'{output_path}{model_name}_ensemble_ROCs_on_each_test_data.png', bbox_inches='tight')
    plt.close()



def plot_roc(tpr, fpr, roc_auc, model, name, iteration):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.title(f"{name} Comparison")
    plt.savefig(f"outputs/{model}/{model}_{name}_{iteration}.png")
    plt.close()


def plot_metric(metric, model, name):
    """
    Creates and saves a bar chart comparing the accuracies of two sets of methods (models).

    Args:
        metric (list): Metric to be plotted.
    Returns:
        None
    """

    x_positions = np.arange(len(metric))
    plt.figure(figsize=(16, 9), dpi=300)
    # Create the bar plot
    plt.bar(x_positions, metric, color='skyblue')
    plt.xticks(x_positions)

    # Customize and save the plot
    plt.xlabel("Imputed Data Set")
    plt.ylabel("Percentage")
    plt.title(f"{name} Comparison")
    plt.savefig(f"outputs/{model}/{model}_{name}.png")
    plt.close()


def plot_accuracies(reports, report_label, name, prospective):
    """
    Creates and saves a bar chart of accuracies from classification reports.

    Args:
        reports (list of dict): List of classification reports with 'accuracy' keys.
        report_label (str): Label to show in the legend.
        name (str): Base name used for plot title and output folder.
        prospective (bool): If True, saves to the prospective output folder.
    """
    accuracies = [report['accuracy'] for report in reports]
    x_positions = np.arange(len(accuracies))
    labels = [f"Imputed set {i+1}" for i in x_positions]

    plt.figure(figsize=(16, 9), dpi=300)
    plt.bar(x_positions, accuracies, label=report_label, color='skyblue', width=0.4)

    # Improve axis formatting
    plt.xlabel("Imputed Data Set")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"Accuracy Comparison – {name.replace('_', ' ').title()}")
    plt.xticks(x_positions, labels=labels, rotation=20, ha='right')
    plt.legend()

    # Ensure output directory exists
    out_dir = f"outputs/{name}/prospective/" if prospective else f"outputs/{name}/"
    os.makedirs(out_dir, exist_ok=True)

    plt.savefig(f"{out_dir}{name}_accuracies_comparison.png")
    plt.close()


def plot_diff_of_accuracies(
        reports1,
        report_1_label,
        reports2,
        report_2_label,
        name):
    """
    Creates and saves a bar chart comparing the difference of two accuracy lists.

    Args:
        reports1 (list): A list of reports from the first imputed data set.
        report_1_label (str): The label for the first set of reports in the plot.
        reports2 (list): A list of reports from the second imputed data set.
        report_2_label (str): The label for the second set of reports in the plot.
    """
    accuracies_1 = [report['accuracy'] for report in reports1]
    accuracies_2 = [report['accuracy'] for report in reports2]
    accuracy_diff = np.subtract(accuracies_1, accuracies_2)
    x_positions = np.arange(len(accuracy_diff))

    plt.figure(figsize=(16, 9), dpi=300)
    # Create the bar plot
    plt.bar(
        x_positions,
        accuracy_diff,
        label='Accuracy Difference',
        color='skyblue',
        width=0.4)

    # Customize and save the plot
    plt.xlabel("Imputed Data Set")
    plt.ylabel("Accuracy")
    plt.title(
        f"Accuracy Difference ({report_1_label} - {report_2_label}) {' '.join(word.capitalize() for word in name.split('_'))}")
    plt.xticks(x_positions, rotation=45, ha='right')
    plt.legend()
    plt.savefig(f"plots/{name}/{name}_accuracies_difference.png")
    plt.close()


def evaluate_model(logits_samples, mean_logits, y_test_tensor):
    """
    :param logits_samples:
    :param mean_logits:
    :param y_test_tensor:
    :param label_encoder:
    :return:
    """
    # calculate the mean and std
    mean_probs = logits_samples.mean(axis=0)
    std_probs = logits_samples.std(axis=0)
    confidence_interval = 1.96 * std_probs
    lower_bound = mean_probs - confidence_interval
    upper_bound = mean_probs + confidence_interval
    lower_bound.clamp_(min=0, max=1)
    upper_bound.clamp_(min=0, max=1)

    num_classes = mean_logits.shape[1]
    num_samples_to_plot = 200
    ground_truth = y_test_tensor[:num_samples_to_plot].cpu().numpy()
    fig, axes = plt.subplots(num_classes, 1, figsize=(10, 5 * num_classes))

    for i in range(num_classes):
        mean = mean_logits[:num_samples_to_plot, i]
        lower = lower_bound[:num_samples_to_plot, i]
        upper = upper_bound[:num_samples_to_plot, i]
        if torch.is_tensor(lower):
            lower = lower.cpu().numpy()
        if torch.is_tensor(upper):
            upper = upper.cpu().numpy()
        corrected_lower = np.maximum(lower, 0)
        corrected_upper = np.minimum(upper, 1)
        ci_lower = mean - corrected_lower
        ci_upper = corrected_upper - mean
        axes[i].errorbar(
            range(num_samples_to_plot),
            mean,
            yerr=[
                ci_lower,
                ci_upper],
            fmt='o',
            color='blue',
            ecolor='red',
            alpha=0.7,
            label='Prediction with 95% CI')
        for j, gt in enumerate(ground_truth):
            if gt == i:
                axes[i].axvline(
                    j,
                    color='green',
                    alpha=0.5,
                    linestyle='--',
                    linewidth=0.5)

        axes[i].set_title(f'{i}')
        axes[i].legend()
    plt.tight_layout()
    plt.show()
