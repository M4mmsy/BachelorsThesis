import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from src.utils.plots import plot_feature_max_correlations
import argparse
import os

class FeatureSelector:
    def __init__(self, corr_thresh=0.85, target_corr_thresh=0.15):
        """
        Parameters:
        - corr_thresh: correlation threshold for greedy filtering between features
            - 0.85 reduces about half of the features
        - target_corr_thresh: minimal absolute correlation with target to keep feature
            - 0.1 reduces the 96 features to about 49 - 91.
            - 0.15 reduces the 96 features to about 40- 80
            - 0.2 reduces the 96 features to about 20-50
            - implemented a dynamic threshhold 
        """
        self.corr_thresh = corr_thresh
        self.target_corr_thresh = target_corr_thresh
        self.selected_features = None

    @staticmethod
    def greedy_feature_selection(corr_matrix, y_corr, threshold):
        selected = []
        # Sort features by absolute correlation with target descending
        features_sorted = y_corr.abs().sort_values(ascending=False).index.tolist()

        for feature in features_sorted:
            if all(abs(corr_matrix.loc[feature, sel]) < threshold for sel in selected):
                selected.append(feature)
        return selected

    def fit(self, X, y):
        # 1. Compute feature-to-target correlations (point biserial correlation)
        y_corr = pd.Series(index=X.columns, dtype=float)
        for col in X.columns:
            corr, _ = pointbiserialr(y, X[col])
            y_corr[col] = corr

        # 2. Filter features weakly correlated with target, fallback if too few features
        thresh = self.target_corr_thresh
        strong_features = y_corr[abs(y_corr) >= thresh].index
        min_features = 30  # minimum acceptable features

        # reduces threshold by 0.01 until more than min_features are present.
        while len(strong_features) < min_features and thresh > 0:
            thresh -= 0.005
            strong_features = y_corr[abs(y_corr) >= thresh].index

        if len(strong_features) < min_features:
            print(f"Warning: Only {len(strong_features)} features left after univariate filtering, threshold lowered to {thresh:.3f}")

        X_filtered = X[strong_features]

        print(f"Features after univariate filtering (|corr| >= {thresh:.3f}): {len(strong_features)}")

        # 3. Compute correlation matrix on filtered features
        corr_matrix = X_filtered.corr()

        min_features = 11  # Mindestanzahl Features, die ausgewählt werden sollen
        threshold = self.corr_thresh  # Startwert, z.B. 0.7 oder 0.85
        max_threshold = 0.95  # obere Grenze, um nicht zu viele Features reinzunehmen

        selected = self.greedy_feature_selection(corr_matrix, y_corr.loc[strong_features], threshold)
        while len(selected) < min_features and threshold <= max_threshold:
            threshold += 0.01
            selected = self.greedy_feature_selection(corr_matrix, y_corr.loc[strong_features], threshold)

        self.selected_features = selected

        print(f"Features remaining after greedy correlation filtering (<{threshold:.2f}): {len(selected)}")

        return X[selected], y

    def transform(self, X):
        if self.selected_features is None:
            raise RuntimeError("FeatureSelector has not been fitted yet.")
        return X[self.selected_features]


class Data_Analysis:
    def __init__(self, corr_thresh=0.85, target_corr_thresh=0.15, min_features=30):
        """
        Parameters:
        - corr_thresh: correlation threshold for greedy filtering between features
        - target_corr_thresh: minimal absolute correlation with target to keep feature
        - min_features: lower bound before dynamic threshold reduction kicks in
        """
        self.corr_thresh = corr_thresh
        self.target_corr_thresh = target_corr_thresh
        self.min_features = min_features

    def analyze(self, X, y, output_dir=None):
        """
        Runs analysis and generates plots without dropping features.
        """

        # --- 1. Feature-to-target correlations ---
        y_corr = pd.Series(index=X.columns, dtype=float)
        for col in X.columns:
            corr, _ = pointbiserialr(y, X[col])
            y_corr[col] = corr

        # Initial filter
        thresh = self.target_corr_thresh
        strong_features = y_corr[abs(y_corr) >= thresh].index

        while len(strong_features) < self.min_features and thresh > 0:
            thresh -= 0.005
            strong_features = y_corr[abs(y_corr) >= thresh].index

        print(f"[INFO] |corr| >= {thresh:.3f} keeps {len(strong_features)} features.")

        # --- 2. Plots ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Histogram of feature-to-target correlations
        plt.figure(figsize=(8, 5))
        sns.histplot(y_corr, bins=20, kde=True)
        plt.axvline(thresh, color='red', linestyle='--', label=f"Threshold {thresh:.3f}")
        plt.axvline(-thresh, color='red', linestyle='--')
        plt.title("Feature-to-Target Correlations")
        plt.xlabel("Correlation with Target")
        plt.ylabel("Feature Count")
        plt.legend()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "target_corr_distribution.png"))
        else:
            plt.show()

        # Correlation heatmap of top-N features
        top_features = y_corr.abs().sort_values(ascending=False).head(30).index
        corr_matrix = X[top_features].corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
        plt.title("Feature-to-Feature Correlation (Top 30 by Target Correlation)")
        if output_dir:
            plt.savefig(os.path.join(output_dir, "feature_corr_heatmap.png"))
        else:
            plt.show()

        # Scatter plot: abs(target_corr) vs mean abs(feature_corr)
        mean_feature_corr = corr_matrix.abs().mean()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=y_corr[top_features].abs(),
            y=mean_feature_corr[top_features]
        )
        plt.xlabel("|Correlation with Target|")
        plt.ylabel("Mean |Correlation| with Other Features")
        plt.title("Target Relevance vs Redundancy")
        if output_dir:
            plt.savefig(os.path.join(output_dir, "relevance_vs_redundancy.png"))
        else:
            plt.show()

        return y_corr

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Correlation Analysis")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with features + target")
    parser.add_argument("--target", type=str, required=True, help="Name of target column")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--corr_thresh", type=float, default=0.85)
    parser.add_argument("--target_corr_thresh", type=float, default=0.15)
    parser.add_argument("--min_features", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df[args.target]
    X = df.drop(columns=[args.target])

    analysis = Data_Analysis(
        corr_thresh=args.corr_thresh,
        target_corr_thresh=args.target_corr_thresh,
        min_features=args.min_features
    )
    analysis.analyze(X, y, output_dir=args.out)

'''
class FeatureSelector:
    def __init__(self, corr_tresh=0.85):
        self.corr_tresh = corr_tresh
        self.selected_features = None

    @staticmethod
    def greedy_feature_selection(corr_matrix, threshold):
        selected = []
        for col in corr_matrix.columns:
            if all(abs(corr_matrix[col][sel]) < threshold for sel in selected):
                selected.append(col)
        return selected

    def fit(self, X, y):
        corr_matrix = X.corr()

        # 1. Greedy feature selection
        selected = self.greedy_feature_selection(corr_matrix, self.corr_tresh)
        self.selected_features = selected
        reduced_X = X[selected]

        # 2. Diagnostic plot
        sorted_corrs = plot_feature_max_correlations(X, self.corr_tresh)
        print(f"Number of features with max correlation < {self.corr_tresh}: {(sorted_corrs < self.corr_tresh).sum()}")

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5},
                    xticklabels=True, yticklabels=True)
        plt.title("Correlation Matrix (Lower Triangle)", fontsize=14)
        plt.tight_layout()
        plt.show()

        print(f"Features remaining after greedy correlation filtering (<{self.corr_tresh}): {len(selected)}")

        return reduced_X, y
        
'''
'''class FeatureSelector:
    def __init__(self, top_mi=None):
        self.top_n = top_mi
        self.selected_features = None

    def fit(self, df, label_col='target',
        corr_thresh=0.67,
        ttest_pval_thresh=0.95):
        y = df[label_col]
        features = df.drop(columns=[label_col])

        # T-test
        pvals = []
        for col in features.columns:
            group0 = features[col][y == 0]
            group1 = features[col][y == 1]
            _, p = ttest_ind(group0, group1, equal_var=False, nan_policy='omit')
            pvals.append(p if not np.isnan(p) else 1.0)
        pval_series = pd.Series(pvals, index=features.columns)
        ttest_mask = pval_series <= ttest_pval_thresh
        print(f"Features passing t-test p-value cutoff ({ttest_pval_thresh}): {ttest_mask.sum()} / {len(ttest_mask)}")

        features = features.loc[:, ttest_mask]

        # Correlation filter
        corr_matrix = features.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
        features = features.drop(columns=to_drop)
        print(f"Features remaining after correlation filtering (>{corr_thresh}): {features.shape[1]}")

        self.selected_features = features.columns.tolist()
        print(f"\nSelected features:\n{self.selected_features}")
        return df[self.selected_features + [label_col]]

'''





'''
class FeatureSelector:
    def __init__(self, var_threshold=0.001, corr_threshold=0.95, pval_threshold=0.05):
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold
        self.pval_threshold = pval_threshold
        self.selected_features = None

    def fit(self, df, label_col='cancer_type'):
        # Assumes df is already preprocessed:
        # - label_col is binary numeric 0/1
        # - no metadata columns
        # - all feature columns are numeric

        y = df[label_col]

        # All columns except label_col are features
        features = df.drop(columns=[label_col])
        feature_cols = features.columns.tolist()

        # 1. Variance threshold
        variances = features.var()
        selected_vars = variances[variances > self.var_threshold].index.tolist()

        # 2. Correlation filter (remove highly correlated features)
        corr_matrix = features[selected_vars].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > self.corr_threshold)]
        selected_corr = [f for f in selected_vars if f not in to_drop]

        # 3. Univariate t-test between classes
        pvals = []
        for f in selected_corr:
            stat, pval = ttest_ind(features.loc[y==0, f], features.loc[y==1, f], equal_var=False)
            pvals.append(pval)
        pval_series = pd.Series(pvals, index=selected_corr)

        # Select features with p-val below threshold
        self.selected_features = pval_series[pval_series < self.pval_threshold].index.tolist()
        print(f"Selected {len(self.selected_features)} features after filtering")

        return self

    def transform(self, df):
        if self.selected_features is None:
            raise RuntimeError("Call fit() before transform()")
        return df[self.selected_features + ['cancer_type']]  # keep label too

    def fit_transform(self, df, label_col='cancer_type'):
        self.fit(df, label_col=label_col)
        return self.transform(df)



class FeatureSelector:
    def __init__(self, df, label_col='target', pos_label=1, neg_label=0):
        """
        df: pandas DataFrame with features and target
        label_col: name of the target column
        pos_label, neg_label: values to encode target as 1 and 0 if needed
        """
        self.df = df.copy()
        self.label_col = label_col
        
        # Encode target to binary 0/1 if not already numeric
        if self.df[self.label_col].dtype != np.number:
            self.df[self.label_col] = self.df[self.label_col].apply(lambda x: 1 if x == pos_label else 0)
        
        # Identify mutation/features columns (exclude label)
        self.feature_cols = [col for col in self.df.columns if col != self.label_col]

    def select_features(self, variance_thresh=0.001, pval_thresh=0.05, plot=False):
        # Normalize mutation features per sample (row)
        df_norm = self.df.copy()
        df_norm[self.feature_cols] = df_norm[self.feature_cols].div(df_norm[self.feature_cols].sum(axis=1), axis=0)
        
        # Variance filter
        feature_vars = df_norm[self.feature_cols].var()
        selected_by_var = feature_vars[feature_vars > variance_thresh].index.tolist()

        # T-test between classes per feature
        pvals = []
        for col in selected_by_var:
            group0 = self.df[col][self.df[self.label_col] == 0]
            group1 = self.df[col][self.df[self.label_col] == 1]
            stat, p = ttest_ind(group0, group1, equal_var=False)
            pvals.append(p)

        pval_df = pd.DataFrame({'feature': selected_by_var, 'pval': pvals}).dropna()
        selected_by_pval = pval_df[pval_df['pval'] < pval_thresh]['feature'].tolist()

        if plot:
            # Variance distribution plot
            plt.figure(figsize=(8, 4))
            plt.hist(feature_vars.values, bins=50, edgecolor='black')
            plt.title("Variance Distribution of Mutation Features")
            plt.xlabel("Feature Variance")
            plt.ylabel("Number of Features")
            plt.tight_layout()
            plt.show()

            # p-value plot
            pval_df_sorted = pval_df.sort_values('pval')
            plt.figure(figsize=(8, 4))
            plt.plot(range(len(pval_df_sorted)), pval_df_sorted['pval'], marker='o', linestyle='-', markersize=3)
            plt.axhline(pval_thresh, color='red', linestyle='--', label=f'p={pval_thresh}')
            plt.title("Sorted p-values for Mutation Features (t-test)")
            plt.xlabel("Feature Rank")
            plt.ylabel("p-value")
            plt.legend()
            plt.tight_layout()
            plt.show()

        print(f"Selected {len(selected_by_pval)} features with variance > {variance_thresh} and p-value < {pval_thresh}")
        return selected_by_pval

'''