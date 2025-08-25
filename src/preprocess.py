from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from src.select_test_datasets import *
from src.utils.plots import *
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import ttest_ind
import shap
import seaborn as sns
import numpy as np
import json


def prepare_freiburg_data():
    df = pd.read_csv(
        'data/Freiburg_Data/analysis_BOPMvsK_0325/cfSigClass_v2.9_full-size_gnomAD_SBSSoNnotsubtracted.SBS96.all', 
        sep='\t'
    ).copy()
    print(f"\nLoaded data with shape: {df.shape}")

    # Encode labels to binary
    df['target'] = df['target'].apply(lambda x: 0 if x == 'control' else 1)

    # Drop non-feature columns if present
    non_feature_cols = ['MutationType']
    df = df.drop(columns=[col for col in non_feature_cols if col in df.columns])

    print(f"Data shape after dropping non-feature columns: {df.shape}")

    # rename mutations to Mutation1, Mutation2, ...., Mutation96 to not have [, > and ] in the names (some models dont like that)
    mutation_cols = [col for col in df.columns if col != 'target']
    print(mutation_cols[:10])
    rename_dict = {old_name: f"Mutation{i+1}" for i, old_name in enumerate(mutation_cols)}
    df = df.rename(columns=rename_dict)

    # Save mapping for interpretability
    with open("mutation_mapping.json", "w") as f:
        json.dump(rename_dict, f, indent=2)

    df_cols = [f"Mutation{i+1}" for i in range(len(mutation_cols))]
    X = df[df_cols].astype(float)  # ensure numeric
    y = df['target'].values

    # Check for missing data
    if X.isnull().any().any():
        print("Warning: Missing values detected in features!")
    return X, y, df_cols

def prepare_paper_data():
    df = pd.read_csv('data/t4.DELFI.sigs.SNPin.csv').copy()
    print(f"\nLoaded data with shape: {df.shape}")

    # Encode labels to binary
    df['cancer_type'] = df['cancer_type'].apply(lambda x: 0 if x == 'Healthy' else 1)

    # Drop non-feature columns if present
    non_feature_cols = ['SLX_barcode','SNP.boolean','evaluable','cancer_type','Stage','MSI','tmb']
    df = df.drop(columns=[col for col in non_feature_cols if col in df.columns])

    print(f"Data shape after dropping non-feature columns: {df.shape}")

    # Identify mutation feature columns
    mutation_cols = [col for col in df.columns if col != 'target']
    rename_dict = {old_name: f"SBS{i+1}" for i, old_name in enumerate(mutation_cols)}
    df = df.rename(columns=rename_dict)

    # Save mapping for interpretability
    with open("mutation_mapping.json", "w") as f:
        json.dump(rename_dict, f, indent=2)

    df_cols = [f"SBS{i+1}" for i in range(len(mutation_cols))]
    X = df[df_cols].astype(float)  # ensure numeric
    y = df['target'].values

    # Check for missing data
    if X.isnull().any().any():
        print("Warning: Missing values detected in features!")

    return X, y, df_cols


def preprocess_nested(X, scaler=None, imputer=None, scaling=False, imputing=False, training=True):
    original_columns = X.columns

    if imputing:
        if training:
            imputer = SimpleImputer(strategy='mean').fit(X)
        X = imputer.transform(X)

    if scaling:
        if training:
            scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    # Wrap back into DataFrame to keep feature names
    X = pd.DataFrame(X, columns=original_columns)

    return X, scaler, imputer


# original from marcus
def preprare_data(shap_selected, scaling, smote=False, undersample=False):
    # Load data
    df = pd.read_csv('data/t4.DELFI.sigs.SNPin.csv')
    print(f'\n----- length of original dataframe: {len(df)} -----\n')

    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)

    data_splits = ['train', 'val', 'test']
    for split in data_splits:
        os.makedirs(f'data/preprocessed_no_mice_{split}/', exist_ok=True)
        os.makedirs(f'data/preprocessed_mice_{split}/', exist_ok=True)

    breakpoint()

    xs_train, ys_train, df_cols, scaler = preprocess(train_df, 'train', scaling=scaling, scaler=None,
                                                     shap_selected=shap_selected, undersample=undersample)
    xs_val, ys_val, df_cols, scaler = preprocess(val_df, 'val', scaling=scaling, scaler=scaler,
                                                 shap_selected=shap_selected, undersample=undersample)
    xs_test, ys_test, df_cols, scaler = preprocess(test_df, 'test', scaling=scaling, scaler=scaler,
                                                   shap_selected=shap_selected, undersample=undersample)

    # Merge and save preprocessed CSVs
    merged_no_mice_df = pd.concat([pd.read_csv(f'data/preprocessed_no_mice_train/train.csv'),
                                   pd.read_csv(f'data/preprocessed_no_mice_val/val.csv'),
                                   pd.read_csv(f'data/preprocessed_no_mice_test/test.csv')],
                                  ignore_index=True)
    merged_no_mice_df.to_csv('data/preprocessed_no_mice_data.csv', index=False)

    merged_mice_complication_df = pd.concat(
        [pd.read_csv(f'data/preprocessed_mice_train/train_0.csv'),
         pd.read_csv(f'data/preprocessed_mice_val/val_0.csv'),
         pd.read_csv(f'data/preprocessed_mice_test/test_0.csv')],
        ignore_index=True)
    merged_mice_complication_df.to_csv('data/preprocessed_mice_data.csv', index=False)

    counts = merged_mice_complication_df['cancer_type'].value_counts()
    total = len(merged_mice_complication_df['cancer_type'])
    percentage_1 = (counts.get(1, 0) / total) * 100
    percentage_0 = (counts.get(0, 0) / total) * 100

    print('-------------------------------------------')
    print(f"Population of 1 without undersampling: {percentage_1:.2f}%")
    print(f"Population of 0 without undersampling: {percentage_0:.2f}%\n")

    return xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols


# The Preprocess from Marcus mixed with mine (SNP Subtraktion and PCA is included here)
def preprocess(df, data_type='train', scaling=True, scaler=None, pca=None,
               shap_selected=False, smote=False, undersample=False, selected_components=None):
    """
    Main function for preprocessing. Preprocesses the data.
    Args:
        df (pd.DataFrame): data to preprocess.
        data_type (str): 'train', 'val', 'test', 'prospective'
        scaling (bool): whether to use scaling
        scaler: standard scaler to use for scaling imputed datasets
        shap_selected (bool): whether to select top-n shap biomarkers for testing validation decrease compared
        to all biomarkers as input
        smote: if smote is used for minority upsampling
        undersample: if True, balance classes through undersampling

    Returns:
        xs (list): m dataframes.
        ys (list): m labels.
        cols (list): column names.
    """

    df = df.reset_index()

    # Convert 'cancer_type' to 0 and all other entries (i.e., healthy) to 1
    df['cancer_type'] = df['cancer_type'].apply(lambda x: 0 if x == 'Healthy' else 1)
    
    columns_to_drop = ['SLX_barcode', 'study', 'stage', 'SNP.boolean', 'maf_targeted_seq']
    df = df.drop(columns=columns_to_drop, axis=1)

    #print(f'NAN Stage: Dropped {len_before - len_after} number of rows {len_before}, {len_after}')
    # SNP Subtraction ( SBS 1 and SBS5 dropped to near 0 because they were age related. I will put them 0 for test, maybe between 0 and 1 is good ?)
    #if 'SBS1' in df.columns:
    #    df['SBS1'] = 0
    #if 'SBS5' in df.columns:
    #    df['SBS5'] = 0    

    df = df.astype(float)

    # Create output directories if they don't exist
    os.makedirs(f'data/preprocessed_no_mice_{data_type}', exist_ok=True)
    os.makedirs(f'data/preprocessed_mice_{data_type}', exist_ok=True)

    # Save the dataframe without MICE
    df.to_csv(f'data/preprocessed_no_mice_{data_type}/{data_type}.csv', index=False)

    # Check if there are still missing values
    if df.isnull().values.any():
        print("[INFO] Missing values detected. Running MICE imputation...")
        dfs = mice(df, 10)
    else:
        dfs = [df]  # So that later code still works (expects a list of dataframes)
    
    # TODO: refactor preprocessing and smote implementation
    if smote:
        dfs_, x_list, y_list = [], [], []
        for idx, df in enumerate(dfs):
            x = df.drop(columns=['cancer_type']).to_numpy()
            y = df['cancer_type'].to_numpy()
            x_list.append(x)
            y_list.append(y.astype(int))

        # Apply SMOTE
        x_list, y_list = apply_smote_to_datasets(x_list, y_list)

        for idx, df in enumerate(dfs):
            # Convert back to DataFrame
            df_smote = pd.DataFrame(x_list[idx], columns=df.drop(
                columns=['cancer_type']).columns)  # Set original column names for features
            df_smote['cancer_type'] = y_list[idx]  # Add the target column back
            dfs_.append(df_smote)
        dfs = dfs_

    ys = []
    xs = []

    if scaling and scaler is None and data_type == 'train':
        scaler = StandardScaler()

    for idx, df in enumerate(dfs):
        y = df['cancer_type']
        y = y.astype(int)
        x = df.drop('cancer_type', axis=1)

        # Undersample training set to balance classes
        if undersample:
            class_0_idx = np.where(y == 0)[0]
            class_1_idx = np.where(y == 1)[0]

            if len(class_0_idx) > len(class_1_idx):
                class_0_idx_downsampled = np.random.choice(class_0_idx, size=len(class_1_idx), replace=False)
                selected_idx = np.concatenate([class_0_idx_downsampled, class_1_idx])
            else:
                class_1_idx_downsampled = np.random.choice(class_1_idx, size=len(class_0_idx), replace=False)
                selected_idx = np.concatenate([class_0_idx, class_1_idx_downsampled])

            np.random.shuffle(selected_idx)
            x = x.iloc[selected_idx]
            y = y.iloc[selected_idx]

        if shap_selected:
            # Columns to select
            columns_to_select = [] # to be defined
            # New DataFrame with selected columns
            x = x[columns_to_select]

        if scaling:
            if data_type == 'train':
                x_scaled = scaler.fit_transform(x)
                pca = PCA()
                x_pca = pca.fit_transform(x_scaled)
                explained_variance = pca.explained_variance_ratio_
                selected_components = explained_variance > 0.01
            else:
                x_scaled = scaler.transform(x)
                x_pca = pca.transform(x_scaled)

            x_pca = x_pca[:, selected_components]

            # Set the correct feature names
            feature_names = [f"SBS_{i}" for i in range(x_pca.shape[1])]

            xs.append(pd.DataFrame(x_pca, columns=feature_names))
        else:
            xs.append(x)
                
        ys.append(y)

        df.to_csv(f'data/preprocessed_mice_{data_type}/{data_type}_{idx}.csv', index=False)

    cols = [*xs[0].columns]

    xs = [df.values for df in xs]
    ys = [df.values for df in ys]

    return xs, ys, cols, scaler, pca if scaling else None, selected_components if scaling else None



if __name__ == '__main__':
    # Load SupplData4 file
    df = pd.read_csv('data/t4.DELFI.sigs.SNPin.csv')

    test_size = 0.15  # 15% for test

    # Preprocess SupplData4 in paper-faithful way
    xs, ys, df_cols, scaler = preprocess_suppldata4(df, data_type='train', scaling=True)

    # xs and ys are lists (of one element), get the arrays
    x = xs[0]
    y = ys[0]

    # Determine number of test samples
    test_samples = int(len(x) * test_size)

    # Split train/test
    x_train = x[:-test_samples]
    y_train = y[:-test_samples]
    x_test = x[-test_samples:]
    y_test = y[-test_samples:]

    # Save processed arrays
    np.save('data/xs_train.npy', x_train)
    np.save('data/xs_test.npy', x_test)
    np.save('data/ys_train.npy', y_train)
    np.save('data/ys_test.npy', y_test)

    # Save feature names
    with open('data/df_cols.pickle', 'wb') as f:
        pickle.dump(df_cols, f)

    # Optional: save scaler if needed for validation/test preprocessing
    with open('data/scaler.pickle', 'wb') as f:
        pickle.dump(scaler, f)