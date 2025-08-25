import warnings
from utils.helper_functions import *
from preprocess import preprare_data
from src.models.tab_transformer import evaluate_ensemble_tab_transformer
from src.models.xgb import evaluate_ensemble_xgboost
from src.models.ffn import evaluate_ensemble_ffn
from src.models.svm import evaluate_ensemble_svm
from src.models.rf import evaluate_ensemble_rf
from src.models.gandalf import evaluate_ensemble_gandalf
from src.models.vi_bnn import evaluate_ensemble_vi_bnn
from src.models.light_gmb import evaluate_ensemble_light_gbm
from src.utils.smote import apply_smote_to_datasets
warnings.filterwarnings('ignore')


def testing(xs_test, ys_test, df_cols, model_name, shap_selected):
    if not os.path.exists(f'outputs/{model_name}'):
        os.makedirs(f'outputs/{model_name}')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    # Hypertrain TabTransformer
    if model_name == 'tab_transformer':
        evaluate_ensemble_tab_transformer(xs_test, ys_test, df_cols, shap_selected)

    # Hypertrain XGBoost
    if model_name == 'xgb':
        evaluate_ensemble_xgboost(xs_test, ys_test, df_cols, shap_selected)

    # Hypertrain LightGBM
    if model_name == 'light_gbm':
        evaluate_ensemble_light_gbm(xs_test, ys_test, df_cols, shap_selected)

    # Hypertrain VI_BNN
    elif model_name == 'vi_bnn':
        evaluate_ensemble_vi_bnn(xs_test, ys_test, df_cols, shap_selected)

    # Hypertrain Feed Forward Neural Network
    elif model_name == 'ffn':
        evaluate_ensemble_ffn(xs_test, ys_test, df_cols, shap_selected)

    # Hypertrain SVM
    elif model_name == 'svm':
        evaluate_ensemble_svm(xs_test, ys_test, df_cols, shap_selected)

    # Hypertrain RF
    elif model_name == 'rf':
        evaluate_ensemble_rf(xs_test, ys_test, df_cols, shap_selected)

    elif model_name == 'gandalf':
        evaluate_ensemble_gandalf(xs_test, ys_test, df_cols, shap_selected)


if __name__ == '__main__':
    model_name = 'light_gbm'
    shap_selected = False
    scaling = False
    select_patients = False
    smote = False

    assert model_name in ['svm', 'rf', 'xgb', 'light_gbm', 'ffn', 'gandalf', 'tab_transformer', 'mcmc_bnn', 'vi_bnn']

    if model_name == 'vi_bnn':
        scaling = True

    _, _, _, _, xs_test, ys_test, df_cols = preprare_data(shap_selected, scaling, smote=smote, undersample=True)

    print(f'\n ----- Testing model {model_name} ----- \n')
    print('-------------------------------------------')

    testing(xs_test, ys_test, df_cols=df_cols, model_name=model_name, shap_selected=shap_selected)
