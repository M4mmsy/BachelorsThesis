import warnings
from src.utils.helper_functions import *
from src.preprocess import *
from src.utils.validation_tools import evaluate_performance
from src.models.tab_transformer import hypertrain_ensemble_tab_transformer
from src.models.tabpfn import hypertrain_nested_tabpfn
from src.models.xgb import *
from src.models.mcmc_bnn import hypertrain_ensemble_mcmc_bnn
from src.models.ffn import hypertrain_ensemble_ffn
from src.models.ngboost import hypertrain_nested_ngboost
from src.models.svm import *
from src.models.rf import hypertrain_nested_rf
from src.models.gandalf import hypertrain_ensemble_gandalf
from src.models.vi_bnn import hypertrain_ensemble_vi_bnn
from src.models.light_gmb import *
warnings.filterwarnings('ignore')

def hypertrain_nested(
    X, y, df_cols, shap_selected, model_name, n_repeats, scaling=False, imputing=False,
    corr_thresh=0.85, target_corr_thresh=0.1
    ):
    if not os.path.exists(f'outputs/{model_name}'):
        os.makedirs(f'outputs/{model_name}')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    if model_name == 'xgb':
        return hypertrain_nested_xgb(X, y, df_cols, shap_selected, n_repeats, scaling, imputing, corr_thresh, target_corr_thresh)

    elif model_name == 'light_gbm':
        return hypertrain_nested_light_gbm(X, y, df_cols, shap_selected, n_repeats, scaling, imputing, corr_thresh, target_corr_thresh)

    elif model_name == 'tabpfn':
        return hypertrain_nested_tabpfn(X, y, df_cols, shap_selected, n_repeats, scaling, imputing, corr_thresh, target_corr_thresh)
    
    elif model_name == 'svm':
        return hypertrain_nested_svm(X, y, df_cols, shap_selected, n_repeats, scaling, imputing, corr_thresh, target_corr_thresh)

    elif model_name == 'rf':
        return hypertrain_nested_rf(X, y, df_cols, shap_selected, n_repeats, scaling, imputing, corr_thresh, target_corr_thresh)

    elif model_name == 'ngboost':
        return hypertrain_nested_ngboost(X, y, df_cols, shap_selected, n_repeats, scaling, imputing, corr_thresh, target_corr_thresh)

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def hypertrain(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols, model_name, shap_selected):
    if not os.path.exists(f'outputs/{model_name}'):
        os.makedirs(f'outputs/{model_name}')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')

    # Hypertrain TabTransformer
    if model_name == 'tab_transformer':
        hypertrain_ensemble_tab_transformer(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                            shap_selected)

    # Hypertrain XGBoost
    if model_name == 'xgb':
        hypertrain_ensemble_xgb(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                    shap_selected)

    # Hypertrain LightGBM
    if model_name == 'light_gbm':
        hypertrain_ensemble_light_gbm(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                      shap_selected)

    # Hypertrain MCMC_BNN
    elif model_name == 'mcmc_bnn':
        hypertrain_ensemble_mcmc_bnn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                     shap_selected)

    # Hypertrain VI_BNN
    elif model_name == 'vi_bnn':
        hypertrain_ensemble_vi_bnn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                   shap_selected)

    # Hypertrain Feed Forward Neural Networkf
    elif model_name == 'ffn':
        hypertrain_ensemble_ffn(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                shap_selected)

    # Hypertrain SVM
    elif model_name == 'svm':
        hypertrain_ensemble_svm(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                shap_selected)

    # Hypertrain RF
    elif model_name == 'rf':
        hypertrain_rf(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                               shap_selected)

    elif model_name == 'gandalf':
        hypertrain_ensemble_gandalf(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols,
                                    shap_selected)

if __name__ == '__main__':
    model_name = 'rf'
    shap_selected = False
    scaling = False
    imputing = False
    n_repeats = 8
    corr_thresh = 0.98
    target_corr_thresh = 0

    X, y, df_cols = prepare_freiburg_data()

    print("training ", model_name)
    models, xs_test_list, ys_test_list = hypertrain_nested(
        X, y, df_cols, shap_selected=shap_selected, model_name=model_name,
        n_repeats=n_repeats, scaling=scaling, imputing=imputing,
        corr_thresh=corr_thresh, target_corr_thresh=target_corr_thresh
    )

    evaluate_performance(models, xs_test_list, ys_test_list, df_cols, model_name=model_name, prospective=False)
    



''' The main function that got me the 0.8AUC
if __name__ == '__main__':
    model_name = 'rf'  # Random Forest
    shap_selected = False  # No SHAP selection for paper replic
    scaling = True        # Scaling + PCA as paper uses

    assert model_name in ['svm', 'rf', 'xgb', 'light_gbm', 'ffn', 'gandalf', 'tab_transformer', 'mcmc_bnn', 'vi_bnn']
    if model_name == 'light_gbm':
        scaling = True  # always scale for LightGBM

    # Use the paper-faithful prepare function
    x_train, y_train, x_val, y_val, x_test, y_test, df_cols, scaler = prepare_data()

    print(f'\n ----- Hypertraining model {model_name} ----- \n')
    print('-------------------------------------------')

    # hypertrain expects lists of arrays per imputed dataset, so wrap in list
    xs_train = [x_train]
    ys_train = [y_train]
    xs_val = [x_val]
    ys_val = [y_val]
    xs_test = [x_test]
    ys_test = [y_test]

    # Call your existing hypertrain function with the data
    hypertrain(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, 
               df_cols=df_cols, model_name=model_name, shap_selected=shap_selected)
'''

'''
if __name__ == '__main__':
    model_name = 'rf'
    shap_selected = False
    scaling = True
    undersample = False

    assert model_name in ['svm', 'rf', 'xgb', 'light_gbm', 'ffn', 'gandalf', 'tab_transformer', 'mcmc_bnn', 'vi_bnn']
    if model_name == 'light_gbm':
        scaling = True

    xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols = preprare_data(shap_selected, scaling,
                                                                                  undersample=undersample)

    print(f'\n ----- Hypertraining model {model_name} ----- \n')
    print('-------------------------------------------')

    hypertrain(xs_train, ys_train, xs_val, ys_val, xs_test, ys_test, df_cols=df_cols, model_name=model_name,
               shap_selected=shap_selected)'''
