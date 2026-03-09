import pandas as pd
from src import config
from src.data_preprocessing import load_data as ld
from src.models import util
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def NB_check(X_train, y_train, X_test, y_test,params):
    model=GaussianNB(**params)
    model.fit(X_train,y_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    print("NB F1 tuning results:\n")
    best_threshold, f1_tuned= tune_f1(y_train, y_prob_train, y_test, y_prob_test)
    return model, best_threshold, f1_tuned

def RF_check(X_train, y_train, X_test, y_test,params):
    model=RandomForestClassifier(**params)
    model.fit(X_train,y_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    print("RF F1 tuning results:\n")
    best_threshold, f1_tuned= tune_f1(y_train, y_prob_train, y_test, y_prob_test)
    return model, best_threshold, f1_tuned
    
def XGB_check(X_train, y_train, X_test, y_test,params):
    model=XGBClassifier(**params)
    model.fit(X_train,y_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    print("XGB F1 tuning results:\n")
    best_threshold, f1_tuned= tune_f1(y_train, y_prob_train, y_test, y_prob_test)
    return model, best_threshold, f1_tuned

def tune_f1(y_train, y_prob_train, y_test, y_prob_test):
    result=[]
    best_threshold = 0.5
    best_f1_train = 0.0
    thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    for threshold in thresholds:
        y_pred_train = (y_prob_train >= threshold).astype(int)
        f1_train = f1_score(y_train, y_pred_train)
        y_pred_test = (y_prob_test >= threshold).astype(int)
        f1_test = f1_score(y_test, y_pred_test)
        if f1_train > best_f1_train:
            best_f1_train = f1_train
            best_threshold = threshold
    y_pred_test_final = (y_prob_test >= best_threshold).astype(int)
    f1_tuned_test = f1_score(y_test, y_pred_test_final)
    
    f1_default = f1_score(y_test, (y_prob_test >= 0.5).astype(int))
    
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Test F1 (default): {f1_default:.4f}")
    print(f"Test F1 (tuned):   {f1_tuned_test:.4f}")
    print(f"Improvement:       {f1_tuned_test - f1_default:+.4f}\n")
    return best_threshold,f1_tuned_test

def threshold(
    X_train_linear, y_train_linear, X_test_linear, y_test_linear,
    X_train_tree, y_train_tree, X_test_tree, y_test_tree,
    X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl,
    label
):
    if label == 'churn':
        params = config.CHURN_PARAMS

        NB_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear, params['naive_bayes'])
        RF_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['random_forest'])
        XGB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['xgboost'])

    elif label == 'high_value':
        params = config.HIGH_VALUE_PARAMS

        XGB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['xgboost'])
        XGB_check(X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl, params['xgboost'])
        RF_check(X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl, params['random_forest'])

    elif label == 'high_future_cancellation':
        params = config.HIGH_RISK_PARAMS

        RF_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['random_forest'])
        RF_check(X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl, params['random_forest'])
        XGB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['xgboost'])
        

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    df_labels=ld.load_and_describe_data(config.customer_filepath_with_unsupervised_labels)
    df_high_val=df[df['churn']==0].copy()
    df_high_val_labels=df_labels[df_labels['churn']==0].copy()
    
    print(f"for churn prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df_labels,model_type='tree')
    threshold(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree,None,None,None,None,label='churn')
    
    print(f"for high value customer prediction:\n")
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_value_data(df_high_val,model_type='tree')
    X_train_tree_lbl, X_test_tree_lbl, y_train_tree_lbl, y_test_tree_lbl=util.high_value_data(df_high_val_labels,model_type='tree')
    threshold(None,None,None,None,X_train_tree, y_train_tree, X_test_tree, y_test_tree, X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl,label='high_value')
    
    print("for high risk customer prediction:\n")
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_risk_data(df,model_type='tree')
    X_train_tree_lbl, X_test_tree_lbl, y_train_tree_lbl, y_test_tree_lbl=util.high_risk_data(df_labels,model_type='tree')
    threshold(None, None, None, None,X_train_tree, y_train_tree, X_test_tree, y_test_tree, X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl,label='high_future_cancellation')