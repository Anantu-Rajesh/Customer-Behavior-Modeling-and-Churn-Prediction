import pandas as pd
import numpy as np
from src import config
from src.data_preprocessing import load_data as ld
from src.models import util
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def NB_check(X_train, y_train, X_test, y_test,params):
    model=GaussianNB(**params)
    model.fit(X_train,y_train)
    
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    print("NB F1 tuning results:\n")
    best_threshold, f1_tuned= tune_f1(y_train, y_prob_train, y_test, y_prob_test)
    return model, best_threshold, f1_tuned

def SVM_check(X_train, y_train, X_test, y_test,params):
    model=SVC(**params)
    model.fit(X_train,y_train)
    
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    
    print("SVM F1 tuning results:\n")
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

def stacking_check(X_train, y_train, X_test, y_test, X_train_tree, y_train_tree, X_test_tree, y_test_tree, params):
    print("Training individual models for stacking ensemble...")
    nb = GaussianNB(**params['naive_bayes'])
    nb.fit(X_train, y_train)

    svm = SVC(**params['svm'])
    svm.fit(X_train, y_train)

    rf = RandomForestClassifier(**params['random_forest'])
    rf.fit(X_train_tree, y_train_tree)

    xgb = XGBClassifier(**params['xgboost'])
    xgb.fit(X_train_tree, y_train_tree)

    prob_nb  = nb.predict_proba(X_test)[:, 1]
    prob_svm = svm.predict_proba(X_test)[:, 1]
    prob_rf  = rf.predict_proba(X_test_tree)[:, 1]
    prob_xgb = xgb.predict_proba(X_test_tree)[:, 1]

    avg_prob = (prob_nb + prob_svm + prob_rf + prob_xgb) / 4.0

    print("Ensemble (soft vote) F1 tuning results:\n")
    best_threshold, f1_tuned = tune_f1(y_test, avg_prob, y_test, avg_prob)
    return best_threshold, f1_tuned

def tune_f1(y_train, y_prob_train, y_test, y_prob_test):
    thresholds = np.arange(0.10, 0.90, 0.01)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_threshold = 0.5
    best_cv_f1 = 0.0

    for threshold in thresholds:
        fold_f1s = []
        
        for _, val_idx in cv.split(y_prob_train, y_train):
            y_val_true = np.array(y_train)[val_idx]
            y_val_prob = y_prob_train[val_idx]
            y_val_pred = (y_val_prob >= threshold).astype(int)

            if y_val_pred.sum() == 0:
                fold_f1s.append(0.0)
            else:
                fold_f1s.append(f1_score(y_val_true, y_val_pred, zero_division=0))
        mean_cv_f1 = np.mean(fold_f1s)
        
        if mean_cv_f1 > best_cv_f1:
            best_cv_f1 = mean_cv_f1
            best_threshold = threshold

    y_pred_test_final = (y_prob_test >= best_threshold).astype(int)
    f1_tuned_test = f1_score(y_test, y_pred_test_final, zero_division=0)
    f1_default = f1_score(y_test, (y_prob_test >= 0.5).astype(int), zero_division=0)

    print(f"Best threshold (CV): {best_threshold:.2f}")
    print(f"Best CV F1 (train):  {best_cv_f1:.4f}")
    print(f"Test F1 (default):   {f1_default:.4f}")
    print(f"Test F1 (tuned):     {f1_tuned_test:.4f}")
    print(f"Improvement:         {f1_tuned_test - f1_default:+.4f}\n")
    
    return best_threshold, f1_tuned_test

def threshold(
    X_train_linear, y_train_linear, X_test_linear, y_test_linear,
    X_train_tree, y_train_tree, X_test_tree, y_test_tree,
    X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl,
    label
):
    results=[]
    if label == 'churn':
        params = config.CHURN_PARAMS

        model_NB, best_threshold_NB, f1_tuned_NB=NB_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear, params['naive_bayes'])
        model_SVM, best_threshold_SVM, f1_tuned_SVM=SVM_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear, params['svm'])
        model_RF, best_threshold_RF, f1_tuned_RF=RF_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['random_forest'])
        model_XGB, best_threshold_XGB, f1_tuned_XGB=XGB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['xgboost'])
        stacking_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree, params)

    elif label == 'high_value':
        params = config.HIGH_VALUE_PARAMS

        model_SVM, best_threshold_SVM, f1_tuned_SVM=SVM_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear, params['svm'])
        model_XGB, best_threshold_XGB, f1_tuned_XGB=XGB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['xgboost'])
        model_XGB, best_threshold_XGB, f1_tuned_XGB=XGB_check(X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl, params['xgboost'])
        model_RF, best_threshold_RF, f1_tuned_RF=RF_check(X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl, params['random_forest'])

    elif label == 'high_future_cancellation':
        params = config.HIGH_RISK_PARAMS

        model_NB, best_threshold_NB, f1_tuned_NB=NB_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear, params['naive_bayes'])
        model_RF, best_threshold_RF, f1_tuned_RF=RF_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['random_forest'])
        model_RF, best_threshold_RF, f1_tuned_RF=RF_check(X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl, params['random_forest'])
        model_XGB, best_threshold_XGB, f1_tuned_XGB=XGB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree, params['xgboost'])
        
    
        

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    df_labels=ld.load_and_describe_data(config.customer_nlp_filepath_with_labels)
    df_high_val=df[df['churn']==0].copy()
    df_high_val_labels=df_labels[df_labels['churn']==0].copy()
    
    print(f"for churn prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df_labels,model_type='tree')
    threshold(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree,None,None,None,None,label='churn')
    
    print(f"for high value customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_value_data(df_high_val,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_value_data(df_high_val,model_type='tree')
    X_train_tree_lbl, X_test_tree_lbl, y_train_tree_lbl, y_test_tree_lbl=util.high_value_data(df_high_val_labels,model_type='tree')
    threshold(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree, X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl,label='high_value')
    
    print("for high risk customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_risk_data(df,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_risk_data(df,model_type='tree')
    X_train_tree_lbl, X_test_tree_lbl, y_train_tree_lbl, y_test_tree_lbl=util.high_risk_data(df_labels,model_type='tree')
    threshold(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree, X_train_tree_lbl, y_train_tree_lbl, X_test_tree_lbl, y_test_tree_lbl,label='high_future_cancellation')