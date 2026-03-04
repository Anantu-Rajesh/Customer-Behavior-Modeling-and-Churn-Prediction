import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from src import config
from src.data_preprocessing import load_data as ld
from src.models import util
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix


#PHASE 1: selecting top 3-4 models ig idk
def LR_check(X_train, y_train, X_test, y_test):
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    model=LogisticRegression(max_iter=1000,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Logistic Regression: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
def KNN_check(X_train, y_train, X_test, y_test):
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"KNN: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
def NB_check(X_train, y_train, X_test, y_test):
    model=GaussianNB()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)   
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Naive Bayes: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
def SVM_check(X_train, y_train, X_test, y_test):
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    model=SVC(probability=True,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)    
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"SVM: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
def DT_check(X_train, y_train, X_test, y_test):
    model=DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)    
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Decision Tree: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
def RF_check(X_train, y_train, X_test, y_test):
    model=RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Random Forest: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")

def GB_check(X_train, y_train, X_test, y_test):
    model=GradientBoostingClassifier(random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"Gradient Boosting: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")

def XGB_check(X_train, y_train, X_test, y_test):
    model=XGBClassifier(n_estimators=200,learning_rate=0.1,random_state=42,use_label_encoder=False,eval_metric="logloss")
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"XGBoost: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")

def CatBoost_check(X_train, y_train, X_test, y_test):
    model=CatBoostClassifier(iterations=200,learning_rate=0.1,verbose=0,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"CatBoost: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")

def LGBM_check(X_train, y_train, X_test, y_test):
    model=LGBMClassifier(n_estimators=200,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"LGBM: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")

def model(X_train, y_train, X_test, y_test):
    LR_check(X_train, y_train, X_test, y_test)
    KNN_check(X_train, y_train, X_test, y_test)
    NB_check(X_train, y_train, X_test, y_test)
    SVM_check(X_train, y_train, X_test, y_test)
    DT_check(X_train, y_train, X_test, y_test)
    RF_check(X_train, y_train, X_test, y_test)
    GB_check(X_train, y_train, X_test, y_test)
    XGB_check(X_train, y_train, X_test, y_test)
    CatBoost_check(X_train, y_train, X_test, y_test)
    LGBM_check(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    df_labels=ld.load_and_describe_data(config.customer_filepath_with_unsupervised_labels)
    '''print(f"evaluating model performance without unsupervised labels...\n")
    X_train, X_test, y_train, y_test=util.churn_data(df)
    model(X_train, y_train, X_test, y_test)'''
    print(f"evaluating model performance with unsupervised labels...\n")
    X_train, X_test, y_train, y_test=util.churn_data(df_labels)
    model(X_train, y_train, X_test, y_test)