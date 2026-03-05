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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def LR_check(X_train, y_train, X_test, y_test):
    model=LogisticRegression(max_iter=1000,random_state=42)
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Logistic Regression CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    model=KNeighborsClassifier(n_neighbors=5)
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc') 
    print(f"KNN CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Naive Bayes CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    model=SVC(probability=True,random_state=42)
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"SVM CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    model=DecisionTreeClassifier(max_depth=5,class_weight='balanced', random_state=42)
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Decision Tree CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    model=RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Random Forest CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"Gradient Boosting CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"XGBoost CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"CatBoost CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
    
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
    model=LGBMClassifier(n_estimators=200,class_weight='balanced',random_state=42)
    
    cv=StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"LGBM CV F1 Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f} CV ROC AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")

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

def model(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree):
    LR_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear)
    KNN_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear)
    NB_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear)
    SVM_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear)
    DT_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    RF_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    GB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    XGB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    CatBoost_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    LGBM_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    df_labels=ld.load_and_describe_data(config.customer_filepath_with_unsupervised_labels)
    print(f"evaluating model performance without unsupervised labels...\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df,model_type='tree')
    model(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    print(f"\n\nevaluating model performance with unsupervised labels...\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df_labels,model_type='tree')
    model(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
   