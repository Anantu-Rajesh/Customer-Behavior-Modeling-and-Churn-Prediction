import pandas as pd
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
    model=LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced')
    
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
    result = {
        'model_name': 'Logistic Regression',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

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
    result = {
        'model_name': 'KNN',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

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
    result = {
        'model_name': 'Naive Bayes',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

def SVM_check(X_train, y_train, X_test, y_test):
    model=SVC(probability=True,random_state=42,class_weight='balanced')
    
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
    result = {
        'model_name': 'SVM',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

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
    result = {
        'model_name': 'Decision Tree',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

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
    result = {
        'model_name': 'Random Forest',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

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
    result = {
        'model_name': 'Gradient Boosting',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

def XGB_check(X_train, y_train, X_test, y_test):
    model=XGBClassifier(n_estimators=200,learning_rate=0.1,random_state=42,use_label_encoder=False,eval_metric="logloss",verbosity=0)
    
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
    result = {
        'model_name': 'XGBoost',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

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
    result = {
        'model_name': 'CatBoost',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]  

def LGBM_check(X_train, y_train, X_test, y_test):
    model=LGBMClassifier(n_estimators=200,class_weight='balanced',random_state=42,verbose=-1)
    
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
    result = {
        'model_name': 'LightGBM',
        'model': model,
        'accuracy': acc,
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'cv_roc_mean': cv_roc.mean(),
        'cv_roc_std': cv_roc.std(),
        'f1': f1,
        'roc_auc': roc_auc
    }
    return [result]

def model(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree):
    result=[]
    result.extend(LR_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear))
    result.extend(KNN_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear))
    result.extend(NB_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear))
    result.extend(SVM_check(X_train_linear, y_train_linear, X_test_linear, y_test_linear))
    result.extend(DT_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    result.extend(RF_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    result.extend(GB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    result.extend(XGB_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    result.extend(CatBoost_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    result.extend(LGBM_check(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    return result

def save_results(result,target,unsupervised_label=False):
    rows=[]
    
    for r in result:
        rows.append(r)
        
    df=pd.DataFrame(rows)
    df['target'] = target
    df['unsupervised_features'] = unsupervised_label
    
    cols = ['model_name', 'target', 'unsupervised_features']
    if 'cv_f1_mean' in df.columns:
        cols.extend(['cv_f1_mean', 'cv_f1_std', 'cv_roc_mean', 'cv_roc_std'])
    if 'f1' in df.columns:
        cols.append('f1')
    if 'roc_auc' in df.columns:
        cols.append('roc_auc')
    if 'accuracy' in df.columns:
        cols.append('accuracy')
    df = df[cols]
    
    suffix = '_with_unsup' if unsupervised_label else '_baseline'
    filename = f'outputs/{target}_results{suffix}.csv'
    df.to_csv(filename, index=False)
    print(f"\n Saved results to {filename}")

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    df_labels=ld.load_and_describe_data(config.customer_filepath_with_unsupervised_labels)
    df_high_val=df[df['churn']==0].copy()
    df_high_val_labels=df_labels[df_labels['churn']==0].copy()
    
    print(f"evaluating model performance without unsupervised labels...\n")
    
    print(f"for churn prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df,model_type='tree')
    results=model(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results,target='churn',unsupervised_label=False)
    
    print(f"for high value customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_value_data(df_high_val,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_value_data(df_high_val,model_type='tree')
    results=model(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results,target='high_value',unsupervised_label=False)
    
    print("for high risk customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_risk_data(df,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_risk_data(df,model_type='tree')
    results=model(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results,target='high_risk',unsupervised_label=False)
    
    print(f"\n\nevaluating model performance with unsupervised labels...\n")
    
    print(f"for churn prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df_labels,model_type='tree')
    results=model(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results,target='churn',unsupervised_label=True)
    
    print(f"for high value customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_value_data(df_high_val_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_value_data(df_high_val_labels,model_type='tree')
    results=model(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results,target='high_value',unsupervised_label=True)
    
    print("for high risk customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_risk_data(df_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_risk_data(df_labels,model_type='tree')
    results=model(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results,target='high_risk',unsupervised_label=True)