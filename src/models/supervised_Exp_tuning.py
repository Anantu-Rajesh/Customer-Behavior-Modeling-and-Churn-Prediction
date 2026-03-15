import pandas as pd
from src import config
from src.data_preprocessing import load_data as ld
from src.models import util
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def LR_tuning(X_train, y_train, X_test, y_test):
    model=LogisticRegression(solver='saga',max_iter=10000,random_state=42)
    
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 50],
    'l1_ratio': [0, 0.5, 1]  
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='f1',n_jobs=2)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV f1: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned Logistic Regression: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    result = {
        'model_name': 'Logistic Regression',
        'best_params': grid.best_params_,
        'best_cv_f1': grid.best_score_,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': acc
    }
    return result

def naive_tuning(X_train, y_train, X_test, y_test):
    model=GaussianNB()
    
    param_grid = {
        'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='f1',n_jobs=2)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV f1: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Naive Bayes: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    result = {
        'model_name': 'Naive Bayes',
        'best_params': grid.best_params_,
        'best_cv_f1': grid.best_score_,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': acc
    }
    return result

def svm_tuning(X_train, y_train, X_test, y_test):
    model=SVC(probability=True,class_weight='balanced',random_state=42)
    
    param_grid = {
    'C': [0.1, 1, 5, 10, 50],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['rbf'],
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='f1',n_jobs=2)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV f1: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned SVM: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    result = {
        'model_name': 'SVM',
        'best_params': grid.best_params_,
        'best_cv_f1': grid.best_score_,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': acc
    }
    return result

def rf_tuning(X_train, y_train, X_test, y_test):
    model=RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=1)
    
    param_grid= {
    'n_estimators': [200, 400, 600],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='f1',n_jobs=2)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV f1: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned Random Forest: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    result = {
        'model_name': 'Random Forest',
        'best_params': grid.best_params_,
        'best_cv_f1': grid.best_score_,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': acc
    }
    return result

def GB_tuning(X_train, y_train, X_test, y_test):
    model=GradientBoostingClassifier(random_state=42)
    
    param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.7, 0.8, 1.0]
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='f1',n_jobs=2)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV f1: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned Gradient Boosting: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    result = {
        'model_name': 'Gradient Boosting',
        'best_params': grid.best_params_,
        'best_cv_f1': grid.best_score_,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': acc
    }
    return result

def xgb_tuning(X_train, y_train, X_test, y_test):
    model=XGBClassifier(use_label_encoder=False,random_state=42,verbosity=0,n_jobs=1)
    
    param_grid={
    'n_estimators': [100, 200],           
    'learning_rate': [0.01, 0.1],         
    'max_depth': [3, 5, 7],               
    'scale_pos_weight': [1, 2, 3]         
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='f1',n_jobs=2)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV f1: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned XGBoost: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    result = {
        'model_name': 'XGBoost',
        'best_params': grid.best_params_,
        'best_cv_f1': grid.best_score_,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy': acc
    }
    return result

def tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree):
    results = []
    results.append(LR_tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear))
    results.append(naive_tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear))
    results.append(svm_tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear))
    results.append(rf_tuning(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    results.append(GB_tuning(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    results.append(xgb_tuning(X_train_tree, y_train_tree, X_test_tree, y_test_tree))
    return results

def save_results(results, target, unsupervised_label=False):
    df = pd.DataFrame(results)
    df['target'] = target
    df['unsupervised_features'] = unsupervised_label
    cols = ['model_name', 'target', 'unsupervised_features', 'best_cv_f1', 'f1', 'roc_auc', 'accuracy', 'best_params']
    df = df[cols]
    suffix = '_with_unsup' if unsupervised_label else '_baseline'
    filename = f'outputs/{target}_tuning_results{suffix}.csv'
    df.to_csv(filename, index=False)
    print(f"\n Saved tuning results to {filename}")
    
if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    df_labels=ld.load_and_describe_data(config.customer_nlp_filepath_with_labels)
    df_high_val=df[df['churn']==0].copy()
    df_high_val_labels=df_labels[df_labels['churn']==0].copy()
    
    print(f"evaluating model performance without unsupervised labels...\n")
    print(f"for churn prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df,model_type='tree')
    results=tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results, target='churn', unsupervised_label=False)

    print(f"for high value customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_value_data(df_high_val,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_value_data(df_high_val,model_type='tree')
    results=tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results, target='high_value', unsupervised_label=False)

    print("for high risk customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_risk_data(df,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_risk_data(df,model_type='tree')
    results=tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results, target='high_risk', unsupervised_label=False)

    print(f"\n\nevaluating model performance with unsupervised labels...\n")
    print(f"for churn prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df_labels,model_type='tree')
    results=tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results, target='churn', unsupervised_label=True)

    print(f"for high value customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_value_data(df_high_val_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_value_data(df_high_val_labels,model_type='tree')
    results=tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results, target='high_value', unsupervised_label=True)

    print("for high risk customer prediction:\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.high_risk_data(df_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_risk_data(df_labels,model_type='tree')
    results=tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear,X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    save_results(results, target='high_risk', unsupervised_label=True)