from src import config
from src.data_preprocessing import load_data as ld
from src.models import util
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def LR_tuning(X_train, y_train, X_test, y_test):
    model=LogisticRegression(solver='saga',max_iter=10000,random_state=42)
    
    param_grid = {
    'C': [0.01, 0.1, 1, 10, 50],
    'l1_ratio': [0, 0.5, 1]   # 0 = L2, 1 = L1, 0.5 = ElasticNet
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='roc_auc',n_jobs=-1)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned Logistic Regression: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
def svm_tuning(X_train, y_train, X_test, y_test):
    model=SVC(probability=True,random_state=42)
    
    param_grid = {
    'C': [0.1, 1, 5, 10, 50],
    'gamma': ['scale', 0.001, 0.01, 0.1],
    'kernel': ['rbf'],
    'class_weight': [None, 'balanced']
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='roc_auc',n_jobs=-1)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned SVM: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
def rf_tuning(X_train, y_train, X_test, y_test):
    model=RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    
    param_grid= {
    'n_estimators': [200, 400, 600],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2']
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='roc_auc',n_jobs=-1)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned Random Forest: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
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
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='roc_auc',n_jobs=-1)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned Gradient Boosting: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    
def xgb_tuning(X_train, y_train, X_test, y_test):
    model=XGBClassifier(use_label_encoder=False,eval_metric="auc",random_state=42,verbosity=0)
    
    param_grid = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'min_child_weight': [1, 3, 5]
    }
    
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='roc_auc',n_jobs=-1)
    grid.fit(X_train,y_train)
    
    best_model=grid.best_estimator_
    
    print("Best Parameters:", grid.best_params_)
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    
    y_pred=best_model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
    else:
        y_prob = best_model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Tuned XGBoost: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")

def tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree):
    LR_tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear)
    svm_tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear)
    rf_tuning(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    GB_tuning(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    xgb_tuning(X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    
if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    df_labels=ld.load_and_describe_data(config.customer_filepath_with_unsupervised_labels)
    print(f"evaluating model performance without unsupervised labels...\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df,model_type='tree')
    tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree)
    print(f"\n\nevaluating model performance with unsupervised labels...\n")
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df_labels,model_type='linear')
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.churn_data(df_labels,model_type='tree')
    tuning(X_train_linear, y_train_linear, X_test_linear, y_test_linear, X_train_tree, y_train_tree, X_test_tree, y_test_tree)