import numpy as np
import pandas as pd
from src import config
from src.models import util
from src.data_preprocessing import load_data as ld
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def high_risk(X_train, y_train, X_test, y_test):
    model=XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=200, scale_pos_weight=3, use_label_encoder=False, random_state=42, verbosity=0, n_jobs=1)
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
    return model

'''if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    X_train_tree, X_test_tree, y_train_tree, y_test_tree=util.high_risk_data(df,model_type='tree')
    model=high_risk(X_train_tree, y_train_tree, X_test_tree, y_test_tree)'''