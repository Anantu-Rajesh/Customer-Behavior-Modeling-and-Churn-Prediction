import numpy as np
import pandas as pd
from src import config
from src.models import util
from src.data_preprocessing import load_data as ld
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def churn(X_train, y_train, X_test, y_test):
    results=[]
    
    model=GaussianNB(var_smoothing=1e-10)
    model.fit(X_train,y_train)
    
    y_pred=model.predict(X_test)
    acc=accuracy_score(y_test,y_pred)   
    f1=f1_score(y_test,y_pred)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({'F1 Score': f1, 'ROC AUC': roc_auc})
    
    print(f"Naive Bayes: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")
    return model,results

'''if __name__ == "__main__":
    df_labels=ld.load_and_describe_data(config.customer_nlp_filepath_with_labels)
    X_train_linear, X_test_linear, y_train_linear, y_test_linear=util.churn_data(df_labels,model_type='linear')
    churn_model, churn_results=churn(X_train_linear, y_train_linear, X_test_linear, y_test_linear)'''