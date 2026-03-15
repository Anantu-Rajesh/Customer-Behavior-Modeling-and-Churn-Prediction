from src.models import util
from src.data_preprocessing import load_data as ld
from src import config
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def if_final(X):
    if_model=IsolationForest(contamination=0.05, random_state=42)
    if_labels=if_model.fit_predict(X)
    if_scores=if_model.decision_function(X)
    return if_model, if_labels,if_scores

def lof_final(X):
    lof_model=LocalOutlierFactor(contamination=0.05, n_neighbors=20,novelty=True)
    lof_model.fit(X)
    lof_labels=lof_model.predict(X)
    lof_scores=lof_model.negative_outlier_factor_
    return lof_model,lof_labels,lof_scores

def anomaly_detection(X):
    if_model, if_labels,if_scores=if_final(X)
    lof_model,lof_labels,lof_scores=lof_final(X)
    return if_model, if_labels,if_scores,lof_model,lof_labels,lof_scores

'''if __name__ == "__main__":   
    df = ld.load_and_describe_data(config.customer_filepath)
    scaler,X_scaled,pca,X_pca=util.utils(df)
    if_model, if_labels,if_scores,lof_model,lof_labels,lof_scores=anomaly_detection(X_pca)'''