import numpy as np
import pandas as pd
from src import config
from src.data_preprocessing import load_data as ld
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score

def cluster_data(df):
    feature_list=[ 'total_purchase', 'num_unique_products', 'product_diversity_ratio', 'std_order_val', 'days_since_last_purchase', 'avg_days_between_orders', 'days_since_last_cancellation','purchase_span', 'cancellation_rate', 'return_purchase_ratio', 'activity_gap']
    X=df[feature_list].copy()
    return X

def skew_handle(X):
    skew_list=['total_purchase', 'num_unique_products', 'std_order_val', 'return_purchase_ratio']
    for col in skew_list:
        X[col]=np.log1p(X[col])
    return X

def scale_data(X):
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled

def pca_transform(X_scaled):
    pca=PCA(n_components=0.95,random_state=42)
    X_pca=pca.fit_transform(X_scaled)
    return X_pca

'''def corr_check(X_scaled,X_pca):
    corr_scaled=np.corr(X_scaled)
    corr_pca=np.corr(X_pca)
    print("Correlation matrix for scaled data:\n", corr_scaled)
    print("\nCorrelation matrix for PCA-transformed data:\n", corr_pca)
'''
def clustering(df):
    X=cluster_data(df)
    X=skew_handle(X)
    X_scaled=scale_data(X)
    X_pca=pca_transform(X_scaled)
    #corr_check(X_scaled,X_pca)
    return X_scaled,X_pca

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    clustering(df)