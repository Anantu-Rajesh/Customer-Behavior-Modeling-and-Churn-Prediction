import numpy as np
import pandas as pd
from src import config
from src.data_preprocessing import load_data as ld
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score

def cluster_data(df):
    feature_list=[ 'total_purchase', 'num_unique_products', 'avg_order_val', 'product_diversity_ratio', 'std_order_val', 'days_since_last_purchase', 'purchase_span', 'avg_days_between_orders', 'days_since_last_cancellation', 'cancellation_rate', 'return_purchase_ratio', 'per_day_purchase_amnt', 'activity_gap']
    X=df[feature_list].copy()
    return X

def skew_handle(X):
    skew_list=['total_purchase', 'num_unique_products', 'avg_order_val', 'std_order_val', 'return_purchase_ratio', 'per_day_purchase_amnt']
    for col in skew_list:
        X[col]=np.log1p(X[col])
    return X

def scale_data(X):
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return X_scaled

def kmeans_check(X_scaled):
    result=[]
    
    for k in range(2,13):
        kmeans=KMeans(n_clusters=k,init='k-means++',n_init=20,max_iter=300,random_state=42)
        labels=kmeans.fit_predict(X_scaled)
        centroids=kmeans.cluster_centers_
        inertia=kmeans.inertia_
        silhouette = silhouette_score(X_scaled, labels)
        
        result.append({
            'k': k,
            'inertia': inertia,
            'silhouette_score': silhouette
        })
        
        print(f"k-val: {k}, inertia: {inertia}, sil_score: {silhouette}\n")
        
    best_k = max(result, key=lambda x: x['silhouette_score'])['k']
    print(f"\nBest k based on silhouette: {best_k}")
    return result
        

def clustering(df):
    X=cluster_data(df)
    X=skew_handle(X)
    X_scaled=scale_data(X)
    results=kmeans_check(X_scaled)


if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    clustering(df)