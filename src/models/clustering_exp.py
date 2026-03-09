import pandas as pd
from src import config
from src.data_preprocessing import load_data as ld
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score
from src.models import util

def kmeans_check(X):
    result_kmeans=[]
    
    for k in range(2,13):
        kmeans=KMeans(n_clusters=k,init='k-means++',n_init=20,max_iter=300,random_state=42)
        labels=kmeans.fit_predict(X)
        centroids=kmeans.cluster_centers_
        inertia=kmeans.inertia_
        silhouette = silhouette_score(X, labels)
        
        result_kmeans.append({
            'k': k,
            'inertia': inertia,
            'silhouette_score': silhouette
        })
        
        
        print(f"k-val: {k}, inertia: {inertia}, sil_score: {silhouette}")
        print(f"Number of samples in each cluster: {pd.Series(labels).value_counts()}\n")
        
    best_k = max(result_kmeans, key=lambda x: x['silhouette_score'])['k']
    print(f"\nBest k based on silhouette: {best_k}\n")
    return result_kmeans

def hierarchical_cl_check(X):
    result_hierarchy=[]
    
    for linkage in['ward', 'complete', 'average']:
        for k in range(2,11):
            model=AgglomerativeClustering(n_clusters=k,linkage=linkage,metric='euclidean')
            labels=model.fit_predict(X)
            silhouette=silhouette_score(X,labels)
            
            result_hierarchy.append({
                'n':k,
                'linkage':linkage,
                'silhouette_score':silhouette
            })
            
            print(f"linkage: {linkage}, k-val: {k}, sil_score: {silhouette}")
            print(f"Number of samples in each cluster: {pd.Series(labels).value_counts()}\n")
            
    best_hierarchy = max(result_hierarchy, key=lambda x: x['silhouette_score'])
    print(f"\nBest hierarchical clustering based on silhouette: {best_hierarchy}\n")
    return result_hierarchy

def dbscan_check(X):
    results_dbscan=[]
    
    for eps in [0.05,0.3,0.5,0.7,1.2]:
        for min_samples in [5,10,15,20,25]:
            model=DBSCAN(eps=eps,min_samples=min_samples)
            labels=model.fit_predict(X)
            noise_ratio = (labels == -1).sum() / len(labels)
            n_clusters=len(set(labels))-(1 if -1 in labels else 0)
            mask = labels != -1
            if n_clusters>1:
                silhouette = silhouette_score(X[mask], labels[mask])
            else:
                silhouette = -1
            
            results_dbscan.append({
                'eps':eps,
                'min_samples':min_samples,
                'silhouette_score':silhouette,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio
            })
            
            print(f"eps: {eps}, min_samples: {min_samples}, sil_score: {silhouette}")
            print(f"Number of samples in each cluster: {pd.Series(labels).value_counts()}\n")
    best_dbscan = max(results_dbscan, key=lambda x: x['silhouette_score'])
    print(f"\nBest DBSCAN clustering based on silhouette: {best_dbscan}\n")
    return results_dbscan
            

def cluster_check(X_scaled,X_pca):
    print(f"performance of kmeans on X_scaled:")
    result_kmeans_1=kmeans_check(X_scaled)
    print(f"performance of kmeans on X_pca:")
    result_kmeans_2=kmeans_check(X_pca)
    print(f"performance of hierarchical clustering on X_scaled:")
    result_hierarchy_1=hierarchical_cl_check(X_scaled)
    print(f"performance of hierarchical clustering on X_pca:")
    result_hierarchy_2=hierarchical_cl_check(X_pca)
    print(f"performance of DBSCAN on X_scaled:")
    result_dbscan_1=dbscan_check(X_scaled)  
    print(f"performance of DBSCAN on X_pca:")
    result_dbscan_2=dbscan_check(X_pca)
    return result_kmeans_1, result_kmeans_2 #, result_hierarchy_1, result_hierarchy_2, result_dbscan_1, result_dbscan_2

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    scaler,X_scaled,pca,X_pca=util.utils(df)
    cluster_check(X_scaled,X_pca)