from src import config
from src.models import util
from src.data_preprocessing import load_data as ld
from sklearn.cluster import KMeans


'''def corr_check(X_scaled,X_pca):
    corr_scaled=np.corr(X_scaled)
    corr_pca=np.corr(X_pca)
    print("Correlation matrix for scaled data:\n", corr_scaled)
    print("\nCorrelation matrix for PCA-transformed data:\n", corr_pca)
'''
def cluster_final(X):
    cluster_model=KMeans(n_clusters=3,init='k-means++',n_init=20,max_iter=300,random_state=42)
    cluster_labels=cluster_model.fit_predict(X)
    return cluster_model,cluster_labels
    
def clustering(X_pca):
    #corr_check(X_scaled,X_pca)
    cluster_model,cluster_labels=cluster_final(X_pca)
    return cluster_model,cluster_labels

if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    scaler,X_scaled,pca,X_pca=util.utils(df)
    cluster_model,cluster_labels=clustering(X_pca)