
import numpy as np
import pandas as pd
from src import config
from src.data_preprocessing import load_data as ld
from src.models import clustering as cl
from src.models import anomaly_detection as ad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def cluster_data(df):
    feature_list=[ 'total_purchase', 'num_unique_products', 'product_diversity_ratio', 'std_order_val', 'days_since_last_purchase', 'avg_days_between_orders', 'days_since_last_cancellation','purchase_span', 'cancellation_rate', 'return_purchase_ratio', 'activity_gap']
    X=df[feature_list].copy()
    return X

def churn_data(df, model_type):
    X=df.copy()
    drop_cols=['customerid','high_value_customer','high_future_cancellation', 'first_purchase_date', 'last_purchase_date','last_cancel_date','cluster_name']
    label_cols=['cluster_label','if_label','lof_label']
    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)
    y=X.pop('churn')
    
    if model_type == 'linear':
        # Apply log transformation to skewed features
        X = skew_handle(X)
        
        # Scale numeric features (but NOT the categorical label columns)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in label_cols]
        
        if cols_to_scale:
            scaler = StandardScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
            print(f"  Scaled {len(cols_to_scale)} features")
    
    elif model_type == 'tree':
        pass
    
    else:
        raise ValueError(f"model_type must be 'tree' or 'linear', got {model_type}")
    
    for col in label_cols:
        if col in X.columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    print(f"Data prepared for {model_type} models:")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Preprocessing: {'Log + Scale' if model_type == 'linear' else 'None (original features)'}")
    print(f"\nFirst few rows:\n{X.head()}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def high_value_data(df, model_type):
    X=df.copy()
    drop_cols=['customerid','churn','high_future_cancellation', 'first_purchase_date', 'last_purchase_date','last_cancel_date','cluster_name']
    label_cols=['cluster_label','if_label','lof_label']
    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)
    y=X.pop('high_value_customer')
    
    if model_type == 'linear':
        # Apply log transformation to skewed features
        X = skew_handle(X)
        
        # Scale numeric features (but NOT the categorical label columns)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in label_cols]
        
        if cols_to_scale:
            scaler = StandardScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
            print(f"  Scaled {len(cols_to_scale)} features")
    
    elif model_type == 'tree':
        pass
    
    else:
        raise ValueError(f"model_type must be 'tree' or 'linear', got {model_type}")
    
    for col in label_cols:
        if col in X.columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    print(f"Data prepared for {model_type} models:")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Preprocessing: {'Log + Scale' if model_type == 'linear' else 'None (original features)'}")
    print(f"\nFirst few rows:\n{X.head()}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def high_risk_data(df, model_type):
    X=df.copy()
    drop_cols=['customerid','high_value_customer','churn', 'first_purchase_date', 'last_purchase_date','last_cancel_date','cluster_name']
    label_cols=['cluster_label','if_label','lof_label']
    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)
    y=X.pop('high_future_cancellation')
    
    if model_type == 'linear':
        # Apply log transformation to skewed features
        X = skew_handle(X)
        
        # Scale numeric features (but NOT the categorical label columns)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in label_cols]
        
        if cols_to_scale:
            scaler = StandardScaler()
            X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
            print(f"  Scaled {len(cols_to_scale)} features")
    
    elif model_type == 'tree':
        pass
    
    else:
        raise ValueError(f"model_type must be 'tree' or 'linear', got {model_type}")
    
    for col in label_cols:
        if col in X.columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True)
    
    print(f"Data prepared for {model_type} models:")
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Preprocessing: {'Log + Scale' if model_type == 'linear' else 'None (original features)'}")
    print(f"\nFirst few rows:\n{X.head()}\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def skew_handle(X):
    skew_list=['total_purchase','count_orders','tot_items','num_unique_products','avg_order_val','avg_items_per_order','max_order_val','min_order_val','std_order_val','total_cancellation_count','total_cancellation_amnt','total_cancelled_qty','return_purchase_ratio','per_day_purchase_amnt']
    for col in skew_list:
        if col in X.columns:
            X[col]=np.log1p(X[col])
    return X

def scale_data(X):
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    return scaler,X_scaled

def pca_transform(X_scaled):
    pca=PCA(n_components=0.95,random_state=42)
    X_pca=pca.fit_transform(X_scaled)
    return pca,X_pca

def label_assign(cluster_labels, if_labels,if_scores, lof_labels, lof_scores, customer_df):
    
    #handling for cluster labels
    customer_df['cluster_label']=cluster_labels
    cluster_names = {
    0: "One-Time Churners",
    1: "Engaged Regulars",
    2: "At-Risk Irregulars"
    }
    customer_df['cluster_name'] = customer_df['cluster_label'].map(cluster_names)
    print(f"Number of samples in each cluster: {customer_df['cluster_label'].value_counts()}\n")
    print(f"Spend distribution by cluster:\n{customer_df.groupby('cluster_label')['total_purchase'].describe()}\n")
    
    profile = customer_df.groupby('cluster_label').agg({
    'customerid': 'count',
    'total_purchase': ['mean', 'median'],
    'count_orders': ['mean', 'median'],
    'avg_order_val': ['mean', 'median'],
    'days_since_last_purchase': ['mean', 'median'],
    'cancellation_rate': ['mean', 'median'],
    'activity_gap': 'mean',
    'churn': 'mean'  # Churn rate per cluster
    }).round(2)

    profile.to_csv("cluster_profile.csv")
    print(f"Cluster profile saved to cluster_profile.csv\n")
    
    #handling for IF labels
    if_labels=np.where(if_labels==-1,1,0)
    customer_df['if_label']=if_labels
    customer_df['if_score']=if_scores
    
    #handling for LOF labels
    lof_labels=np.where(lof_labels==-1,1,0)
    customer_df['lof_label']=lof_labels     
    customer_df['lof_score']=lof_scores
    
    print(f"anomaly detection labels assigned to customer_df\n")
    
    return customer_df
 
    
def utils(df):
    X=cluster_data(df)
    X=skew_handle(X)
    scaler,X_scaled=scale_data(X)
    pca,X_pca=pca_transform(X_scaled)
    return scaler,X_scaled,pca,X_pca  

'''if __name__ == "__main__":
    df = ld.load_and_describe_data(config.customer_filepath)
    scaler,X_scaled,pca,X_pca=utils(df)
    cluster_model,cluster_labels=cl.clustering(X_pca)
    nlp_features = ld.load_and_describe_data(config.customer_nlp_filepath)
    if_model, if_labels,if_scores,lof_model,lof_labels,lof_scores=ad.anomaly_detection(X_pca)
    customer_df=label_assign(cluster_labels, if_labels,if_scores, lof_labels, lof_scores, df)
    customer_df.to_csv(config.customer_filepath_with_unsupervised_labels, index=False)
    print(f"Customer data with cluster and anomaly labels saved to {config.customer_filepath_with_unsupervised_labels}\n")  
    customer_df_final = customer_df.merge(nlp_features,on='customerid',how='left')
    customer_df_final = customer_df_final.fillna({
    'product_cluster_diversity': 0,
    'primary_product_cluster': -1,
    'product_cluster_entropy': 0
    })
    customer_df_final.to_csv(config.customer_nlp_filepath_with_labels, index=False)
    print(f"Customer data with cluster, anomaly labels and NLP features saved to {config.customer_nlp_filepath_with_labels}\n")'''

