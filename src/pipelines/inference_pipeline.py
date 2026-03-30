import os
import json
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src import config
from src.data_preprocessing import load_data as ld
from src.data_preprocessing import clean_data as cd
from src.data_preprocessing import feature_eng as fe
from src.models import util

_MODELS_CACHE={}

def load_model():
    global _MODELS_CACHE
    
    if _MODELS_CACHE:
        return _MODELS_CACHE
    
    models = {
        'scaler': joblib.load('stuff/scaler.pkl'),
        'churn': joblib.load('stuff/supervised/churn_model.pkl'),
        'high_value': joblib.load('stuff/supervised/high_value_model.pkl'),
        'high_risk': joblib.load('stuff/supervised/high_risk_model.pkl'),
        'pca': joblib.load('stuff/unsupervised/pca.pkl'),
        'cluster': joblib.load('stuff/unsupervised/cluster_model.pkl'),
        'isolation_forest': joblib.load('stuff/unsupervised/isolation_forest.pkl'),
        'lof': joblib.load('stuff/unsupervised/lof_novelty.pkl'),
        'umap': joblib.load('stuff/nlp/umap_reducer.pkl'),
        'product_kmeans': joblib.load('stuff/nlp/product_kmeans.pkl'),
        'supervised_scaler': joblib.load('stuff/supervised/scaler.pkl')
    }
    
    _MODELS_CACHE = models
    return models

def check_ip_cols(df):
    cols=['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate','UnitPrice', 'CustomerID', 'Country']
    
    df_cols_lower = [col.lower() for col in df.columns]
    required_lower = [col.lower() for col in cols]
    
    missing_cols = [col for col in cols if col.lower() not in df_cols_lower]
    
    return len(missing_cols) == 0, missing_cols


def _load_metrics():
    metrics_path = 'stuff/supervised/results.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "churn": {"f1_score": 0.687, "roc_auc": 0.735},
        "high_value": {"f1_score": 0.640, "roc_auc": 0.858},
        "high_risk": {"f1_score": 0.312, "roc_auc": 0.791},
    }


def predict_all_customers(df):
    models = load_model()
    raw_rows = len(df)
    warnings = []

    df = ld.normalise_col_names(df)
    df = cd.clean_data(df)
    
    ref_date = df['invoicedate'].max()
    df_purchase = fe.purchase_features(df, ref_date)
    df_cancel = fe.cancellation_features(df, ref_date)
    customer_df = fe.merge_datasets(df_purchase, df_cancel)
    customer_df = fe.derive_features(customer_df)
    
    '''num_cols = ['total_purchase', 'count_orders', 'tot_items', 'num_unique_products', 'avg_order_val', 'avg_items_per_order', 'product_diversity_ratio', 'max_order_val', 'min_order_val', 'std_order_val', 'days_since_last_purchase', 'days_since_first_purchase', 'purchase_span', 'avg_days_between_orders', 'total_cancellation_count', 'total_cancellation_amnt', 'total_cancelled_qty', 'days_since_last_cancellation', 'cancellation_rate', 'order_completion_rate', 'return_purchase_ratio', 'per_day_purchase_amnt']
    print(f"Numerical columns: {num_cols}")'''
    
    cluster_df = util.cluster_data(customer_df)
    cluster_df = util.skew_handle(cluster_df)
    
    cluster_df_scaled = models['scaler'].transform(cluster_df)
    cluster_df_pca = models['pca'].transform(cluster_df_scaled)
    
    cluster_labels = models['cluster'].predict(cluster_df_pca)
    
    if_labels = models['isolation_forest'].predict(cluster_df_pca)
    if_scores = models['isolation_forest'].score_samples(cluster_df_pca)
    
    lof_labels = models['lof'].predict(cluster_df_pca)
    lof_scores = models['lof'].score_samples(cluster_df_pca)
    
    customer_df_labeled = util.label_assign(cluster_labels, if_labels, if_scores, lof_labels, lof_scores, customer_df)
    
    products = df[['stockcode', 'description']].drop_duplicates()
    products = products[products['description'].notna()]
    
    text_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    descriptions = products['description'].tolist()
    embeddings = text_model.encode(descriptions, show_progress_bar=False)
    
    embeddings_reduced = models['umap'].transform(embeddings)
    
    product_clusters = models['product_kmeans'].predict(embeddings_reduced)
    products['product_cluster'] = product_clusters
    
    nlp_features = fe.create_product_features(df, products)
    
    customer_df_final = customer_df_labeled.merge(nlp_features, on='customerid', how='left')
    customer_df_final = customer_df_final.fillna({
    'product_cluster_diversity': 0,
    'primary_product_cluster': -1,
    'product_cluster_entropy': 0
    })
    
    X = customer_df_final.copy()
    drop_cols=['customerid','high_value_customer','high_future_cancellation', 'first_purchase_date', 'last_purchase_date','last_cancel_date','cluster_name']
    label_cols=['cluster_label','if_label','lof_label']
    for col in drop_cols:
        if col in X.columns:
            X.drop(columns=col, inplace=True)
    
    for col in label_cols:
        if col in X.columns:
            X = pd.get_dummies(X, columns=[col], drop_first=True) # X can be used directly for high value and high risk as those are tree based models
    
    churn_X = util.skew_handle(X.copy())
    
    numeric_cols = churn_X.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in label_cols]
        
    if cols_to_scale:
        churn_X[cols_to_scale] = models['supervised_scaler'].transform(churn_X[cols_to_scale])

    if hasattr(models['churn'], 'feature_names_in_'):
        for col in models['churn'].feature_names_in_:
            if col not in churn_X.columns:
                churn_X[col] = 0
        churn_X = churn_X[list(models['churn'].feature_names_in_)]

    if hasattr(models['high_value'], 'feature_names_in_'):
        for col in models['high_value'].feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X_hv = X[list(models['high_value'].feature_names_in_)]
    else:
        X_hv = X

    if hasattr(models['high_risk'], 'feature_names_in_'):
        for col in models['high_risk'].feature_names_in_:
            if col not in X.columns:
                X[col] = 0
        X_hr = X[list(models['high_risk'].feature_names_in_)]
    else:
        X_hr = X
        
    churn_probs=models['churn'].predict_proba(churn_X)[:, 1]
    high_value_probs=models['high_value'].predict_proba(X_hv)[:, 1]
    high_risk_probs=models['high_risk'].predict_proba(X_hr)[:, 1]
    
    churn_predictions=models['churn'].predict(churn_X)
    high_value_predictions=models['high_value'].predict(X_hv)
    high_risk_predictions=models['high_risk'].predict(X_hr)
    
    churn_tiers=util.create_churn_tiers(churn_probs)
    high_value_tiers=util.create_high_value_tiers(high_value_probs)
    high_risk_tiers=util.create_high_risk_tiers(high_risk_probs)
    customer_df['churn_tier']=churn_tiers
    customer_df['high_value_tier']=high_value_tiers
    customer_df['high_risk_tier']=high_risk_tiers
      
    predictions_df = pd.DataFrame({
        'customerid': customer_df['customerid'].values,
        'churn_probability': churn_probs,
        'churn_prediction': churn_predictions,
        'churn_tier': churn_tiers,
        'high_risk_probability': high_risk_probs,
        'high_risk_prediction': high_risk_predictions,
        'high_risk_tier': high_risk_tiers
    })
    
    high_value_df = pd.DataFrame({
        'customerid':customer_df_final['customerid'].values,
        'high_value_probability': high_value_probs,
        'high_value_prediction': high_value_predictions,
        'high_value_tier': high_value_tiers
    })
    predictions_df = predictions_df.merge(high_value_df, on='customerid', how='left')

    predictions_df['high_value_probability'] = predictions_df['high_value_probability'].fillna(0)
    predictions_df['high_value_prediction'] = predictions_df['high_value_prediction'].fillna(0)
    predictions_df['high_value_tier'] = predictions_df['high_value_tier'].fillna('N/A (Churned)')

    upload_status = {
        'rows_in': str(raw_rows),
        'rows_clean': str(len(df)),
        'customers': str(customer_df_final['customerid'].nunique()),
        'features_created': str(customer_df_final.shape[1]),
    }

    if len(df) < raw_rows:
        warnings.append(f"Dropped {raw_rows - len(df)} rows during cleaning.")

    return {
        'predictions_df': predictions_df,
        'customer_features_df': customer_df_final,
        'metrics': _load_metrics(),
        'warnings': warnings,
        'upload_status': upload_status,
    }
    
if __name__ == "__main__":
    pass
