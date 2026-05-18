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
from src.models.churn import _ensemble_predict_proba, _ensemble_predict

_MODELS_CACHE={}

def load_model():
    global _MODELS_CACHE
    
    if _MODELS_CACHE:
        return _MODELS_CACHE
    
    models = {
        'scaler': joblib.load('stuff/unsupervised/scaler.pkl'),
        'churn': joblib.load('stuff/supervised/churn_model.pkl'),
        'high_value': joblib.load('stuff/supervised/high_value_model.pkl'),
        'high_risk': joblib.load('stuff/supervised/high_risk_model.pkl'),
        'pca': joblib.load('stuff/unsupervised/pca.pkl'),
        'cluster': joblib.load('stuff/unsupervised/cluster_model.pkl'),
        'isolation_forest': joblib.load('stuff/unsupervised/isolation_forest.pkl'),
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

    if 'invoicedate' not in df.columns:
        raise ValueError("Required column 'InvoiceDate' (invoicedate) is missing after cleaning.")

    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    invalid_dates = int(df['invoicedate'].isna().sum())
    if invalid_dates > 0:
        warnings.append(f"Dropped {invalid_dates} rows with invalid InvoiceDate values.")
        df = df.dropna(subset=['invoicedate'])

    if df.empty:
        raise ValueError("No valid InvoiceDate values available after parsing. Ensure the uploaded file has a valid 'InvoiceDate' column in a recognisable date format.")
    
    ref_date = df['invoicedate'].max()
    df_before = df.copy()  
    if df_before.empty:
        raise ValueError("No transactions exist before the reference date. Dataset may be too short.")

    purchases = df_before[~df_before['is_cancellation']].copy()

    order_totals = purchases.groupby(['customerid', 'invoiceno']).agg(
        order_total=('purchase_amnt', 'sum')
    ).reset_index()

    df_purchase = purchases.groupby('customerid').agg(
        total_purchase=('purchase_amnt', 'sum'),
        count_orders=('invoiceno', 'nunique'),
        tot_items=('purchase_qty', 'sum'),
        first_purchase_date=('invoicedate', 'min'),
        last_purchase_date=('invoicedate', 'max'),
        num_unique_products=('stockcode', 'nunique')
    ).reset_index()

    order_features = order_totals.groupby('customerid').agg(
        max_order_val=('order_total', 'max'),
        min_order_val=('order_total', 'min'),
        std_order_val=('order_total', 'std')
    ).reset_index()

    df_purchase['avg_order_val'] = df_purchase['total_purchase'] / df_purchase['count_orders']
    df_purchase['avg_items_per_order'] = df_purchase['tot_items'] / df_purchase['count_orders']
    df_purchase['product_diversity_ratio'] = np.where(
        df_purchase['tot_items'] > 0,
        df_purchase['num_unique_products'] / df_purchase['tot_items'],
        0
    )
    df_purchase = df_purchase.merge(order_features, on='customerid', how='left')

    df_purchase['first_purchase_date'] = pd.to_datetime(df_purchase['first_purchase_date'])
    df_purchase['last_purchase_date'] = pd.to_datetime(df_purchase['last_purchase_date'])
    df_purchase['days_since_last_purchase'] = (ref_date - df_purchase['last_purchase_date']).dt.days
    df_purchase['days_since_first_purchase'] = (ref_date - df_purchase['first_purchase_date']).dt.days
    df_purchase['purchase_span'] = (df_purchase['last_purchase_date'] - df_purchase['first_purchase_date']).dt.days
    df_purchase['avg_days_between_orders'] = df_purchase.apply(
        lambda row: row['purchase_span'] / (row['count_orders'] - 1) if row['count_orders'] > 1 else 0, axis=1
    )
    df_purchase['std_order_val'] = df_purchase['std_order_val'].fillna(0)
    df_purchase['min_order_val'] = df_purchase['min_order_val'].fillna(df_purchase['max_order_val'])

    cancellations = df_before[df_before['is_cancellation']].copy()
    df_cancel = cancellations.groupby('customerid').agg(
        total_cancellation_count=('invoiceno', 'nunique'),
        total_cancellation_amnt=('cancel_amnt', 'sum'),
        total_cancelled_qty=('cancel_qty', 'sum'),
        last_cancel_date=('invoicedate', 'max')
    ).reset_index()
    df_cancel['last_cancel_date'] = pd.to_datetime(df_cancel['last_cancel_date'])
    df_cancel['days_since_last_cancellation'] = (ref_date - df_cancel['last_cancel_date']).dt.days

    customer_df = df_purchase.merge(df_cancel, on='customerid', how='left')
    customer_df['total_cancellation_count'] = customer_df['total_cancellation_count'].fillna(0)
    customer_df['total_cancellation_amnt'] = customer_df['total_cancellation_amnt'].fillna(0)
    customer_df['total_cancelled_qty'] = customer_df['total_cancelled_qty'].fillna(0)
    customer_df['days_since_last_cancellation'] = customer_df['days_since_last_cancellation'].fillna(
        customer_df['days_since_first_purchase']
    )

    customer_df['cancellation_rate'] = (
        customer_df['total_cancellation_count'] / 
        (customer_df['count_orders'] + customer_df['total_cancellation_count'])
    ).fillna(0)
    customer_df['return_purchase_ratio'] = np.where(
        customer_df['tot_items'] > 0,
        customer_df['total_cancelled_qty'] / customer_df['tot_items'], 0
    ).astype(float)
    customer_df['return_purchase_ratio'] = customer_df['return_purchase_ratio'].replace([np.inf, -np.inf], 0)
    customer_df['per_day_purchase_amnt'] = np.where(
        customer_df['days_since_first_purchase'] > 0,
        customer_df['total_purchase'] / customer_df['days_since_first_purchase'], 0
    )
    customer_df['order_completion_rate'] = np.where(
        (customer_df['count_orders'] + customer_df['total_cancellation_count']) > 0,
        customer_df['count_orders'] / (customer_df['count_orders'] + customer_df['total_cancellation_count']), 0
    )
    customer_df['activity_gap'] = 0
    multi_order = customer_df['count_orders'] > 1
    customer_df.loc[multi_order, 'activity_gap'] = (
        customer_df.loc[multi_order, 'days_since_last_purchase'] > 
        2 * customer_df.loc[multi_order, 'avg_days_between_orders']
    ).astype(int)
    customer_df = customer_df.replace([np.inf, -np.inf], 0)
    
    cluster_df = util.cluster_data(customer_df)
    cluster_df = util.skew_handle(cluster_df)
    
    cluster_df_scaled = models['scaler'].transform(cluster_df)
    cluster_df_pca = models['pca'].transform(cluster_df_scaled)
    
    cluster_labels = models['cluster'].predict(cluster_df_pca)
    
    if_labels = models['isolation_forest'].predict(cluster_df_pca)
    if_scores = models['isolation_forest'].score_samples(cluster_df_pca)
    
    customer_df_labeled = util.label_assign(cluster_labels, if_labels, if_scores, None, None, customer_df)
    
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

    if 'churn' not in customer_df_final.columns:
        customer_df_final['churn'] = 0
    if 'high_value_customer' not in customer_df_final.columns:
        customer_df_final['high_value_customer'] = 0
    if 'high_future_cancellation' not in customer_df_final.columns:
        customer_df_final['high_future_cancellation'] = 0

    X = customer_df_final.copy()
    churn_X = util.prepare_for_inference(X, models, model_key='churn', model_type='linear',scaler_key='supervised_scaler')
    churn_X_tree = util.prepare_for_inference(X, models, model_key='churn', model_type='tree')
    X_hv = util.prepare_for_inference(X, models, model_key='high_value', model_type='tree')
    X_hr = util.prepare_for_inference(customer_df, models, model_key='high_risk', model_type='tree')
        
    churn_probs=_ensemble_predict_proba(models['churn'], churn_X, churn_X_tree)
    high_value_probs=models['high_value'].predict_proba(X_hv)[:, 1]
    high_risk_probs=models['high_risk'].predict_proba(X_hr)[:, 1]
    
    churn_predictions=_ensemble_predict(models['churn'], churn_X, churn_X_tree)
    high_value_predictions = (high_value_probs >= 0.53).astype(int)
    high_risk_predictions=models['high_risk'].predict(X_hr)
    
    churn_tiers = util.create_churn_tiers(churn_probs)
    high_value_tiers = util.create_high_value_tiers(high_value_probs)
    high_risk_tiers = util.create_high_risk_tiers(high_risk_probs)

    predictions_df = pd.DataFrame({
        'customerid': customer_df_final['customerid'].values,
        'churn_probability': churn_probs,
        'churn_prediction': churn_predictions,
        'churn_tier': churn_tiers,
        'high_value_probability': high_value_probs,
        'high_value_prediction': high_value_predictions,
        'high_value_tier': high_value_tiers,
        'high_risk_probability': high_risk_probs,
        'high_risk_prediction': high_risk_predictions,
        'high_risk_tier': high_risk_tiers,
    })
    
    churned_mask = predictions_df['churn_tier'] == 'High Risk'
    non_vip_churned = churned_mask & (predictions_df['high_value_tier'] != 'VIP') & (predictions_df['high_value_tier'] != 'Growing Potential')
    predictions_df.loc[non_vip_churned, 'high_value_tier'] = 'N/A (Churned)'

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
