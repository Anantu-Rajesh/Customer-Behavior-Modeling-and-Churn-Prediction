import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src import config
from src.data_preprocessing import load_data as ld
from src.data_preprocessing import clean_data as cd
from src.data_preprocessing import feature_eng as fe
from src.models import util
from src.models import clustering as cl
from src.models import anomaly_detection as ad
from src.models import churn as churn_model_lib
from src.models import high_risk_customer as high_risk_lib
from src.models import high_val_customer as high_val_lib
from src.visualization import visual
from src.models import save_all


def ensure_dirs():
    os.makedirs("stuff", exist_ok=True)
    os.makedirs("stuff/unsupervised", exist_ok=True)
    os.makedirs("stuff/supervised", exist_ok=True)
    os.makedirs("stuff/nlp", exist_ok=True)


def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}


def run_train_pipeline():
    ensure_dirs()

    raw_df = ld.load_and_describe_data(config.org_filepath)
    clean_df = cd.clean_data(raw_df)
    umap_reducer,product_kmeans=fe.feature_eng(clean_df) 

    customer_df = ld.load_and_describe_data(config.customer_filepath)
    scaler, X_scaled, pca, X_pca = util.utils(customer_df)

    cluster_model, cluster_labels = cl.cluster_final(X_pca)
    if_model, if_labels, if_scores, lof_model, lof_labels, lof_scores = ad.anomaly_detection(X_pca)

    labeled_customer_df = util.label_assign(cluster_labels, if_labels, if_scores, lof_labels, lof_scores, customer_df.copy())
    labeled_customer_df.to_csv(config.customer_filepath_with_unsupervised_labels, index=False)

    nlp_features = ld.load_and_describe_data(config.customer_nlp_filepath)
    labeled_nlp_df = labeled_customer_df.merge(nlp_features, on="customerid", how="left")
    labeled_nlp_df = labeled_nlp_df.fillna({
        "product_cluster_diversity": 0,
        "primary_product_cluster": -1,
        "product_cluster_entropy": 0
    })
    labeled_nlp_df.to_csv(config.customer_nlp_filepath_with_labels, index=False)

    X_churn,X_train_churn, X_test_churn, y_train_churn, y_test_churn, supervised_scaler = util.churn_data(labeled_nlp_df, model_type="linear")
    churn_model, churn_result_list = churn_model_lib.churn(
        X_train_churn, y_train_churn, X_test_churn, y_test_churn
    )

    churn_results = {
        "f1": churn_result_list[0]["F1 Score"],
        "roc_auc": churn_result_list[0]["ROC AUC"]
    }

    hv_df = labeled_nlp_df[labeled_nlp_df["churn"] == 0].copy()
    X_hv,X_train_hv, X_test_hv, y_train_hv, y_test_hv = util.high_value_data(hv_df, model_type="tree")
    high_value_model = high_val_lib.high_val(
        X_train_hv, y_train_hv, X_test_hv, y_test_hv
    )
    high_value_results = compute_metrics(high_value_model, X_test_hv, y_test_hv)

    X_hr,X_train_hr, X_test_hr, y_train_hr, y_test_hr = util.high_risk_data(customer_df, model_type="tree")
    high_risk_model = high_risk_lib.high_risk(
        X_train_hr, y_train_hr, X_test_hr, y_test_hr
    )
    high_risk_results = compute_metrics(high_risk_model, X_test_hr, y_test_hr)

    save_all.save_unsupervised(scaler, pca, cluster_model, if_model, lof_model)
    save_all.save_supervised(
        supervised_scaler,
        churn_model,
        high_risk_model,
        high_value_model,
        churn_results,
        high_risk_results,
        high_value_results
    )
    churn_model=joblib.load(config.churn_model_path)
    high_value_model=joblib.load(config.high_value_model_path)
    high_risk_model=joblib.load(config.high_risk_model_path)
    
    pred=[]
    
    churn_probs=churn_model.predict_proba(X_churn)[:, 1]
    churn_predictions=churn_model.predict(X_churn)  
    high_value_probs=high_value_model.predict_proba(X_hv)[:, 1]
    high_value_predictions=high_value_model.predict(X_hv)
    high_risk_probs=high_risk_model.predict_proba(X_hr)[:, 1]
    high_risk_predictions=high_risk_model.predict(X_hr)
    
    churn_tiers=util.create_churn_tiers(churn_probs)
    high_value_tiers=util.create_high_value_tiers(high_value_probs)
    high_risk_tiers=util.create_high_risk_tiers(high_risk_probs)
    
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
        'customerid':hv_df['customerid'].values,
        'high_value_probability': high_value_probs,
        'high_value_prediction': high_value_predictions,
        'high_value_tier': high_value_tiers
    })
    predictions_df = predictions_df.merge(high_value_df, on='customerid', how='left')

    predictions_df['high_value_probability'] = predictions_df['high_value_probability'].fillna(0)
    predictions_df['high_value_prediction'] = predictions_df['high_value_prediction'].fillna(0)
    predictions_df['high_value_tier'] = predictions_df['high_value_tier'].fillna('N/A (Churned)')

    predictions_df.to_csv('./data/customer_predictions.csv', index=False)
    
    visual.run_visuals(predictions_df, labeled_nlp_df)
    
    save_all.save_nlp(umap_reducer, product_kmeans)


if __name__ == "__main__":
    run_train_pipeline()