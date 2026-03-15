import joblib
import json
from src import config
from src.data_preprocessing import load_data as ld
from src.models import util

def save_nlp(umap_reducer, product_kmeans): 
    joblib.dump(umap_reducer, 'stuff/nlp/umap_reducer.pkl')
    joblib.dump(product_kmeans, 'stuff/nlp/product_kmeans.pkl')
    print(f"All NLP models saved to stuff/nlp/")

def save_unsupervised(scaler,pca,cluster_model, if_model, lof_model):
    joblib.dump(scaler, 'stuff/scaler.pkl')
    joblib.dump(pca, 'stuff/unsupervised/pca.pkl')
    joblib.dump(cluster_model, 'stuff/unsupervised/cluster_model.pkl')
    joblib.dump(if_model, 'stuff/unsupervised/isolation_forest.pkl')
    joblib.dump(lof_model, 'stuff/unsupervised/lof_novelty.pkl')   
    print("All unsupervised models saved to stuff/unsupervised/")
    
def save_supervised(churn_model, high_risk_model, high_value_model,churn_results, high_risk_results, high_value_results):
    #for churn model  
    joblib.dump(churn_model, 'stuff/supervised/churn_model.pkl')
    print("Saved to stuff/supervised/churn_model.pkl\n")
    
    #for high risk
    joblib.dump(high_risk_model, 'stuff/supervised/high_risk_model.pkl')
    print("Saved to stuff/supervised/high_risk_model.pkl\n")
    
    #for high value
    joblib.dump(high_value_model, 'stuff/supervised/high_value_model.pkl')
    print("Saved to stuff/supervised/high_value_model.pkl\n")
    
    data={
        'churn': {
            'model_file': 'churn_model.pkl',
            'model_type': 'Naive Bayes',
            'threshold': 0.50,
            'f1_score': churn_results['f1'],
            'roc_auc': churn_results['roc_auc']
        },
        'high_value': {
            'model_file': 'high_value_model.pkl',
            'model_type': 'XGBoost',
            'threshold': 0.50,
            'f1_score': high_value_results['f1'],
            'roc_auc': high_value_results['roc_auc']
        },
        'high_risk': {
            'model_file': 'high_risk_model.pkl',
            'model_type': 'XGBoost',
            'threshold': 0.50,
            'f1_score': high_risk_results['f1'],
            'roc_auc': high_risk_results['roc_auc']
        }
    }

    with open('stuff/supervised/results.json', 'w') as f:
        json.dump(data, f,indent=2)
        print("\nAll models saved successfully!")
    
