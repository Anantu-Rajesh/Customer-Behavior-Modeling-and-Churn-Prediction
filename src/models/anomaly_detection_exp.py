import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.data_preprocessing import load_data as ld
from src import config
from src.models import util

def IF_check(X, customer_df): 
    contam_vals=[0.03, 0.05, 0.07, 0.1]
    known_outliers=[12346, 15823, 17548, 15838]
    result_IF=[]
    
    for contam in contam_vals:
        model=IsolationForest(contamination=contam,random_state=42)
        labels=model.fit_predict(X)
        labels=np.where(labels==-1,1,0)
        count_outliers=labels.sum()
        pct_outliers=count_outliers/len(X)*100
        outlier_mask=labels==1
        outlier_id=customer_df[outlier_mask]['customerid'].values
        known_found=sum(np.isin(outlier_id, known_outliers))
        outlier_spend=customer_df[outlier_mask]['total_purchase'].mean()
        normal_spend=customer_df[~outlier_mask]['total_purchase'].mean()
        ratio_spend=outlier_spend/normal_spend if normal_spend>0 else np.inf 
        result_IF.append({
            'contamination': contam,
            'num_outliers': count_outliers,
            'pct_outliers': pct_outliers,
            'outlier_id': outlier_id,
            'known_outliers_found': known_found,
            'outlier_spend': outlier_spend,
            'normal_spend': normal_spend,
            'ratio_spend': ratio_spend
        })  
        print(f"IF with contamination={contam}: Found outliers: {count_outliers} known outliers found: {known_found}, spend ratio: {ratio_spend:.2f}")   
    
    return result_IF        

def LOF_check(X, customer_df):
    nbrs=[10, 20, 30, 50]
    known_outliers=[12346, 15823, 17548, 15838]
    result_LOF=[]
    
    for nbr in nbrs:
        model=LocalOutlierFactor(n_neighbors=nbr, contamination=0.05)
        labels=model.fit_predict(X) 
        labels=np.where(labels==-1,1,0)
        count_outliers=np.sum(labels)
        pct_outliers=count_outliers/len(X)*100
        outlier_mask=labels==1
        outlier_id=customer_df[outlier_mask]['customerid'].values
        known_found=sum(np.isin(outlier_id, known_outliers))
        outlier_spend=customer_df[outlier_mask]['total_purchase'].mean()
        normal_spend=customer_df[~outlier_mask]['total_purchase'].mean()
        ratio_spend=outlier_spend/normal_spend if normal_spend>0 else np.inf
        result_LOF.append({
            'n_neighbors': nbr,
            'num_outliers': count_outliers,
            'pct_outliers': pct_outliers,
            'outlier_id': outlier_id,
            'known_outliers_found': known_found,
            'outlier_spend': outlier_spend,
            'normal_spend': normal_spend,
            'ratio_spend': ratio_spend
        })
        print(f"LOF with n_neighbors={nbr}: Found outliers: {count_outliers} known outliers found: {known_found}, spend ratio: {ratio_spend:.2f}")
    return result_LOF

def compare_methods(result_IF_pca, result_LOF_pca):
    if_ids = result_IF_pca[result_IF_pca['contamination'] == 0.05]['outlier_id'].iloc[0]
    lof_ids = result_LOF_pca[result_LOF_pca['n_neighbors'] == 20 ]['outlier_id'].iloc[0]

    if_set = set(if_ids)
    lof_set = set(lof_ids)

    common = if_set & lof_set
    only_if = if_set - lof_set
    only_lof = lof_set - if_set

    print(f"IF outliers: {len(if_set)}")
    print(f"LOF outliers: {len(lof_set)}")
    print(f"Common outliers: {len(common)}")
    print(f"Only IF: {len(only_if)}")
    print(f"Only LOF: {len(only_lof)}")

def anomaly_detection_check(X_scaled, X_pca, customer_df):
    print("Isolation forest chlte hue:")
    print("scaled data pe chlte hue:")
    result_IF_scaled=pd.DataFrame(IF_check(X_scaled,customer_df))

    print("PCA data pe chlte hue:")
    result_IF_pca=pd.DataFrame(IF_check(X_pca,customer_df))
    
    print("\nLocal Outlier Factor chlte hue:")
    print("scaled data pe chlte hue:")  
    result_LOF_scaled=pd.DataFrame(LOF_check(X_scaled,customer_df))
    
    print("PCA data pe chlte hue:")
    result_LOF_pca=pd.DataFrame(LOF_check(X_pca,customer_df))
    
    print("\nComparing IF and LOF results on PCA data:")
    compare_methods(result_IF_pca, result_LOF_pca)
   

if __name__ == "__main__":
    # Load data
    df = ld.load_and_describe_data(config.customer_filepath)
    # Get clustering data (X_scaled and X_pca)
    scaler,X_scaled,pca,X_pca=util.utils(df)  
    # Run experiments
    anomaly_detection_check(X_scaled, X_pca, df)
    # Save results
    # results_df = pd.DataFrame(results)
 