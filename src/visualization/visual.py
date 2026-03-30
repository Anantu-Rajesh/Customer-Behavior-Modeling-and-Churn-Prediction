import pandas as pd
from src.visualization import tier_plots as tp, behav_plots as bp, cluster_plot as cp, anomaly_plots as ap
from src import config

def run_visuals(predictions_df, customer_features_df):
    '''print("Loading data...")
    
    predictions_df = pd.read_csv(config.customer_predictions_filepath)
    customer_features_df = pd.read_csv(config.customer_nlp_filepath_with_labels)'''
    
    print(f" Loaded {len(predictions_df)} predictions, {len(customer_features_df)} customers\n")
    
    tp.tier_plots(predictions_df)
    
    bp.behav_plots(predictions_df, customer_features_df)
    
    cp.cluster_plots(predictions_df, customer_features_df)
    
    ap.anomaly_plots(predictions_df, customer_features_df)
    
if __name__ == "__main__":
    run_visuals()