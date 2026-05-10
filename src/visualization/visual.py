import pandas as pd
from src.visualization import tier_plots as tp, behav_plots as bp, cluster_plot as cp, anomaly_plots as ap, feature_importance as fp, data_overview as do
from src import config

def run_visuals(X_hr, X_churn_tree, X_hv, predictions_df, customer_features_df):
    if predictions_df is None or customer_features_df is None:
        predictions_df = pd.read_csv(config.customer_predictions_filepath)
        customer_features_df = pd.read_csv(config.customer_nlp_filepath_with_labels)

    if customer_features_df is None or customer_features_df.empty:
        raise ValueError('customer_features_df is required for visual generation')
    
    print(f" Loaded {len(predictions_df)} predictions, {len(customer_features_df)} customers\n")
    
    do.data_overview(customer_features_df)
    
    tp.tier_plots(predictions_df)
    
    bp.behav_plots(predictions_df, customer_features_df)
    
    cp.cluster_plots(predictions_df, customer_features_df)
    
    ap.anomaly_plots(predictions_df, customer_features_df)

    fp.feature_importance_plots(customer_features_df,X_hr,X_churn_tree,X_hv)

    segment_behavior_table = tp.get_segment_behavior_summary_table(predictions_df, customer_features_df)
    print("Segment behavior summary table")
    print(segment_behavior_table)
    
'''if __name__ == "__main__":
    run_visuals(predictions_df=None, customer_features_df=None)'''