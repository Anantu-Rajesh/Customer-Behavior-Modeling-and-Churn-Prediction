import pandas as pd
import plotly.graph_objects as go


CHURN_COLORS = {
    'Low Risk': '#2ecc71',      
    'Medium Risk': '#f39c12',   
    'High Risk': '#e74c3c'      
}

HIGH_VALUE_COLORS = {
    'Standard': '#95a5a6',              
    'Growing Potential': '#3498db',     
    'VIP': '#9b59b6'                    
}

HIGH_RISK_COLORS = {
    'Normal': '#2ecc71',            
    'Watch List': '#f39c12',        
    'Urgent Attention': '#e74c3c'   
}

def get_active_customers(df):
    return df[df['high_value_tier'] != 'N/A (Churned)'].copy()

def get_anomaly_summary(customer_features_df):
    if_anomalies = (customer_features_df['if_label'] == 1).sum() if 'if_label' in customer_features_df.columns else 0
    lof_anomalies = (customer_features_df['lof_label'] == 1).sum() if 'lof_label' in customer_features_df.columns else 0
    
    if 'if_label' in customer_features_df.columns and 'lof_label' in customer_features_df.columns:
        total_anomalies = ((customer_features_df['if_label'] == 1) | 
                          (customer_features_df['lof_label'] == 1)).sum()
    else:
        total_anomalies = max(if_anomalies, lof_anomalies)
    
    total_customers = len(customer_features_df)
    
    return {
        'total_anomalies': total_anomalies,
        'anomaly_percentage': (total_anomalies / total_customers * 100),
        'if_anomalies': if_anomalies,
        'lof_anomalies': lof_anomalies,
        'total_customers': total_customers
    }


def get_anomaly_customers_table(customer_features_df, predictions_df, n=20):
    if 'if_label' in customer_features_df.columns and 'lof_label' in customer_features_df.columns:
        df_anomalies = customer_features_df[
            (customer_features_df['if_label'] == 1) | 
            (customer_features_df['lof_label'] == 1)
        ].copy()
    else:
        print("Warning: Anomaly labels not found in customer_features_df")
        return pd.DataFrame()

    df_merged = df_anomalies.merge(
        predictions_df[['customerid', 'churn_tier', 'high_value_tier', 
                       'churn_probability']],
        on='customerid',
        how='left'
    )

    if 'if_score' in df_merged.columns:
        df_top = df_merged.nsmallest(n, 'if_score')  
    else:
        df_top = df_merged.head(n)
    
    result = df_top[[
        'customerid',
        'total_purchase',
        'count_orders',
        'days_since_last_purchase',
        'churn_tier',
        'high_value_tier'
    ]].copy()
    
    result['total_purchase'] = result['total_purchase'].round(2)

    result.columns = ['Customer ID', 'Total Spend (£)', 'Orders', 
                     'Days Since Purchase', 'Churn Tier', 'Value Tier']
    
    return result


def plot_anomaly_distribution_by_tier(customer_features_df, predictions_df):
    df_merged = customer_features_df.merge(
        predictions_df[['customerid', 'churn_tier']],
        on='customerid',
        how='left'
    )

    if 'if_label' in df_merged.columns and 'lof_label' in df_merged.columns:
        df_merged['is_anomaly'] = ((df_merged['if_label'] == 1) | 
                                   (df_merged['lof_label'] == 1)).astype(int)
    else:
        print("Warning: Anomaly labels not found")
        return go.Figure()

    tier_order = ['Low Risk', 'Medium Risk', 'High Risk']
    anomaly_pcts = []
    
    for tier in tier_order:
        df_tier = df_merged[df_merged['churn_tier'] == tier]
        pct = df_tier['is_anomaly'].mean() * 100
        anomaly_pcts.append(pct)
    
    fig = go.Figure(data=[go.Bar(
        x=tier_order,
        y=anomaly_pcts,
        marker_color=[CHURN_COLORS[tier] for tier in tier_order],
        text=[f'{pct:.1f}%' for pct in anomaly_pcts],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Anomaly Rate: %{y:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title='Anomaly Rate by Churn Risk Tier',
        xaxis_title='Churn Tier',
        yaxis_title='Percentage of Anomalies',
        height=400,
        showlegend=False
    )
    #fig.show()
    return fig

def anomaly_plots(predictions_df, customer_features_df):

    print("\n CATEGORY 4: Anomaly Visualizations...")
    
    anomaly_metrics = get_anomaly_summary(customer_features_df)
    print(f"Anomaly summary: {anomaly_metrics['total_anomalies']} anomalies ({anomaly_metrics['anomaly_percentage']:.1f}%)")
    
    table_anomalies = get_anomaly_customers_table(customer_features_df, predictions_df, n=10)
    print(f"Anomaly customers table: {len(table_anomalies)} rows")
    
    fig_anomaly_dist = plot_anomaly_distribution_by_tier(customer_features_df, predictions_df)
    print("Anomaly distribution by tier")
    
'''if __name__ == "__main__":
    results = anomaly_plots(config.customer_predictions_filepath, config.customer_nlp_filepath_with_labels)'''