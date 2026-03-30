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

def plot_cluster_distribution_pie(customer_features_df):
    cluster_names = {
        0: 'One-Time Churners',
        1: 'Engaged Regulars',
        2: 'At-Risk Irregulars'
    }

    cluster_counts = customer_features_df['cluster_label'].value_counts().sort_index()

    labels = [cluster_names.get(i, f'Cluster {i}') for i in cluster_counts.index]

    colors = ['#e74c3c', '#2ecc71', '#f39c12']  # Red, Green, Orange
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=cluster_counts.values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Customer Cluster Distribution (Unsupervised Segmentation)',
        height=400,
        showlegend=True
    )
    #fig.show()
    return fig


def get_cluster_profile_table(customer_features_df, predictions_df):
    cluster_names = {
        0: 'One-Time Churners',
        1: 'Engaged Regulars',
        2: 'At-Risk Irregulars'
    }

    df_merged = customer_features_df.merge(
        predictions_df[['customerid', 'churn_prediction']],
        on='customerid',
        how='left'
    )
    
    profiles = []
    
    for cluster_id in sorted(df_merged['cluster_label'].unique()):
        df_cluster = df_merged[df_merged['cluster_label'] == cluster_id]
        
        profiles.append({
            'Cluster': cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
            'Count': len(df_cluster),
            'Avg Spend (£)': df_cluster['total_purchase'].mean(),
            'Avg Orders': df_cluster['count_orders'].mean(),
            'Churn Rate (%)': df_cluster['churn_prediction'].mean() * 100,
            'Avg Days Since Purchase': df_cluster['days_since_last_purchase'].mean()
        })
    
    result = pd.DataFrame(profiles)

    result['Avg Spend (£)'] = result['Avg Spend (£)'].round(2)
    result['Avg Orders'] = result['Avg Orders'].round(1)
    result['Churn Rate (%)'] = result['Churn Rate (%)'].round(1)
    result['Avg Days Since Purchase'] = result['Avg Days Since Purchase'].round(0)
    
    return result


def plot_churn_rate_by_cluster(customer_features_df, predictions_df):
    cluster_names = {
        0: 'One-Time Churners',
        1: 'Engaged Regulars',
        2: 'At-Risk Irregulars'
    }

    df_merged = customer_features_df.merge(
        predictions_df[['customerid', 'churn_prediction']],
        on='customerid',
        how='left'
    )

    churn_rates = []
    cluster_labels = []
    
    for cluster_id in sorted(df_merged['cluster_label'].unique()):
        df_cluster = df_merged[df_merged['cluster_label'] == cluster_id]
        churn_rate = df_cluster['churn_prediction'].mean() * 100
        
        cluster_labels.append(cluster_names.get(cluster_id, f'Cluster {cluster_id}'))
        churn_rates.append(churn_rate)
    
    colors = ['#e74c3c', '#2ecc71', '#f39c12']
    
    fig = go.Figure(data=[go.Bar(
        x=cluster_labels,
        y=churn_rates,
        marker_color=colors,
        text=[f'{rate:.1f}%' for rate in churn_rates],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Churn Rate: %{y:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title='Churn Rate by Customer Cluster',
        xaxis_title='Cluster',
        yaxis_title='Churn Rate (%)',
        height=400,
        showlegend=False
    )
    #fig.show()
    return fig


def plot_cluster_vs_tier_breakdown(customer_features_df, predictions_df):
    cluster_names = {
        0: 'One-Time Churners',
        1: 'Engaged Regulars',
        2: 'At-Risk Irregulars'
    }

    df_merged = customer_features_df.merge(
        predictions_df[['customerid', 'churn_tier']],
        on='customerid',
        how='left'
    )

    crosstab = pd.crosstab(
        df_merged['cluster_label'],
        df_merged['churn_tier'],
        normalize='index'
    ) * 100  

    tier_order = ['Low Risk', 'Medium Risk', 'High Risk']
    crosstab = crosstab[tier_order]

    crosstab.index = [cluster_names.get(i, f'Cluster {i}') for i in crosstab.index]
    
    fig = go.Figure()
    
    for tier in tier_order:
        fig.add_trace(go.Bar(
            x=crosstab.index,
            y=crosstab[tier],
            name=tier,
            marker_color=CHURN_COLORS[tier],
            hovertemplate='<b>%{x}</b><br>' + tier + ': %{y:.1f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title='Churn Tier Distribution by Cluster',
        xaxis_title='Cluster',
        yaxis_title='Percentage of Customers',
        barmode='stack',
        height=450,
        showlegend=True,
        legend_title='Churn Tier'
    )
    #fig.show()
    return fig

def cluster_plots(predictions_df, customer_features_df):
    print("\n CATEGORY 3: Cluster Visualizations...")
    
    fig_cluster_pie = plot_cluster_distribution_pie(customer_features_df)
    print("Cluster distribution pie chart")
    
    table_cluster = get_cluster_profile_table(customer_features_df, predictions_df)
    print("Cluster profile table")
    
    fig_churn_cluster = plot_churn_rate_by_cluster(customer_features_df, predictions_df)
    print("Churn rate by cluster")
    
    fig_cluster_tier = plot_cluster_vs_tier_breakdown(customer_features_df, predictions_df)
    print("Cluster vs tier breakdown")

'''if __name__ == "__main__":
    results = cluster_plots(config.customer_predictions_filepath, config.customer_nlp_filepath_with_labels)'''