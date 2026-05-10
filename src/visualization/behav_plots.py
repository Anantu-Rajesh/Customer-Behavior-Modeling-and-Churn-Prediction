import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

PLOT_BG = '#ffffff'
PAPER_BG = '#fbfdff'
GRID = '#e8eef5'
TEXT = '#1f2937'


def _apply_theme(fig, title, height=450, showlegend=False):
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', x=0.02, xanchor='left'),
        template='plotly_white',
        height=height,
        showlegend=showlegend,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        margin=dict(t=70, b=40, l=15, r=15),
        font=dict(family='Trebuchet MS, sans-serif', color=TEXT, size=13),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0.0,
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#e5e7eb',
            borderwidth=1,
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    return fig

def get_active_customers(df):
    return df[df['high_value_tier'] != 'N/A (Churned)'].copy()

def plot_spend_by_tier_boxplot(predictions_df, customer_features_df):
    df_merged = predictions_df.merge(
        customer_features_df[['customerid', 'total_purchase']],
        on='customerid',
        how='left'
    )

    fig = go.Figure()
    
    tier_order = ['Low Risk', 'Medium Risk', 'High Risk']
    colors = [CHURN_COLORS[tier] for tier in tier_order]
    
    for tier, color in zip(tier_order, colors):
        df_tier = df_merged[df_merged['churn_tier'] == tier]
        
        fig.add_trace(go.Box(
            y=df_tier['total_purchase'],
            name=tier,
            marker_color=color,
            boxmean='sd',
            fillcolor=color,
            line=dict(color=color, width=2),
            hovertemplate='£%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_traces(boxpoints='outliers', jitter=0.25, pointpos=0)
    return _apply_theme(fig, 'Total Spend Distribution by Churn Risk Tier', height=460, showlegend=False)


def plot_recency_by_tier_violin(predictions_df, customer_features_df):
    df_merged = predictions_df.merge(
        customer_features_df[['customerid', 'days_since_last_purchase']],
        on='customerid',
        how='left'
    )

    fig = go.Figure()
    
    tier_order = ['Low Risk', 'Medium Risk', 'High Risk']
    colors = [CHURN_COLORS[tier] for tier in tier_order]
    
    for tier, color in zip(tier_order, colors):
        df_tier = df_merged[df_merged['churn_tier'] == tier]
        
        fig.add_trace(go.Violin(
            y=df_tier['days_since_last_purchase'],
            name=tier,
            fillcolor=color,
            opacity=0.78,
            box_visible=True,
            meanline_visible=True,
            hovertemplate='%{y} days<extra></extra>'
        ))
    
    return _apply_theme(fig, 'Days Since Last Purchase by Churn Risk Tier', height=460, showlegend=False)


def plot_order_frequency_by_tier(predictions_df, customer_features_df):
    df_active = get_active_customers(predictions_df)
    
    df_merged = df_active.merge(
        customer_features_df[['customerid', 'count_orders']],
        on='customerid',
        how='left'
    )
    
    fig = go.Figure()
    
    tier_order = ['Standard', 'Growing Potential', 'VIP']
    colors = [HIGH_VALUE_COLORS[tier] for tier in tier_order]
    
    for tier, color in zip(tier_order, colors):
        df_tier = df_merged[df_merged['high_value_tier'] == tier]
        
        fig.add_trace(go.Box(
            y=df_tier['count_orders'],
            name=tier,
            marker_color=color,
            boxmean='sd',
            fillcolor=color,
            line=dict(color=color, width=2),
            hovertemplate='%{y} orders<extra></extra>'
        ))
    
    fig.update_traces(boxpoints='outliers', jitter=0.22, pointpos=0)
    return _apply_theme(fig, 'Order Frequency by High-Value Tier', height=460, showlegend=False)


def plot_avg_order_value_by_tier(predictions_df, customer_features_df):
    df_active = get_active_customers(predictions_df)
    
    df_merged = df_active.merge(
        customer_features_df[['customerid', 'avg_order_val']],
        on='customerid',
        how='left'
    )
    
    fig = go.Figure()
    
    tier_order = ['Standard', 'Growing Potential', 'VIP']
    colors = [HIGH_VALUE_COLORS[tier] for tier in tier_order]
    
    for tier, color in zip(tier_order, colors):
        df_tier = df_merged[df_merged['high_value_tier'] == tier]
        
        fig.add_trace(go.Violin(
            y=df_tier['avg_order_val'],
            name=tier,
            fillcolor=color,
            opacity=0.78,
            box_visible=True,
            meanline_visible=True,
            hovertemplate='£%{y:,.2f}<extra></extra>'
        ))
    
    return _apply_theme(fig, 'Average Order Value by High-Value Tier', height=460, showlegend=False)


def get_top_spenders_with_risk(predictions_df, customer_features_df, n=20):
    df_merged = predictions_df.merge(
        customer_features_df[['customerid', 'total_purchase', 'count_orders', 
                              'days_since_last_purchase']],
        on='customerid',
        how='left'
    )

    df_top = df_merged.nlargest(n, 'total_purchase')
    
    result = df_top[[
        'customerid',
        'total_purchase',
        'count_orders',
        'days_since_last_purchase',
        'churn_tier',
        'churn_probability',
        'high_value_tier'
    ]].copy()

    result['total_purchase'] = result['total_purchase'].round(2)
    result['churn_probability'] = result['churn_probability'].round(3)

    result['status'] = result.apply(
        lambda row: ' AT RISK!' if row['churn_tier'] == 'High Risk'
        else ' Monitor' if row['churn_tier'] == 'Medium Risk'
        else ' Safe',
        axis=1
    )

    result.columns = ['Customer ID', 'Total Spend (£)', 'Orders', 'Days Since Purchase',
                     'Churn Tier', 'Churn Probability', 'Value Tier', 'Status']
    
    return result


def plot_feature_averages_by_churn_tier(predictions_df, customer_features_df):
    features = ['total_purchase', 'count_orders', 'days_since_last_purchase', 
                'avg_order_val', 'cancellation_rate']
    
    df_merged = predictions_df.merge(
        customer_features_df[['customerid'] + features],
        on='customerid',
        how='left'
    )

    tier_order = ['Low Risk', 'Medium Risk', 'High Risk']
    
    averages = []
    for tier in tier_order:
        df_tier = df_merged[df_merged['churn_tier'] == tier]
        avg_values = df_tier[features].mean()
        averages.append(avg_values)

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Total Spend', 'Orders', 'Days Since Purchase',
                       'Avg Order Value', 'Cancellation Rate', '')
    )
    
    colors = [CHURN_COLORS[tier] for tier in tier_order]
    
    for i, feature in enumerate(features):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        values = [averages[j][feature] for j in range(3)]
        
        fig.add_trace(
            go.Bar(x=tier_order, y=values, marker_color=colors,
                  showlegend=False, hovertemplate='%{y:.2f}<extra></extra>'),
            row=row, col=col
        )
    
    return _apply_theme(fig, 'Average Feature Values by Churn Risk Tier', height=610, showlegend=False)

def behav_plots(predictions_df, customer_features_df):
    print("\n CATEGORY 2: Customer Behavior Visuals...")
    
    fig_spend = plot_spend_by_tier_boxplot(predictions_df, customer_features_df)
    print("Spend by tier boxplot")
    
    fig_recency = plot_recency_by_tier_violin(predictions_df, customer_features_df)
    print("Recency by tier violin plot")
    
    fig_orders = plot_order_frequency_by_tier(predictions_df, customer_features_df)
    print("Order frequency by tier")
    
    fig_avg_order = plot_avg_order_value_by_tier(predictions_df, customer_features_df)
    print("Avg order value by tier")
    
    table_spenders = get_top_spenders_with_risk(predictions_df, customer_features_df, n=10)
    print("Top spenders with risk table")
    
    fig_features = plot_feature_averages_by_churn_tier(predictions_df, customer_features_df)
    print("Feature averages by tier")
    
'''if __name__ == "__main__":
    results = behav_plots(config.customer_predictions_filepath, config.customer_nlp_filepath_with_labels)'''