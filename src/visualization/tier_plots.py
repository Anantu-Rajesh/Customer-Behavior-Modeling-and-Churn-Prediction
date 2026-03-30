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

def plot_tier_distribution_pie(df, tier_column, title, color_map):
    tier_counts = df[tier_column].value_counts()
    colors = [color_map.get(tier, '#cccccc') for tier in tier_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=tier_counts.index,
        values=tier_counts.values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='auto',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        showlegend=True,
        height=400
    )
    #fig.show()
    return fig


def plot_churn_tier_distribution(df):
    return plot_tier_distribution_pie(
        df, 
        'churn_tier', 
        'Churn Risk Distribution',
        CHURN_COLORS
    )


def plot_high_value_tier_distribution(df):
    df_active = get_active_customers(df)
    return plot_tier_distribution_pie(
        df_active,
        'high_value_tier',
        'High-Value Customer Distribution',
        HIGH_VALUE_COLORS
    )


def plot_high_risk_tier_distribution(df):
    return plot_tier_distribution_pie(
        df,
        'high_risk_tier',
        'Cancellation Risk Distribution',
        HIGH_RISK_COLORS
    )


def plot_probability_histogram(df, prob_column, title, thresholds, colors):
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[prob_column],
        nbinsx=40,
        marker=dict(
            color='#3498db',
            line=dict(color='white', width=1)
        ),
        name='Distribution',
        hovertemplate='Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    
    fig.add_vrect(
        x0=0, x1=thresholds['low'],
        fillcolor=colors['low'], opacity=0.15,
        line_width=0, annotation_text="Low", annotation_position="top left"
    )
    fig.add_vrect(
        x0=thresholds['low'], x1=thresholds['high'],
        fillcolor=colors['medium'], opacity=0.15,
        line_width=0, annotation_text="Medium", annotation_position="top"
    )
    fig.add_vrect(
        x0=thresholds['high'], x1=1.0,
        fillcolor=colors['high'], opacity=0.15,
        line_width=0, annotation_text="High", annotation_position="top right"
    )
    
    fig.add_vline(x=thresholds['low'], line_dash="dash", line_color="gray", line_width=2)
    fig.add_vline(x=thresholds['high'], line_dash="dash", line_color="gray", line_width=2)
    
    fig.update_layout(
        title=title,
        xaxis_title='Probability',
        yaxis_title='Number of Customers',
        height=400,
        showlegend=False
    )
    #fig.show()
    return fig


def plot_churn_probability_histogram(df):
    return plot_probability_histogram(
        df,
        'churn_probability',
        'Churn Probability Distribution',
        {'low': 0.40, 'high': 0.70},
        {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
    )


def plot_high_value_probability_histogram(df):
    df_active = get_active_customers(df)
    return plot_probability_histogram(
        df_active,
        'high_value_probability',
        'High-Value Probability Distribution',
        {'low': 0.40, 'high': 0.70},
        {'low': '#95a5a6', 'medium': '#3498db', 'high': '#9b59b6'}
    )


def plot_high_risk_probability_histogram(df):
    return plot_probability_histogram(
        df,
        'high_risk_probability',
        'Cancellation Risk Probability Distribution',
        {'low': 0.30, 'high': 0.60},
        {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
    )


def plot_churn_vs_value_scatter(df):
    df_active = get_active_customers(df)
    
    fig = go.Figure()

    for tier in ['Standard', 'Growing Potential', 'VIP']:
        df_tier = df_active[df_active['high_value_tier'] == tier]
        
        fig.add_trace(go.Scatter(
            x=df_tier['churn_probability'],
            y=df_tier['high_value_probability'],
            mode='markers',
            name=tier,
            marker=dict(
                color=HIGH_VALUE_COLORS[tier],
                size=8,
                opacity=0.6,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate=(
                '<b>%{fullData.name}</b><br>'
                'Churn Probability: %{x:.2f}<br>'
                'Value Probability: %{y:.2f}<br>'
                '<extra></extra>'
            )
        ))

    fig.add_hline(y=0.70, line_dash="dash", line_color="gray", line_width=2, opacity=0.7)
    fig.add_vline(x=0.70, line_dash="dash", line_color="gray", line_width=2, opacity=0.7)

    fig.add_annotation(
        x=0.2, y=0.85,
        text="<b>Champions</b><br>(Low Churn, High Value)",
        showarrow=False,
        font=dict(size=11, color="green"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4
    )
    fig.add_annotation(
        x=0.85, y=0.85,
        text="<b>🚨 At-Risk VIPs</b><br>(High Churn, High Value)",
        showarrow=False,
        font=dict(size=11, color="red"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4
    )
    fig.add_annotation(
        x=0.2, y=0.2,
        text="<b>Standard Active</b><br>(Low Churn, Low Value)",
        showarrow=False,
        font=dict(size=11, color="gray"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4
    )
    fig.add_annotation(
        x=0.85, y=0.2,
        text="<b>At-Risk</b><br>(High Churn, Low Value)",
        showarrow=False,
        font=dict(size=11, color="orange"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=4
    )
    
    fig.update_layout(
        title='<b>Customer Segmentation: Churn Risk vs Value Potential</b>',
        xaxis_title='Churn Probability',
        yaxis_title='High-Value Probability',
        height=600,
        showlegend=True,
        legend=dict(
            title="Value Tier",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    #fig.show()
    return fig


def plot_segment_matrix_heatmap(df):
    df_active = get_active_customers(df)

    segment_matrix = pd.crosstab(
        df_active['churn_tier'],
        df_active['high_value_tier']
    )

    churn_order = ['Low Risk', 'Medium Risk', 'High Risk']
    value_order = ['Standard', 'Growing Potential', 'VIP']
    
    segment_matrix = segment_matrix.reindex(
        index=churn_order,
        columns=value_order,
        fill_value=0
    )

    fig = go.Figure(data=go.Heatmap(
        z=segment_matrix.values,
        x=segment_matrix.columns,
        y=segment_matrix.index,
        colorscale='Blues',
        text=segment_matrix.values,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='Churn: %{y}<br>Value: %{x}<br>Customers: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Customer Segment Matrix (Churn × Value)',
        xaxis_title='High-Value Tier',
        yaxis_title='Churn Risk Tier',
        height=400
    )
    
    #fig.show()
    return fig

def plot_spend_by_churn_tier(df, customer_features_df):
    df_merged = df.merge(
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
            hovertemplate='%{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Total Spend Distribution by Churn Risk Tier',
        yaxis_title='Total Purchase Amount (£)',
        xaxis_title='Churn Risk Tier',
        height=400,
        showlegend=False
    )
    
    #fig.show()
    return fig


def plot_days_since_purchase_by_tier(df, customer_features_df):
    df_merged = df.merge(
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
            opacity=0.6,
            hovertemplate='%{y} days<extra></extra>'
        ))
    
    fig.update_layout(
        title='Days Since Last Purchase by Churn Risk Tier',
        yaxis_title='Days Since Last Purchase',
        xaxis_title='Churn Risk Tier',
        height=400,
        showlegend=False
    )
    
    fig.show()
    return fig

def get_summary_metrics(df):
    df_active = get_active_customers(df)

    at_risk_vips = len(df_active[
        (df_active['churn_tier'] == 'High Risk') & 
        (df_active['high_value_tier'] == 'VIP')
    ])
    
    metrics = {
        'total_customers': len(df),
        'high_churn_count': len(df[df['churn_tier'] == 'High Risk']),
        'high_churn_pct': len(df[df['churn_tier'] == 'High Risk']) / len(df) * 100,
        'vip_count': len(df_active[df_active['high_value_tier'] == 'VIP']),
        'vip_pct': len(df_active[df_active['high_value_tier'] == 'VIP']) / len(df_active) * 100,
        'at_risk_vips': at_risk_vips,
        'urgent_attention_count': len(df[df['high_risk_tier'] == 'Urgent Attention'])
    }
    
    return metrics


def get_top_at_risk_customers(df, n=20):
    df_sorted = df.nlargest(n, 'churn_probability')
    
    result = df_sorted[[
        'customerid',
        'churn_probability',
        'churn_tier',
        'high_value_tier',
        'high_risk_tier'
    ]].copy()

    result['churn_probability'] = result['churn_probability'].round(3)

    result['recommended_action'] = result.apply(
        lambda row: ' URGENT: Contact immediately' if row['high_value_tier'] == 'VIP'
        else ' Re-engage with offer' if row['high_value_tier'] == 'Growing Potential'
        else 'Monitor',
        axis=1
    )
    
    return result


def get_top_vip_customers(df, n=20):
    df_active = get_active_customers(df)
    df_sorted = df_active.nlargest(n, 'high_value_probability')
    
    result = df_sorted[[
        'customerid',
        'high_value_probability',
        'high_value_tier',
        'churn_probability',
        'churn_tier'
    ]].copy()
    
    result['high_value_probability'] = result['high_value_probability'].round(3)
    result['churn_probability'] = result['churn_probability'].round(3)

    result['status'] = result.apply(
        lambda row: ' AT RISK!' if row['churn_tier'] == 'High Risk'
        else ' Watch' if row['churn_tier'] == 'Medium Risk'
        else ' Safe',
        axis=1
    )

    result['recommended_action'] = result.apply(
        lambda row: 'Urgent retention offer' if row['churn_tier'] == 'High Risk'
        else 'VIP treatment + monitoring' if row['churn_tier'] == 'Medium Risk'
        else 'Continue VIP benefits',
        axis=1
    )
    
    return result


def get_at_risk_vips(df, n=10):
    df_active = get_active_customers(df)
    
    at_risk_vips = df_active[
        (df_active['churn_tier'] == 'High Risk') & 
        (df_active['high_value_tier'] == 'VIP')
    ].copy()

    at_risk_vips = at_risk_vips.nlargest(n, 'churn_probability')
    
    result = at_risk_vips[[
        'customerid',
        'churn_probability',
        'high_value_probability',
        'high_risk_tier'
    ]].copy()
    
    result['churn_probability'] = result['churn_probability'].round(3)
    result['high_value_probability'] = result['high_value_probability'].round(3)
    
    result['action'] = ' CONTACT IMMEDIATELY - High-value customer about to leave!'
    
    return result


def get_segment_summary(df):
    df_active = get_active_customers(df)

    segments = []

    champions = df_active[
        (df_active['churn_tier'] == 'Low Risk') & 
        (df_active['high_value_tier'] == 'VIP')
    ]
    segments.append({
        'Segment': 'Champions',
        'Count': len(champions),
        'Percentage': len(champions) / len(df_active) * 100,
        'Strategy': 'Reward loyalty, exclusive offers, referral program'
    })

    at_risk_vips = df_active[
        (df_active['churn_tier'] == 'High Risk') & 
        (df_active['high_value_tier'] == 'VIP')
    ]
    segments.append({
        'Segment': ' At-Risk VIPs',
        'Count': len(at_risk_vips),
        'Percentage': len(at_risk_vips) / len(df_active) * 100,
        'Strategy': 'URGENT: Personal outreach, retention offers, VIP treatment'
    })

    growing_stars = df_active[
        (df_active['churn_tier'] == 'Low Risk') & 
        (df_active['high_value_tier'] == 'Growing Potential')
    ]
    segments.append({
        'Segment': 'Growing Stars',
        'Count': len(growing_stars),
        'Percentage': len(growing_stars) / len(df_active) * 100,
        'Strategy': 'Nurture with upsells, bundles, upgrade incentives'
    })

    potential_churners = df_active[
        (df_active['churn_tier'].isin(['Medium Risk', 'High Risk'])) & 
        (df_active['high_value_tier'] == 'Growing Potential')
    ]
    segments.append({
        'Segment': ' Potential Churners',
        'Count': len(potential_churners),
        'Percentage': len(potential_churners) / len(df_active) * 100,
        'Strategy': 'Re-engagement campaigns, special offers, feedback surveys'
    })

    at_risk = df_active[
        (df_active['churn_tier'] == 'High Risk') & 
        (df_active['high_value_tier'] == 'Standard')
    ]
    segments.append({
        'Segment': 'At-Risk',
        'Count': len(at_risk),
        'Percentage': len(at_risk) / len(df_active) * 100,
        'Strategy': 'Win-back campaign, discounts, or let churn naturally'
    })

    standard_active = df_active[
        (df_active['churn_tier'] == 'Low Risk') & 
        (df_active['high_value_tier'] == 'Standard')
    ]
    segments.append({
        'Segment': 'Standard Active',
        'Count': len(standard_active),
        'Percentage': len(standard_active) / len(df_active) * 100,
        'Strategy': 'Standard service, occasional promotions'
    })
    
    result = pd.DataFrame(segments)
    result['Percentage'] = result['Percentage'].round(1)
    
    return result
    
def tier_plots(predictions_df):
    print("\n CATEGORY 1: Tier Visualizations (8 charts)...")
    print("\n1. Tier Distribution Pie Charts...")
    fig1 = plot_churn_tier_distribution(predictions_df)
    print("Churn tier pie chart")
    
    fig2 = plot_high_value_tier_distribution(predictions_df)
    print("High-value tier pie chart")
    
    fig3 = plot_high_risk_tier_distribution(predictions_df)
    print("High-risk tier pie chart")

    print("\n2. Probability Histograms...")
    fig4 = plot_churn_probability_histogram(predictions_df)
    print("Churn probability histogram")
    
    fig5 = plot_high_value_probability_histogram(predictions_df)
    print("High-value probability histogram")
    
    fig6 = plot_high_risk_probability_histogram(predictions_df)
    print("High-risk probability histogram")

    print("\n3. Scatter Plot (Churn vs Value)...")
    fig7 = plot_churn_vs_value_scatter(predictions_df)
    print("Churn vs Value scatter plot ")

    print("\n4. Segment Matrix Heatmap...")
    fig8 = plot_segment_matrix_heatmap(predictions_df)
    print("Segment matrix heatmap")

    print("\n5. Summary Metrics...")
    metrics = get_summary_metrics(predictions_df)
    print(f"Total customers: {metrics['total_customers']}")
    print(f"High churn risk: {metrics['high_churn_count']} ({metrics['high_churn_pct']:.1f}%)")
    print(f"VIP customers: {metrics['vip_count']} ({metrics['vip_pct']:.1f}%)")
    print(f"At-Risk VIPs: {metrics['at_risk_vips']} ")

    print("\n6. Customer Tables...")
    top_risk = get_top_at_risk_customers(predictions_df, n=10)
    print(f"Top 10 at-risk customers")
    
    top_vips = get_top_vip_customers(predictions_df, n=10)
    print(f"Top 10 VIP customers")
    
    at_risk_vips = get_at_risk_vips(predictions_df, n=10)
    print(f"At-Risk VIPs: {len(at_risk_vips)} customers")
    
    segment_summary = get_segment_summary(predictions_df)
    print(f"Segment summary: {len(segment_summary)} segments\n")
    
    print("\nSegment Summary:")
    print(segment_summary.to_string(index=False))
    
    print("\n At-Risk VIPs (URGENT):")
    if len(at_risk_vips) > 0:
        print(at_risk_vips.to_string(index=False))
    else:
        print(" No at-risk VIPs found!")

'''if __name__ == "__main__":
    results = tier_plots(config.customer_predictions_filepath)'''