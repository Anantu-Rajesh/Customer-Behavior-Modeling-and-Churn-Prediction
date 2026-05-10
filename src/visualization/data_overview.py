import numpy as np
import pandas as pd
import plotly.graph_objects as go


PLOT_BG = '#ffffff'
PAPER_BG = '#fbfdff'
GRID = '#e8eef5'
TEXT = '#1f2937'
MUTED = '#64748b'

OVERVIEW_FEATURES = [
    'total_purchase',
    'avg_order_val',
    'count_orders',
    'purchase_span',
    'days_since_last_purchase',
    'return_purchase_ratio',
    'cancellation_rate',
]


def _apply_theme(fig, title, height=420, showlegend=False):
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
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def _numeric_series(customer_features_df, column):
    if column not in customer_features_df.columns:
        return pd.Series(dtype='float64')

    series = pd.to_numeric(customer_features_df[column], errors='coerce').dropna()
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _distribution_figure(customer_features_df, column, title, x_title, color='#2563eb', height=420):
    series = _numeric_series(customer_features_df, column)
    fig = go.Figure()

    if series.empty:
        fig.add_annotation(
            text='No data available for this metric.',
            x=0.5,
            y=0.5,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(color=MUTED, size=13),
        )
        return _apply_theme(fig, title, height=height, showlegend=False)

    bin_count = min(max(series.nunique(), 12), 40)
    fig.add_trace(go.Histogram(
        x=series,
        nbinsx=bin_count,
        marker=dict(color=color, line=dict(color='white', width=1)),
        hovertemplate=f'{x_title}: %{{x}}<br>Count: %{{y}}<extra></extra>',
    ))

    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title='Customers')
    return _apply_theme(fig, title, height=height, showlegend=False)


def summarize_customer_overview(customer_features_df):
    total_customers = int(len(customer_features_df))
    avg_spend_series = _numeric_series(customer_features_df, 'total_purchase')
    avg_orders_series = _numeric_series(customer_features_df, 'count_orders')

    return {
        'total_customers': total_customers,
        'avg_spend': float(avg_spend_series.mean()) if not avg_spend_series.empty else 0.0,
        'avg_orders': float(avg_orders_series.mean()) if not avg_orders_series.empty else 0.0,
    }


def plot_total_purchase_distribution(customer_features_df):
    return _distribution_figure(
        customer_features_df,
        'total_purchase',
        'Total Purchase Distribution',
        'Total Purchase (£)',
        color='#2563eb',
        height=430,
    )


def plot_avg_order_value_distribution(customer_features_df):
    return _distribution_figure(
        customer_features_df,
        'avg_order_val',
        'Average Order Value Distribution',
        'Average Order Value (£)',
        color='#0f766e',
        height=430,
    )


def plot_count_orders_distribution(customer_features_df):
    return _distribution_figure(
        customer_features_df,
        'count_orders',
        'Order Count Distribution',
        'Orders per Customer',
        color='#7c3aed',
        height=410,
    )


def plot_purchase_span_distribution(customer_features_df):
    return _distribution_figure(
        customer_features_df,
        'purchase_span',
        'Purchase Span Distribution',
        'Purchase Span (days)',
        color='#ea580c',
        height=410,
    )


def plot_days_since_last_purchase_distribution(customer_features_df):
    return _distribution_figure(
        customer_features_df,
        'days_since_last_purchase',
        'Days Since Last Purchase Distribution',
        'Days Since Last Purchase',
        color='#dc2626',
        height=430,
    )


def plot_return_purchase_ratio_distribution(customer_features_df):
    return _distribution_figure(
        customer_features_df,
        'return_purchase_ratio',
        'Return Purchase Ratio Distribution',
        'Return Purchase Ratio',
        color='#0891b2',
        height=430,
    )


def plot_cancellation_rate_distribution(customer_features_df):
    return _distribution_figure(
        customer_features_df,
        'cancellation_rate',
        'Cancellation Rate Distribution',
        'Cancellation Rate',
        color='#be123c',
        height=430,
    )


def plot_feature_relationship_heatmap(customer_features_df, features=None):
    feature_list = features or OVERVIEW_FEATURES
    available_features = [feature for feature in feature_list if feature in customer_features_df.columns]

    fig = go.Figure()

    if len(available_features) < 2:
        fig.add_annotation(
            text='At least two overview features are required for a correlation heatmap.',
            x=0.5,
            y=0.5,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(color=MUTED, size=13),
        )
        return _apply_theme(fig, 'Feature Relationships', height=470, showlegend=False)

    corr_df = customer_features_df[available_features].apply(pd.to_numeric, errors='coerce').corr().round(2)

    fig.add_trace(go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        zmid=0,
        colorscale='RdBu',
        reversescale=True,
        colorbar=dict(title='Corr.'),
        text=corr_df.astype(str).values,
        texttemplate='%{text}',
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>',
    ))

    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None, autorange='reversed')
    return _apply_theme(fig, 'Feature Relationships', height=470, showlegend=False)


def build_data_overview_figures(customer_features_df):
    return {
        'total_purchase_distribution': plot_total_purchase_distribution(customer_features_df),
        'avg_order_value_distribution': plot_avg_order_value_distribution(customer_features_df),
        'count_orders_distribution': plot_count_orders_distribution(customer_features_df),
        'purchase_span_distribution': plot_purchase_span_distribution(customer_features_df),
        'days_since_last_purchase_distribution': plot_days_since_last_purchase_distribution(customer_features_df),
        'return_purchase_ratio_distribution': plot_return_purchase_ratio_distribution(customer_features_df),
        'cancellation_rate_distribution': plot_cancellation_rate_distribution(customer_features_df),
        'feature_relationship_heatmap': plot_feature_relationship_heatmap(customer_features_df),
    }


def data_overview(customer_features_df):
    print("\n CATEGORY 1: Data Overview Visuals...")

    summary = summarize_customer_overview(customer_features_df)
    print(
        f"Overview summary: {summary['total_customers']} customers, "
        f"avg spend £{summary['avg_spend']:.2f}, avg orders {summary['avg_orders']:.2f}"
    )

    figures = build_data_overview_figures(customer_features_df)
    for label in figures:
        print(label.replace('_', ' ').title())

    return figures