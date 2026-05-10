"""Data Overview Page - Quick Summary & Navigation Hub"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

feat_df = st.session_state.get(
    "features_df",
    st.session_state.get("customer_features_df", pd.DataFrame()),
)

from src.visualization.data_overview import (
    plot_avg_order_value_distribution,
    plot_cancellation_rate_distribution,
    plot_count_orders_distribution,
    plot_days_since_last_purchase_distribution,
    plot_feature_relationship_heatmap,
    plot_purchase_span_distribution,
    plot_return_purchase_ratio_distribution,
    plot_total_purchase_distribution,
    summarize_customer_overview,
)

if feat_df.empty:
    st.markdown("# Data Overview")
    st.markdown(
        '<p class="page-subtitle">Customer-level summary metrics and behavioral distributions.</p>',
        unsafe_allow_html=True,
    )
    st.info("No customer feature data is available yet. Load data on the Home page first.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# Data Overview")
st.markdown(
    '<p class="page-subtitle">Customer overview, spending patterns, engagement behavior, recency, risk indicators, and feature relationships.</p>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Section 1 ─────────────────────────────────────────────────────────────────
st.markdown("## Section 1: Customer Overview")
summary = summarize_customer_overview(feat_df)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Total Customers", f"{summary['total_customers']:,}")
metric_col2.metric("Avg Spend", f"£{summary['avg_spend']:,.2f}")
metric_col3.metric("Avg Orders", f"{summary['avg_orders']:,.2f}")

st.markdown("---")

# ── Section 2 ─────────────────────────────────────────────────────────────────
st.markdown("## Section 2: Spending & Value")
section2_left, section2_right = st.columns(2)

with section2_left:
    fig = plot_total_purchase_distribution(feat_df)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width='stretch')

with section2_right:
    fig = plot_avg_order_value_distribution(feat_df)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ── Section 3 ─────────────────────────────────────────────────────────────────
st.markdown("## Section 3: Engagement Behavior")
section3_left, section3_right = st.columns(2)

with section3_left:
    fig = plot_count_orders_distribution(feat_df)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width='stretch')

with section3_right:
    fig = plot_purchase_span_distribution(feat_df)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ── Section 4 ─────────────────────────────────────────────────────────────────
st.markdown("## Section 4: Recency & Activity")
fig = plot_days_since_last_purchase_distribution(feat_df)
fig.update_layout(height=390)
st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ── Section 5 ─────────────────────────────────────────────────────────────────
st.markdown("## Section 5: Risk Indicators")
section5_left, section5_right = st.columns(2)

with section5_left:
    fig = plot_return_purchase_ratio_distribution(feat_df)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width='stretch')

with section5_right:
    fig = plot_cancellation_rate_distribution(feat_df)
    fig.update_layout(height=380)
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ── Section 6 ─────────────────────────────────────────────────────────────────
st.markdown("## Section 6: Feature Relationships")
fig = plot_feature_relationship_heatmap(feat_df)
fig.update_layout(height=520)
st.plotly_chart(fig, width='stretch')