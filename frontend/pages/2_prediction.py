"""Predictions Page - ML Model Predictions and Risk Analysis"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.tier_plots import (
    get_active_customers,
    get_top_at_risk_customers,
    get_top_vip_customers,
    plot_churn_tier_distribution,
    plot_churn_probability_histogram,
    plot_high_risk_tier_distribution,
    plot_high_risk_probability_histogram,
    plot_high_value_tier_distribution,
    plot_high_value_probability_histogram,
)
from src.visualization.feature_importance import (
    plot_feature_importance_churn,
    plot_feature_importance_high_value,
    plot_feature_importance_risk,
)
from src.visualization.behav_plots import (
    plot_spend_by_tier_boxplot,
    plot_recency_by_tier_violin,
    plot_avg_order_value_by_tier,
    plot_order_frequency_by_tier,
)
from src.models import util
from src.pipelines import inference_pipeline as inf

# ── Guard ─────────────────────────────────────────────────────────────────────
pred_df = st.session_state.get("predictions_df", pd.DataFrame())
feat_df = st.session_state.get("features_df", pd.DataFrame())

models = inf.load_model() if st.session_state.get("data_loaded") and not pred_df.empty else {}
churn_tree_df = util.prepare_for_inference(feat_df.reset_index(drop=True), models, model_key="churn", model_type="tree") if models else feat_df
high_value_tree_df = util.prepare_for_inference(feat_df, models, model_key="high_value", model_type="tree") if models else feat_df
high_risk_tree_df = util.prepare_for_inference(feat_df, models, model_key="high_risk", model_type="tree") if models else feat_df

if not st.session_state.get("data_loaded") or pred_df.empty:
    st.markdown("#  Predictions")
    st.markdown(
        '<p class="page-subtitle">ML predictions for churn, high-value, and risk classification.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style='text-align:center; padding:3rem 2rem; background:white;
                border:1px solid #e2e8f0; border-radius:12px;'>
        <div style='font-size:2.5rem; margin-bottom:1rem;'></div>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:1.1rem;
                    font-weight:600; color:#475569; margin-bottom:0.5rem;'>
            No predictions yet
        </div>
        <div style='font-size:0.875rem; color:#94a3b8;'>
            Upload your dataset on the Home page to generate predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("#  Predictions")
st.markdown(
    '<p class="page-subtitle">Machine learning predictions for churn, high-value, and risk classification.</p>',
    unsafe_allow_html=True,
)

# ── Summary metrics ───────────────────────────────────────────────────────────
total = len(pred_df)

high_churn = int((pred_df["churn_tier"] == "High Risk").sum()) if "churn_tier" in pred_df.columns else 0
high_churn_pct = high_churn / total * 100 if total else 0

high_value = int((pred_df["high_value_tier"] == "VIP").sum()) if "high_value_tier" in pred_df.columns else 0
high_value_pct = high_value / total * 100 if total else 0

high_risk = int((pred_df["high_risk_tier"] == "Urgent Attention").sum()) if "high_risk_tier" in pred_df.columns else 0
high_risk_pct = high_risk / total * 100 if total else 0

m1, m2, m3 = st.columns(3)
m1.metric("High Churn Risk",      f"{high_churn_pct:.1f}%", f"{high_churn:,} customers")
m2.metric("VIP Customers",        f"{high_value_pct:.1f}%", f"{high_value:,} customers")
m3.metric("Urgent Risk Customers",f"{high_risk_pct:.1f}%",  f"{high_risk:,} customers")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  Churn", "  High Value", "  High Risk"])


# ─── Churn Tab ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Churn Prediction Analysis")
    st.markdown(
        '<p style="color:#64748b;font-size:0.875rem;margin-top:-0.5rem;margin-bottom:1.2rem;">'
        "Identifies customers likely to stop purchasing based on behavioural patterns.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if "churn_probability" in pred_df.columns:
            fig = plot_churn_probability_histogram(pred_df)
            fig.update_layout(height=360)
            st.plotly_chart(fig, width='stretch')

            st.markdown(
                '<div style="display:flex;gap:1.5rem;font-size:0.82rem;color:#475569;">'
                '<span>🟢 <strong>Low Risk</strong> 0–0.4</span>'
                '<span>🟡 <strong>Medium Risk</strong> 0.4–0.7</span>'
                '<span>🔴 <strong>High Risk</strong> 0.7–1.0</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Churn probability data not available.")

    with col2:
        if "churn_tier" in pred_df.columns:
            fig = plot_churn_tier_distribution(pred_df)
            fig.update_layout(height=360)
            st.plotly_chart(fig, width='stretch')

    st.markdown("#### Top Features Driving Churn Prediction")
    st.plotly_chart(plot_feature_importance_churn(feat_df, churn_tree_df), width='stretch')

    st.markdown("#### Behavior vs Churn Risk")
    behav1, behav2 = st.columns(2)
    with behav1:
        st.markdown("**Spend by Churn Risk**")
        st.plotly_chart(plot_spend_by_tier_boxplot(pred_df, feat_df), width='stretch')
    with behav2:
        st.markdown("**Recency by Churn Risk**")
        st.plotly_chart(plot_recency_by_tier_violin(pred_df, feat_df), width='stretch')

    st.markdown("#### Top 20 At-Risk Customers")
    if "churn_probability" in pred_df.columns:
        top_churn = get_top_at_risk_customers(pred_df, n=20)
        if "total_purchase" in feat_df.columns and "customerid" in feat_df.columns:
            top_churn = top_churn.merge(
                feat_df[["customerid", "total_purchase", "days_since_last_purchase"]],
                on="customerid", how="left",
            )
        disp = top_churn.copy()
        if "churn_probability" in disp.columns:
            disp["churn_probability"] = disp["churn_probability"].map("{:.3f}".format)
        if "total_purchase" in disp.columns:
            disp["total_purchase"] = disp["total_purchase"].map("£{:,.2f}".format)
        st.dataframe(disp, width='stretch', height=380, hide_index=True)

    st.info(
        "**Business interpretation:** High-risk customers should be prioritised for "
        "retention campaigns, personalised offers, and win-back strategies."
    )


# ─── High Value Tab ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("### High-Value Customer Analysis")
    st.markdown(
        '<p style="color:#64748b;font-size:0.875rem;margin-top:-0.5rem;margin-bottom:1.2rem;">'
        "Identifies customers with high revenue potential and long-term value.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if "high_value_probability" in pred_df.columns:
            fig = plot_high_value_probability_histogram(pred_df)
            fig.update_layout(height=360)
            st.plotly_chart(fig, width='stretch')

            st.markdown(
                '<div style="display:flex;gap:1.5rem;font-size:0.82rem;color:#475569;">'
                '<span>⚪ <strong>Standard</strong> 0–0.4</span>'
                '<span>🔵 <strong>Growing Potential</strong> 0.4–0.7</span>'
                '<span>🟣 <strong>VIP</strong> 0.7–1.0</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("High-value probability data not available.")

    with col2:
        if "high_value_tier" in pred_df.columns:
            active_df = get_active_customers(pred_df)
            fig = plot_high_value_tier_distribution(active_df)
            fig.update_layout(height=360)
            st.plotly_chart(fig, width='stretch')

    st.markdown("#### Top Features Driving High-Value Classification")
    st.plotly_chart(plot_feature_importance_high_value(high_value_tree_df, feat_df), width='stretch')

    st.markdown("#### Behavior vs High-Value Potential")
    behav1, behav2 = st.columns(2)
    with behav1:
        st.markdown("**Order Value by Tier**")
        st.plotly_chart(plot_avg_order_value_by_tier(pred_df, feat_df), width='stretch')
    with behav2:
        st.markdown("**Order Frequency by Tier**")
        st.plotly_chart(plot_order_frequency_by_tier(pred_df, feat_df), width='stretch')

    st.markdown("#### Top 20 VIP Customers")
    if "high_value_probability" in pred_df.columns:
        top_value = get_top_vip_customers(pred_df, n=20)
        if "total_purchase" in feat_df.columns and "customerid" in feat_df.columns:
            top_value = top_value.merge(
                feat_df[["customerid", "total_purchase", "count_orders"]],
                on="customerid", how="left",
            )
        disp = top_value.copy()
        if "high_value_probability" in disp.columns:
            disp["high_value_probability"] = disp["high_value_probability"].map("{:.3f}".format)
        if "total_purchase" in disp.columns:
            disp["total_purchase"] = disp["total_purchase"].map("£{:,.2f}".format)
        st.dataframe(disp, width='stretch', height=380, hide_index=True)

    st.info(
        "**Business interpretation:** VIP customers should receive exclusive loyalty "
        "programmes, early product access, and premium customer service."
    )


# ─── High Risk Tab ───────────────────────────────────────────────────────────
with tab3:
    st.markdown("### High-Risk Customer Analysis")
    st.markdown(
        '<p style="color:#64748b;font-size:0.875rem;margin-top:-0.5rem;margin-bottom:1.2rem;">'
        "Identifies customers with high cancellation and return behaviour.</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        if "high_risk_probability" in pred_df.columns:
            fig = plot_high_risk_probability_histogram(pred_df)
            fig.update_layout(height=360)
            st.plotly_chart(fig, width='stretch')

            st.markdown(
                '<div style="display:flex;gap:1.5rem;font-size:0.82rem;color:#475569;">'
                '<span>🟢 <strong>Normal</strong> 0–0.3</span>'
                '<span>🟡 <strong>Watch List</strong> 0.3–0.6</span>'
                '<span>🔴 <strong>Urgent Attention</strong> 0.6–1.0</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("High-risk probability data not available.")

    with col2:
        if "high_risk_tier" in pred_df.columns:
            fig = plot_high_risk_tier_distribution(pred_df)
            fig.update_layout(height=360)
            st.plotly_chart(fig, width='stretch')

    st.markdown("#### Top Features Driving High-Risk Classification")
    st.plotly_chart(plot_feature_importance_risk(high_risk_tree_df, feat_df), width='stretch')

    st.markdown("#### Top 20 High-Risk Customers")
    if "high_risk_probability" in pred_df.columns:
        top_risk = pred_df.nlargest(20, "high_risk_probability")[
            ["customerid", "high_risk_probability", "high_risk_tier"]
        ]
        if "cancellation_rate" in feat_df.columns and "customerid" in feat_df.columns:
            top_risk = top_risk.merge(
                feat_df[["customerid", "cancellation_rate", "total_cancellation_count"]],
                on="customerid", how="left",
            )
        disp = top_risk.copy()
        if "high_risk_probability" in disp.columns:
            disp["high_risk_probability"] = disp["high_risk_probability"].map("{:.3f}".format)
        if "cancellation_rate" in disp.columns:
            disp["cancellation_rate"] = disp["cancellation_rate"].map("{:.2%}".format)
        st.dataframe(disp, width='stretch', height=380, hide_index=True)

    st.info(
        "**Business interpretation:** High-risk customers warrant product quality review, "
        "improved descriptions, enhanced support, and flexible return policies."
    )

st.markdown("---")

# ── Tier classification reference ─────────────────────────────────────────────
st.markdown("## Tier Classification System")

tier_data = {
    "Model":      ["Churn", "High-Value", "High-Risk"],
    "Low Tier":   ["< 0.40  →  Low Risk",    "< 0.40  →  Standard",     "< 0.30  →  Normal"],
    "Mid Tier":   ["0.40–0.70  →  Medium Risk", "0.40–0.70  →  Growing Potential", "0.30–0.60  →  Watch List"],
    "High Tier":  ["> 0.70  →  High Risk",   "> 0.70  →  VIP",          "> 0.60  →  Urgent Attention"],
}
st.dataframe(pd.DataFrame(tier_data), width='stretch', hide_index=True)
st.caption("Thresholds are tuned to balance precision and recall for optimal business decision-making.")