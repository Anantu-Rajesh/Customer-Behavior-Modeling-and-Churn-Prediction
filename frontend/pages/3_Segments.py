"""Customer Segments Page - Clustering and Segmentation Analysis"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.cluster_plot import (
    get_cluster_profile_table,
    plot_churn_rate_by_cluster,
    plot_cluster_distribution_pie,
    plot_cluster_vs_tier_breakdown,
    plot_spend_distribution_by_cluster,
    plot_product_diversity_by_cluster,
)
from src.visualization.tier_plots import (
    get_segment_summary,
    get_segment_behavior_summary_table,
    plot_segment_matrix_heatmap,
)

# ── Guard ─────────────────────────────────────────────────────────────────────
pred_df = st.session_state.get("predictions_df", pd.DataFrame())
feat_df = st.session_state.get("features_df", pd.DataFrame())

if not st.session_state.get("data_loaded") or pred_df.empty:
    st.markdown("#  Customer Segments")
    st.markdown(
        '<p class="page-subtitle">Advanced segmentation using unsupervised learning.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style='text-align:center; padding:3rem 2rem; background:white;
                border:1px solid #e2e8f0; border-radius:12px;'>
        <div style='font-size:2.5rem; margin-bottom:1rem;'></div>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:1.1rem;
                    font-weight:600; color:#475569; margin-bottom:0.5rem;'>
            No segment data loaded
        </div>
        <div style='font-size:0.875rem; color:#94a3b8;'>
            Upload your dataset on the Home page to generate customer segments.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("#  Customer Segments")
st.markdown(
    '<p class="page-subtitle">Advanced segmentation using unsupervised learning and behavioural clustering.</p>',
    unsafe_allow_html=True,
)

#  SECTION 1: Business Segments (MOST IMPORTANT FIRST)
st.markdown("##  Section 1: Business Segments")

strategy_df = get_segment_summary(pred_df)

seg_left, seg_right = st.columns([1.1, 1.9])

'''with seg_left:
    st.markdown("### Segment Share")
    if not strategy_df.empty and {"Segment", "Percentage"}.issubset(strategy_df.columns):
        chart_df = strategy_df[["Segment", "Percentage"]].set_index("Segment")
        st.bar_chart(chart_df, width="stretch")
    else:
        st.info("Segment percentage data not available.")'''

with seg_right:
    st.markdown("### Segment Strategy Table")
    if not strategy_df.empty:
        strategy_display = strategy_df.copy()
        description_map = {
            "Champions": "High value, low churn risk customers",
            " At-Risk VIPs": "VIP customers with high churn probability",
            "Growing Stars": "Rising value customers with healthy engagement",
            " Potential Churners": "Medium/high churn customers with growth potential",
            "At-Risk": "Standard value customers likely to churn",
            "Standard Active": "Stable baseline customers",
        }
        strategy_display["Description"] = strategy_display["Segment"].map(description_map).fillna("Customer behavior segment")
        strategy_display = strategy_display[["Segment", "Description", "Strategy"]]
        strategy_display.columns = ["Segment Name", "Description", "Suggested Strategy"]
        st.dataframe(strategy_display, width="stretch", hide_index=True, height="auto")
    else:
        st.info("Segment strategy data not available.")

st.markdown("---")

#  SECTION 2: Segment Behavior (MAKE THIS STRONG)
st.markdown("##  Section 2: Segment Behavior")
segment_behavior_df = get_segment_behavior_summary_table(pred_df, feat_df)
st.dataframe(segment_behavior_df, width="stretch", hide_index=True, height="auto")

st.markdown("---")

#  SECTION 3: Unsupervised Clusters (SEPARATE CLEARLY)
st.markdown("##  Section 3: Unsupervised Clusters")
st.caption("Unsupervised clustering reveals hidden behavioral groupings beyond rule-based segmentation")

if "cluster_label" in feat_df.columns:
    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("### Cluster Distribution")
        st.plotly_chart(plot_cluster_distribution_pie(feat_df), width='stretch')
    with top_right:
        st.markdown("### Cluster Summary")
        cluster_profile_df = get_cluster_profile_table(feat_df, pred_df)
        st.dataframe(cluster_profile_df, width='stretch', hide_index=True, height='auto')

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown("### Churn Rate vs Cluster")
        st.plotly_chart(plot_churn_rate_by_cluster(feat_df, pred_df), width='stretch')
    with bottom_right:
        st.markdown("### Tier Distribution by Cluster")
        st.plotly_chart(plot_cluster_vs_tier_breakdown(feat_df, pred_df), width='stretch')

    st.markdown("---")
    st.markdown("**Behavioral Profile by Cluster**")
    behav_left, behav_right = st.columns(2)
    with behav_left:
        st.markdown("### Spend Distribution by Cluster")
        st.plotly_chart(plot_spend_distribution_by_cluster(feat_df), width='stretch')
    with behav_right:
        st.markdown("### Product Diversity by Cluster")
        st.plotly_chart(plot_product_diversity_by_cluster(feat_df), width='stretch')
else:
    st.info("Cluster data not available.")

st.markdown("---")

#  SECTION 4: Cluster Interpretation Cards
st.markdown("##  Section 4: Cluster Interpretation")

seg_col1, seg_col2, seg_col3 = st.columns(3)

with seg_col1:
    st.markdown("""
    <div class="insight-card" style="border-left:3px solid #ef4444;">
        <h4 style="color:#ef4444;">One-Time Churners</h4>
        <p style="font-size:0.82rem;font-weight:600;color:#475569;margin:0 0 0.3rem;">What it means</p>
        <ul>
            <li>Low purchase frequency</li>
            <li>High recency gap</li>
        </ul>
        <p style="font-size:0.82rem;font-weight:600;color:#475569;margin:0.75rem 0 0.3rem;">Action</p>
        <ul>
            <li>Win-back campaigns</li>
            <li>Strong incentives</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with seg_col2:
    st.markdown("""
    <div class="insight-card" style="border-left:3px solid #3b82f6;">
        <h4 style="color:#3b82f6;">Engaged Regulars</h4>
        <p style="font-size:0.82rem;font-weight:600;color:#475569;margin:0 0 0.3rem;">What it means</p>
        <ul>
            <li>High purchase frequency</li>
            <li>Low recency gap</li>
        </ul>
        <p style="font-size:0.82rem;font-weight:600;color:#475569;margin:0.75rem 0 0.3rem;">Action</p>
        <ul>
            <li>VIP loyalty programmes</li>
            <li>Early product access</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with seg_col3:
    st.markdown("""
    <div class="insight-card" style="border-left:3px solid #f59e0b;">
        <h4 style="color:#f59e0b;">At-Risk Irregulars</h4>
        <p style="font-size:0.82rem;font-weight:600;color:#475569;margin:0 0 0.3rem;">What it means</p>
        <ul>
            <li>Inconsistent patterns</li>
            <li>Moderate frequency</li>
        </ul>
        <p style="font-size:0.82rem;font-weight:600;color:#475569;margin:0.75rem 0 0.3rem;">Action</p>
        <ul>
            <li>Re-engagement campaigns</li>
            <li>Personalised offers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)