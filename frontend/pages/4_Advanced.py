"""Advanced Insights Page - Model Performance, Anomalies, and Downloads"""

from datetime import datetime
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.anomaly_plots import (
    get_anomaly_customers_table,
    get_anomaly_summary,
    plot_anomaly_distribution_by_tier,
)

# ── Guard ─────────────────────────────────────────────────────────────────────
pred_df = st.session_state.get("predictions_df", pd.DataFrame())
feat_df = st.session_state.get("features_df", pd.DataFrame())
metrics = st.session_state.get("metrics", {})

if not st.session_state.get("data_loaded") or pred_df.empty:
    st.markdown("#  Advanced Insights")
    st.markdown(
        '<p class="page-subtitle">Model performance, anomaly detection, and data exports.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style='text-align:center; padding:3rem 2rem; background:white;
                border:1px solid #e2e8f0; border-radius:12px;'>
        <div style='font-size:2.5rem; margin-bottom:1rem;'></div>
        <div style='font-family:"Space Grotesk",sans-serif; font-size:1.1rem;
                    font-weight:600; color:#475569; margin-bottom:0.5rem;'>
            No data loaded
        </div>
        <div style='font-size:0.875rem; color:#94a3b8;'>
            Upload your dataset on the Home page to view advanced insights.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

_chart_layout = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=45, b=30, l=10, r=10),
    font=dict(family="DM Sans"),
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("#  Advanced Insights")
st.markdown(
    '<p class="page-subtitle">Model performance, feature importance, anomaly detection, and data exports.</p>',
    unsafe_allow_html=True,
)

# ── Section 1: Product/NLP Insights ───────────────────────────────────────────
st.markdown("##  Section 1: Product/NLP Insights")

if "product_cluster_diversity" in feat_df.columns:
    sec1_left, sec1_right = st.columns(2)

    with sec1_left:
        fig = px.histogram(
            feat_df, x="product_cluster_diversity", nbins=30,
            title="Product Cluster Diversity",
            labels={"product_cluster_diversity": "Unique Product Clusters"},
            color_discrete_sequence=["#8b5cf6"],
        )
        fig.update_layout(showlegend=False, height=320, **_chart_layout)
        fig.update_yaxes(gridcolor="#f1f5f9")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, width='stretch')

    with sec1_right:
        if "product_cluster_entropy" in feat_df.columns:
            fig = px.histogram(
                feat_df, x="product_cluster_entropy", nbins=30,
                title="Product Cluster Entropy",
                labels={"product_cluster_entropy": "Entropy Score"},
                color_discrete_sequence=["#06b6d4"],
            )
            fig.update_layout(showlegend=False, height=320, **_chart_layout)
            fig.update_yaxes(gridcolor="#f1f5f9")
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Entropy feature not available.")

    st.markdown("Higher entropy = diverse purchasing behavior")
else:
    st.info("NLP features not available in the current dataset.")

st.markdown("---")

# ── Section 2: Anomaly Detection ─────────────────────────────────────────────
st.markdown("##  Section 2: Anomaly Detection")

summary = get_anomaly_summary(feat_df) if {"if_label"}.issubset(feat_df.columns) else {}
if_anom = int(summary.get("if_anomalies", 0))
total_anom = int(summary.get("total_anomalies", 0))
anom_pct = float(summary.get("anomaly_percentage", 0.0))

if "if_label" in feat_df.columns:
    a1, = st.columns(1)
    a1.metric("Total Anomalies",       f"{total_anom:,}", f"{anom_pct:.1f}% of customers")

    st.plotly_chart(plot_anomaly_distribution_by_tier(feat_df, pred_df), width='stretch')

    anomaly_customers_df = get_anomaly_customers_table(feat_df, pred_df, n=20)
    st.markdown("#### Top Anomalous Customers")
    st.dataframe(anomaly_customers_df, width='stretch', hide_index=True, height=320)

    st.markdown("""
    **Isolation Forest** — detects global outliers (customers very different from the overall population).  
    Anomalies may indicate fraud, data errors, or unique behavioral segments requiring further investigation.
    """)
else:
    st.info("Anomaly detection labels not available in the current dataset.")

st.markdown("---")

# ── Section 3: Downloads ─────────────────────────────────────────────────────
st.markdown("##  Section 3: Downloads")
st.caption("All current downloads are grouped here.")

# ── Downloads ─────────────────────────────────────────────────────────────────
st.markdown("## Download Datasets")

def _to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def _ts(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

dl1, dl2, dl3 = st.columns(3)

with dl1:
    st.download_button(
        " Full Predictions",
        data=_to_csv(pred_df),
        file_name=_ts("predictions"),
        mime="text/csv",
        width='stretch',
    )

with dl2:
    st.download_button(
        " Customer Features",
        data=_to_csv(feat_df),
        file_name=_ts("features"),
        mime="text/csv",
        width='stretch',
    )

with dl3:
    complete_df = (
        pred_df.merge(feat_df, on="customerid", how="left")
        if "customerid" in pred_df.columns and "customerid" in feat_df.columns
        else pred_df
    )
    st.download_button(
        " Complete Dataset",
        data=_to_csv(complete_df),
        file_name=_ts("complete_dataset"),
        mime="text/csv",
        width='stretch',
    )

with st.expander(" Segment-Specific Downloads"):
    s1, s2 = st.columns(2)

    with s1:
        if "churn_tier" in pred_df.columns:
            hc_df = pred_df[pred_df["churn_tier"] == "High Risk"]
            st.download_button(
                f" High Churn Risk  ({len(hc_df):,})",
                data=_to_csv(hc_df),
                file_name=_ts("high_churn"),
                mime="text/csv",
                width='stretch',
                key="dl_churn",
            )
        if "high_value_tier" in pred_df.columns:
            vip_df = pred_df[pred_df["high_value_tier"] == "VIP"]
            st.download_button(
                f" VIP Customers  ({len(vip_df):,})",
                data=_to_csv(vip_df),
                file_name=_ts("vip_customers"),
                mime="text/csv",
                width='stretch',
                key="dl_vip",
            )

    with s2:
        if "high_risk_tier" in pred_df.columns:
            hr_df = pred_df[pred_df["high_risk_tier"] == "Urgent Attention"]
            st.download_button(
                f" High Risk  ({len(hr_df):,})",
                data=_to_csv(hr_df),
                file_name=_ts("high_risk"),
                mime="text/csv",
                width='stretch',
                key="dl_risk",
            )
        if "if_label" in feat_df.columns and "lof_label" in feat_df.columns:
            anom_df = feat_df[(feat_df["if_label"] == 1) | (feat_df["lof_label"] == 1)]
            if "customerid" in anom_df.columns and "customerid" in pred_df.columns:
                anom_df = anom_df.merge(pred_df, on="customerid", how="left")
            st.download_button(
                f" Anomalous Customers  ({len(anom_df):,})",
                data=_to_csv(anom_df),
                file_name=_ts("anomalies"),
                mime="text/csv",
                width='stretch',
                key="dl_anomaly",
            )

st.markdown("---")