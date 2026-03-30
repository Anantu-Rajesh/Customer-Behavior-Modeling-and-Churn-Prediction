import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.anomaly_plots import (
    get_anomaly_customers_table as anomaly_table,
    get_anomaly_summary,
    plot_anomaly_distribution_by_tier as anomaly_by_churn_tier,
)


def anomaly_summary(cust_df):
    raw = get_anomaly_summary(cust_df)
    return {
        "total_anomalies": raw.get("total_anomalies", 0),
        "anomaly_pct": raw.get("anomaly_percentage", 0),
        "if_count": raw.get("if_anomalies", 0),
        "lof_count": raw.get("lof_anomalies", 0),
    }

st.title("Advanced Analytics")
st.caption("Anomaly diagnostics, model quality tracking, and methodology guidance.")

if not st.session_state.get("is_processed", False):
    st.warning("No processed dataset available. Upload a CSV in the sidebar to begin.")
    st.stop()

pred_df = st.session_state.predictions_df.copy()
cust_df = st.session_state.customer_features_df.copy()
metrics = st.session_state.metrics

summary = anomaly_summary(cust_df)

m1, m2, m3 = st.columns(3)
m1.metric("Total Anomalies", f"{summary['total_anomalies']:,}", f"{summary['anomaly_pct']:.2f}%")
m2.metric("Isolation Forest Anomalies", f"{summary['if_count']:,}")
m3.metric("Local Outlier Factor Anomalies", f"{summary['lof_count']:,}")

st.divider()

#ISSUE: Logic problem, no anomalous customers being shown in the table, gotta fix it bbg!!!
st.subheader("Top Anomalous Customers")
st.dataframe(anomaly_table(pred_df, cust_df, n=20), use_container_width=True, hide_index=True, height=420)

st.divider()
st.plotly_chart(anomaly_by_churn_tier(cust_df, pred_df), use_container_width=True)

st.divider()
st.subheader("Model Performance")
c1, c2, c3 = st.columns(3)

churn = metrics.get("churn", {})
hv = metrics.get("high_value", {})
hr = metrics.get("high_risk", {})

c1.metric("Churn F1", f"{churn.get('f1_score', 0):.3f}")
c1.metric("Churn ROC-AUC", f"{churn.get('roc_auc', 0):.3f}")

c2.metric("High-Value F1", f"{hv.get('f1_score', 0):.3f}")
c2.metric("High-Value ROC-AUC", f"{hv.get('roc_auc', 0):.3f}")

c3.metric("High-Risk F1", f"{hr.get('f1_score', 0):.3f}")
c3.metric("High-Risk ROC-AUC", f"{hr.get('roc_auc', 0):.3f}")

st.divider()
st.subheader("Methodology")
st.markdown(
    """
1. Pipeline overview: Unsupervised labels are generated first, then NLP product features, then supervised prediction models.
2. Tiers: Probability bands map to operational labels for churn, high-value potential, and cancellation risk.
3. Interpretation: Higher probabilities indicate stronger model confidence in the positive class for each target.
4. Actions: Prioritize At-Risk VIPs first, then Growing Potential churners, then monitor Standard low-value cohorts.
"""
)
