import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.cluster_plot import (
    get_cluster_profile_table as cluster_profile_table,
    plot_churn_rate_by_cluster as cluster_churn_bar,
    plot_cluster_distribution_pie as cluster_distribution_pie,
    plot_cluster_vs_tier_breakdown as cluster_tier_stacked,
)
from src.visualization.tier_plots import (
    get_segment_summary as segment_strategy_table,
    plot_segment_matrix_heatmap as segment_heatmap,
)


def csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def timestamp_filename(prefix: str) -> str:
    from datetime import datetime
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

st.title("Customer Segments")
st.caption("Cross-tier segmentation and cluster diagnostics for strategy design.")

if not st.session_state.get("is_processed", False):
    st.warning("No processed dataset available. Upload a CSV in the sidebar to begin.")
    st.stop()

pred_df = st.session_state.predictions_df.copy()
cust_df = st.session_state.customer_features_df.copy()

st.plotly_chart(segment_heatmap(pred_df), use_container_width=True)

st.divider()
st.subheader("Segment Strategy Summary")
st.dataframe(segment_strategy_table(pred_df), use_container_width=True, hide_index=True, height=280)

st.divider()
st.subheader("Cluster Analysis")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(cluster_distribution_pie(cust_df), use_container_width=True)
with c2:
    profile_df = cluster_profile_table(cust_df,pred_df)
    st.dataframe(profile_df, use_container_width=True, hide_index=True, height=430)

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(cluster_churn_bar(cust_df,pred_df), use_container_width=True)
with c4:
    st.plotly_chart(cluster_tier_stacked(cust_df,pred_df), use_container_width=True)

segmented = pred_df.merge(cust_df[["customerid", "cluster_label", "cluster_name"]], on="customerid", how="left")
st.download_button(
    label="Download Segmented Customers",
    data=csv_download(segmented),
    file_name=timestamp_filename("segmented_customers"),
    mime="text/csv",
    use_container_width=True,
)
