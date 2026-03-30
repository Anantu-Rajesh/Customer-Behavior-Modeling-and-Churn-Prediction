import pandas as pd
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.behav_plots import (
    plot_recency_by_tier_violin as recency_violin,
    plot_spend_by_tier_boxplot as spend_boxplot,
)
from src.visualization.tier_plots import (
    get_top_at_risk_customers as at_risk_customers_table,
    plot_churn_probability_histogram as churn_histogram,
)


def csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def timestamp_filename(prefix: str) -> str:
    from datetime import datetime
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

st.title("Churn Analysis")
st.caption("Risk concentration, spend behavior, recency patterns, and recommended intervention actions.")

if not st.session_state.get("is_processed", False):
    st.warning("No processed dataset available. Upload a CSV in the sidebar to begin.")
    st.stop()

pred_df = st.session_state.predictions_df.copy()
cust_df = st.session_state.customer_features_df.copy()

left, right = st.columns([2, 1])
with left:
    selected_tiers = st.multiselect(
        "Filter by churn tiers",
        options=["Low Risk", "Medium Risk", "High Risk"],
        default=["Low Risk", "Medium Risk", "High Risk"],
    )
with right:
    customer_search = st.text_input("Customer ID Search")

filtered = pred_df[pred_df["churn_tier"].isin(selected_tiers)].copy()
if customer_search.strip():
    filtered = filtered[filtered["customerid"].astype(str).str.contains(customer_search.strip())]

st.plotly_chart(churn_histogram(filtered), use_container_width=True)

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(spend_boxplot(filtered, cust_df), use_container_width=True)
with col2:
    st.plotly_chart(recency_violin(filtered, cust_df), use_container_width=True)

st.divider()
st.subheader("Top At-Risk Customers")
threshold = st.slider("High Risk Threshold", min_value=0.50, max_value=0.95, value=0.70, step=0.01)
atrisk = filtered[filtered["churn_probability"] >= threshold].copy()
if atrisk.empty:
    st.info("No customers meet the selected threshold.")
    table_df = pd.DataFrame()
else:
    table_df = at_risk_customers_table(atrisk, n=20)
    st.dataframe(table_df, use_container_width=True, height=420, hide_index=True)

st.download_button(
    label="Download At-Risk Customers",
    data=csv_download(table_df if not table_df.empty else atrisk),
    file_name=timestamp_filename("at_risk_customers"),
    mime="text/csv",
    use_container_width=True,
)
