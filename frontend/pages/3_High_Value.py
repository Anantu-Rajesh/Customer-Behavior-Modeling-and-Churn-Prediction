import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.tier_plots import (
    HIGH_VALUE_COLORS,
    get_active_customers as active_customers,
    get_top_vip_customers as vip_customers_table,
    plot_churn_vs_value_scatter as churn_vs_value_scatter,
    plot_tier_distribution_pie as tier_pie,
)


def csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def timestamp_filename(prefix: str) -> str:
    from datetime import datetime
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

st.title("High-Value Customers")
st.caption("Prioritize retention and growth decisions for valuable customer segments.")

if not st.session_state.get("is_processed", False):
    st.warning("No processed dataset available. Upload a CSV in the sidebar to begin.")
    st.stop()

pred_df = st.session_state.predictions_df.copy()
cust_df = st.session_state.customer_features_df.copy()
active_df = active_customers(pred_df)

st.plotly_chart(
    tier_pie(active_df, "high_value_tier", "High-Value Tier Distribution (Active Customers)", HIGH_VALUE_COLORS),
    use_container_width=True,
)

st.divider()
st.plotly_chart(churn_vs_value_scatter(pred_df), use_container_width=True)

st.divider()
st.subheader("Top VIP Customers")
vip_df = vip_customers_table(pred_df, n=20)
st.dataframe(vip_df, use_container_width=True, height=420, hide_index=True)

st.divider()
vip_active = active_df[active_df["high_value_tier"] == "VIP"]
at_risk_vips = vip_active[vip_active["churn_tier"] == "High Risk"]

revenue_at_risk = 0.0
if not at_risk_vips.empty and "total_purchase" in cust_df.columns:
    revenue_at_risk = (
        at_risk_vips[["customerid"]]
        .merge(cust_df[["customerid", "total_purchase"]], on="customerid", how="left")["total_purchase"]
        .fillna(0)
        .sum()
    )

st.info(
    f"VIP Retention Insights: High churn VIP count = {len(at_risk_vips)} | "
    f"Potential revenue at risk (historical spend proxy) = GBP {revenue_at_risk:,.2f}"
)

st.download_button(
    label="Download VIP Customers",
    data=csv_download(vip_df),
    file_name=timestamp_filename("vip_customers"),
    mime="text/csv",
    use_container_width=True,
)
