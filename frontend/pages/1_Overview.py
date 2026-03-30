import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.tier_plots import (
	CHURN_COLORS,
	HIGH_RISK_COLORS,
	HIGH_VALUE_COLORS,
	get_active_customers as active_customers,
	get_at_risk_vips as at_risk_vips_table,
	get_summary_metrics as calculate_kpis,
	plot_tier_distribution_pie as tier_pie,
)


def csv_download(df):
	return df.to_csv(index=False).encode("utf-8")


def timestamp_filename(prefix: str) -> str:
	from datetime import datetime
	return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

st.title("Customer Behavior Analysis Dashboard")
st.caption("Overview of churn risk, value potential, and cancellation risk across customers.")

if not st.session_state.get("is_processed", False):
	st.warning("No processed dataset available. Upload a CSV in the sidebar to begin.")
	st.stop()

pred_df = st.session_state.predictions_df.copy()
active_df = active_customers(pred_df)
kpi = calculate_kpis(pred_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Customers Analyzed", f"{kpi['total_customers']:,}")
m2.metric("High Churn Risk", f"{kpi['high_churn_count']:,}", f"{kpi['high_churn_pct']:.1f}%")
m3.metric("VIP Customers", f"{kpi['vip_count']:,}", f"{kpi['vip_pct']:.1f}%")
m4.metric("At-Risk VIPs", f"{kpi['at_risk_vips']:,}")

st.divider()

# First row: 2 charts
c1, c2 = st.columns(2)
with c1:
	st.plotly_chart(
		tier_pie(pred_df, "churn_tier", "Churn Risk Distribution", CHURN_COLORS),
		use_container_width=True,
	)
with c2:
	st.plotly_chart(
		tier_pie(active_df, "high_value_tier", "High-Value Distribution (Active Customers)", HIGH_VALUE_COLORS),
		use_container_width=True,
	)

# Second row: 1 chart centered 
#ISSUE: The pie chart name and labels are overlapping. Need to adjust layout or font size to fix this!!!
c3 = st.columns([1, 2, 1])
with c3[1]:
	st.plotly_chart(
		tier_pie(pred_df, "high_risk_tier", "Cancellation Risk Distribution", HIGH_RISK_COLORS),
		use_container_width=True,
	)

st.divider()

st.subheader("Critical Alerts: Top At-Risk VIP Customers")
alerts_df = at_risk_vips_table(pred_df, n=10)
st.dataframe(alerts_df, use_container_width=True, height=360, hide_index=True)

st.divider()
st.download_button(
	label="Download Full Predictions",
	data=csv_download(pred_df),
	file_name=timestamp_filename("full_predictions"),
	mime="text/csv",
	use_container_width=True,
)
