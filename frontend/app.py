import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines import inference_pipeline as inf


def apply_global_style() -> str:
        return """
<style>
:root {
    --primary: #3498db;
    --success: #2ecc71;
    --warning: #f39c12;
    --danger: #e74c3c;
    --text: #2c3e50;
    --card: #f8f9fa;
}
html, body, [class*="css"]  {
    font-family: "Trebuchet MS", sans-serif;
    color: var(--text);
    font-size: 16px;
}
h1, h2, h3, h4 {
    font-weight: 700;
    color: var(--text);
}
h1 { font-size: 32px; }
h2, h3 { font-size: 24px; }
.stDataFrame, .stTable { font-size: 14px; }
[data-testid="stMetric"] {
    background-color: var(--card);
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
</style>
"""

st.set_page_config(
    page_title="Customer Behavior Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(apply_global_style(), unsafe_allow_html=True)


@st.cache_data
def load_default_predictions() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    pred_path = Path("data/customer_predictions.csv")
    feat_path = Path("data/processed/customer_nlp_features_with_labels.csv")
    metrics_path = Path("stuff/supervised/results.json")

    pred_df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    feat_df = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()

    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = {
            "churn": {"f1_score": 0.687, "roc_auc": 0.735},
            "high_value": {"f1_score": 0.640, "roc_auc": 0.858},
            "high_risk": {"f1_score": 0.312, "roc_auc": 0.791},
        }

    return pred_df, feat_df, metrics


if "is_processed" not in st.session_state:
    default_pred, default_feat, default_metrics = load_default_predictions()
    st.session_state.predictions_df = default_pred
    st.session_state.customer_features_df = default_feat
    st.session_state.metrics = default_metrics
    st.session_state.is_processed = not default_pred.empty and not default_feat.empty
    st.session_state.upload_status = {}
    st.session_state.warnings = []
    st.session_state.data_source = "Default repository dataset"


with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

    if uploaded_file is not None:
        incoming = pd.read_csv(uploaded_file)
        valid, missing_cols = inf.check_ip_cols(incoming)

        if not valid:
            st.error(
                "Missing required columns: " + ", ".join(missing_cols)
            )
        else:
            st.success(
                f"Upload validated. Rows: {len(incoming):,}. "
                f"Date range: {pd.to_datetime(incoming['InvoiceDate'], errors='coerce').min()} to "
                f"{pd.to_datetime(incoming['InvoiceDate'], errors='coerce').max()}"
            )

            if st.button("Process Uploaded Data", use_container_width=True):
                progress = st.progress(0)
                with st.spinner("Processing data and generating predictions..."):
                    progress.progress(10)
                    result = inf.predict_all_customers(incoming)
                    progress.progress(85)

                    st.session_state.predictions_df = result["predictions_df"]
                    st.session_state.customer_features_df = result["customer_features_df"]
                    st.session_state.metrics = result["metrics"]
                    st.session_state.warnings = result["warnings"]
                    st.session_state.upload_status = result["upload_status"]
                    st.session_state.is_processed = True
                    st.session_state.data_source = uploaded_file.name
                    progress.progress(100)
                st.success("Processing complete. Dashboard refreshed with uploaded data.")

    st.divider()
    st.subheader("System Status")
    st.write(f"Models loaded: {'Yes' if st.session_state.is_processed else 'No'}")
    st.write(f"Data source: {st.session_state.data_source}")

    if st.session_state.upload_status:
        st.write(f"Rows processed: {st.session_state.upload_status.get('rows_clean', 'N/A')}")
        st.write(f"Customers: {st.session_state.upload_status.get('customers', 'N/A')}")
        st.write(f"Features created: {st.session_state.upload_status.get('features_created', 'N/A')}")

    if st.session_state.warnings:
        st.divider()
        st.subheader("Data Quality Warnings")
        for warning in st.session_state.warnings:
            st.warning(warning)

    st.divider()
    if st.button("Clear Cached Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared.")


overview = st.Page("pages/1_Overview.py", title="Overview")
churn = st.Page("pages/2_Churn_Analysis.py", title="Churn Analysis")
high_value = st.Page("pages/3_High_Value.py", title="High-Value Customers")
segments = st.Page("pages/4_Segments.py", title="Customer Segments")
advanced = st.Page("pages/5_Advanced.py", title="Advanced Analytics")

navigator = st.navigation([overview, churn, high_value, segments, advanced], position="sidebar")
navigator.run()
