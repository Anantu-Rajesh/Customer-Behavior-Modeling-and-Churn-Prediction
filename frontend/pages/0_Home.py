"""Home page main content for upload, run flow, and dataset overviews."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_resource
def load_inference_pipeline():
    """Load inference functions and model resources once."""
    from src.pipelines import inference_pipeline as inf

    if hasattr(inf, "load_model"):
        inf.load_model()
    return inf.predict_all_customers, inf.check_ip_cols

def validate_upload_columns(df: pd.DataFrame):
    """Lightweight upload validator that doesn't import heavy pipeline code.

    This mirrors `check_ip_cols` but keeps validation local to avoid importing
    the `inference_pipeline` module (which triggers heavy imports).
    """
    required = [
        'InvoiceNo', 'StockCode', 'Description', 'Quantity',
        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'
    ]
    df_cols_lower = [c.lower() for c in df.columns]
    missing = [c for c in required if c.lower() not in df_cols_lower]
    return (len(missing) == 0, missing)

@st.cache_data
def load_training_transactions() -> pd.DataFrame:
    """Load sample/training transaction dataset.

    Attempts several common file types in `data/raw/` and returns the
    first readable dataset. Returns an empty DataFrame if none found or
    if reading fails.
    """
    base = PROJECT_ROOT / "data" / "raw" / "online_retail"
    candidates = [
        (base.with_suffix('.csv'), 'csv'),
        (base.with_suffix('.xlsx'), 'excel'),
        (base.with_suffix('.xls'), 'excel'),
        (base.with_suffix('.parquet'), 'parquet'),
    ]

    for path, kind in candidates:
        try:
            if path.exists():
                if kind == 'csv':
                    return pd.read_csv(path)
                if kind == 'excel':
                    return pd.read_excel(path)
                if kind == 'parquet':
                    return pd.read_parquet(path)
        except Exception:
            # If one format fails to read for any reason, try the next.
            continue

    return pd.DataFrame()


def _choose_customer_df() -> pd.DataFrame:
    """Get customer-level dataframe from session state."""
    return st.session_state.get(
        "features_df",
        st.session_state.get("customer_features_df", pd.DataFrame()),
    )


def _derive_avg_order_value(txn_df: pd.DataFrame, customer_df: pd.DataFrame) -> float:
    """Derive avg order value from transaction or customer level data."""
    if {"Quantity", "UnitPrice"}.issubset(txn_df.columns):
        txn_vals = pd.to_numeric(txn_df["Quantity"], errors="coerce") * pd.to_numeric(
            txn_df["UnitPrice"], errors="coerce"
        )
        txn_vals = txn_vals.dropna()
        if not txn_vals.empty:
            return float(txn_vals.mean())

    if "avg_order_val" in customer_df.columns:
        vals = pd.to_numeric(customer_df["avg_order_val"], errors="coerce").dropna()
        if not vals.empty:
            return float(vals.mean())
    return 0.0


# 1) Title + tagline
st.markdown("# Customer Intelligence & Risk Prediction")
st.markdown(
    "<p class='page-subtitle'>"
    "ML-powered analytics workspace to evaluate churn risk, value potential, and cancellation behavior."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# 2) Upload section + 3) two action buttons
st.markdown("## Upload & Run")
upload_col, run_col, sample_col = st.columns([3.2, 1.2, 1.4])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload transactional CSV",
        type=["csv"],
        help=(
            "Required columns: CustomerID, InvoiceNo, StockCode, Description, "
            "Quantity, InvoiceDate, UnitPrice, Country"
        ),
    )

with run_col:
    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
    run_analysis_clicked = st.button(
        "Process Data",
        type="primary",
        width="stretch",
        disabled=(uploaded_file is None),
    )

with sample_col:
    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
    sample_results_clicked = st.button("Use Sample Data", width="stretch")

if run_analysis_clicked:
    if uploaded_file is None:
        st.warning("Please upload a CSV before running analysis.")
    else:
        try:
            incoming_df = pd.read_csv(uploaded_file)
            valid, missing = validate_upload_columns(incoming_df)
            if not valid:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                # Load models on demand (one-time) and then run predictions.
                if not st.session_state.get("models_loaded", False):
                    with st.spinner("Loading ML models (one-time setup)..."):
                        predict_fn, _ = load_inference_pipeline()
                        st.session_state.models_loaded = True
                else:
                    predict_fn, _ = load_inference_pipeline()

                with st.spinner("Running inference pipeline on uploaded file..."):
                    progress = st.progress(0)
                    progress.progress(25, "Validating and cleaning data...")
                    result = predict_fn(incoming_df)
                    progress.progress(85, "Finalizing predictions...")

                    st.session_state.predictions_df = result["predictions_df"]
                    st.session_state.features_df = result["customer_features_df"]
                    st.session_state.customer_features_df = result["customer_features_df"]
                    st.session_state.metrics = result.get("metrics", {})
                    st.session_state.data_loaded = True
                    st.session_state.is_processed = True
                    st.session_state.upload_info = result.get("upload_status", {})
                    st.session_state.data_source = uploaded_file.name

                    st.session_state.home_txn_df = incoming_df
                    st.session_state.home_view_mode = "uploaded"

                    progress.progress(100, "Done")
                st.success("Analysis completed using uploaded dataset.")
        except Exception as exc:
            st.error(f"Failed to process uploaded file: {exc}")

if sample_results_clicked:
    training_txn_df = load_training_transactions()
    # If raw transactions not available or unreadable, fall back to processed
    # features/predictions that are loaded at startup.
    if training_txn_df.empty:
        feat_df = st.session_state.get("customer_features_df", pd.DataFrame())
        pred_df = st.session_state.get("predictions_df", pd.DataFrame())
        if feat_df.empty and pred_df.empty:
            st.warning(
                "Sample/training transaction file not found. Place online_retail.csv/.xlsx in data/raw/ or provide processed feature CSVs."
            )
        else:
            # Use processed CSVs already loaded by app.py
            st.session_state.home_txn_df = training_txn_df
            st.session_state.home_view_mode = "training"
            st.session_state.is_processed = True
            st.success("Sample data selected from processed features. Showing sample results.")
    else:
        # Use CSVs already loaded by app.py (predictions + features).
        st.session_state.home_txn_df = training_txn_df
        st.session_state.home_view_mode = "training"
        # Mark processed so other pages/cards render sample stats.
        st.session_state.is_processed = True
        st.success("Sample data selected. Showing sample results.")

    '''# Show metrics if available
    metrics = st.session_state.get("metrics", {})
    if metrics:
        metric_df = pd.DataFrame(
            {
                "Model": ["Churn", "High Value", "High Risk"],
                "F1 Score": [
                    metrics.get("churn", {}).get("f1_score", 0.0),
                    metrics.get("high_value", {}).get("f1_score", 0.0),
                    metrics.get("high_risk", {}).get("f1_score", 0.0),
                ],
                "ROC-AUC": [
                    metrics.get("churn", {}).get("roc_auc", 0.0),
                    metrics.get("high_value", {}).get("roc_auc", 0.0),
                    metrics.get("high_risk", {}).get("roc_auc", 0.0),
                ],
            }
        )
        st.markdown("### Training Result Snapshot")
        st.dataframe(metric_df, width="stretch", hide_index=True)'''

if st.session_state.get("is_processed", False):
    st.markdown("---")

    # Determine active view dataset
    active_txn_df = st.session_state.get("home_txn_df", pd.DataFrame())
    if active_txn_df.empty:
        active_txn_df = load_training_transactions()

    active_customer_df = _choose_customer_df()
    active_pred_df = st.session_state.get("predictions_df", pd.DataFrame())

    # 5) Quick cards
    total_transactions = int(len(active_txn_df))
    if "customerid" in active_customer_df.columns:
        total_customers = int(active_customer_df["customerid"].nunique())
    elif "customerid" in active_pred_df.columns:
        total_customers = int(active_pred_df["customerid"].nunique())
    elif "CustomerID" in active_txn_df.columns:
        total_customers = int(active_txn_df["CustomerID"].nunique())
    else:
        total_customers = 0

    avg_order_value = _derive_avg_order_value(active_txn_df, active_customer_df)

    card1, card2, card3 = st.columns(3)
    card1.metric("Total Customers", f"{total_customers:,}")
    card2.metric("Total Transactions", f"{total_transactions:,}")
    card3.metric("Avg Order Value", f"£{avg_order_value:,.2f}")

    st.markdown("---")

    # 6 & 7) Two-column dataset overview section
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("### Transactional Dataset Overview")
        if active_txn_df.empty:
            st.info("No transactional dataset selected yet.")
        else:
            n_txn = len(active_txn_df)
            missing = int(active_txn_df.isna().sum().sum())

            date_col = "InvoiceDate" if "InvoiceDate" in active_txn_df.columns else "invoicedate"
            if date_col in active_txn_df.columns:
                date_series = pd.to_datetime(active_txn_df[date_col], errors="coerce")
                min_date = date_series.min()
                max_date = date_series.max()
                date_range_text = f"{min_date} to {max_date}" if pd.notna(min_date) and pd.notna(max_date) else "N/A"
            else:
                date_range_text = "N/A"

            st.write(f"Transactions: {n_txn:,}")
            st.write(f"Missing Values: {missing:,}")
            st.write(f"Date Range: {date_range_text}")
            st.markdown("#### Sample Rows")
            st.dataframe(active_txn_df.head(10), width="stretch", hide_index=True, height=260)

    with right_col:
        st.markdown("### Customer-Level Dataset Overview")
        if active_customer_df.empty:
            st.info("Customer-level dataset is not available yet.")
        else:
            customer_count = (
                int(active_customer_df["customerid"].nunique()) if "customerid" in active_customer_df.columns else len(active_customer_df)
            )

            purchase_col = "total_purchase" if "total_purchase" in active_customer_df.columns else None
            if purchase_col:
                purchase_series = pd.to_numeric(active_customer_df[purchase_col], errors="coerce")
                avg_purchase = float(purchase_series.mean()) if not purchase_series.dropna().empty else 0.0
                max_purchase = float(purchase_series.max()) if not purchase_series.dropna().empty else 0.0
                min_purchase = float(purchase_series.min()) if not purchase_series.dropna().empty else 0.0
            else:
                avg_purchase = 0.0
                max_purchase = 0.0
                min_purchase = 0.0

            st.write(f"Total Customers: {customer_count:,}")
            st.write(f"Avg Purchase: £{avg_purchase:,.2f}")
            st.write(f"Max Purchase: £{max_purchase:,.2f}")
            st.write(f"Min Purchase: £{min_purchase:,.2f}")
            st.markdown("#### Sample Rows")
            st.dataframe(active_customer_df.head(10), width="stretch", hide_index=True, height=260)

    st.markdown("---")

    # 8) Prompt to continue
    st.success("Ready to continue: go to the next page from the sidebar (Data Overview) to explore deeper insights.")
else:
    st.markdown("---")
    st.info("Upload a transactional CSV and click 'Process Data', or click 'Use Sample Data' to load sample results (no models are loaded until you process).")