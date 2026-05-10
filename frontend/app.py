"""Customer Intelligence & Risk Prediction System - Main Application"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_data
def load_default_data():
    """Load default predictions and features."""
    pred_path = PROJECT_ROOT / "data" / "customer_predictions.csv"
    feat_path = PROJECT_ROOT / "data" / "processed" / "customer_nlp_features_with_labels.csv"
    metrics_path = PROJECT_ROOT / "stuff" / "supervised" / "results.json"

    pred_df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    feat_df = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {
            "churn": {"f1_score": 0.687, "roc_auc": 0.735},
            "high_value": {"f1_score": 0.640, "roc_auc": 0.858},
            "high_risk": {"f1_score": 0.312, "roc_auc": 0.791},
        }

    return pred_df, feat_df, metrics


st.set_page_config(
    page_title="Customer Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
    }

    .main .block-container {
        padding: 2rem 2.5rem 3rem 2.5rem;
        max-width: 1400px;
        background: #f0f2f6;
    }

    /* ── Hide Streamlit chrome ── */
    [data-testid="stSidebarNav"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header {
        visibility: visible;
        background: transparent;
    }
    [data-testid="stDecoration"] { display: none; }

    /* ── Sidebar shell ── */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b;
        min-width: 240px !important;
        max-width: 240px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 0;
    }

    /* ── Sidebar toggle arrow (hamburger button) ── */
    [data-testid="collapsedControl"] {
        color: #64748b !important;
        background: #1e293b !important;
        border-radius: 0 6px 6px 0 !important;
        top: 1.2rem !important;
        z-index: 1000 !important;
    }

    [data-testid="collapsedControl"]:hover {
        background: #334155 !important;
        color: #e2e8f0 !important;
    }

    /* ── Sidebar content ── */
    .sidebar-inner {
        padding: 1.5rem 1rem 1rem 1rem;
        height: 100vh;
        display: flex;
        flex-direction: column;
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.75rem 0.5rem 1.5rem 0.5rem;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1.5rem;
    }

    .sidebar-logo-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }

    .sidebar-logo-text {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9 !important;
        line-height: 1.2;
    }

    .sidebar-logo-sub {
        font-size: 0.7rem;
        color: #64748b !important;
        font-weight: 400;
    }

    /* ── Nav section label ── */
    .nav-section-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #475569 !important;
        padding: 0 0.5rem;
        margin-bottom: 0.4rem;
    }

    /* ── Nav buttons ── */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        text-align: left !important;
        background: transparent !important;
        border: none !important;
        color: #94a3b8 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
        padding: 0.55rem 0.75rem !important;
        border-radius: 6px !important;
        margin-bottom: 2px !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1e293b !important;
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] .stButton > button:focus {
        box-shadow: none !important;
        outline: none !important;
    }

    /* Active nav button — injected via st.markdown + data attribute trick */
    .nav-btn-active > button {
        background: #1e3a5f !important;
        color: #60a5fa !important;
        font-weight: 500 !important;
        border-left: 2px solid #3b82f6 !important;
        padding-left: calc(0.75rem - 2px) !important;
    }

    /* ── Status badge ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 20px;
    }

    .status-pill.loaded {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-pill.empty {
        background: rgba(245, 158, 11, 0.12);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0.5rem;
        font-size: 0.78rem;
    }

    .sidebar-stat-label { color: #64748b; }
    .sidebar-stat-value { color: #cbd5e1; font-weight: 500; }

    /* ── Main page headings ── */
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 1.9rem !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.2 !important;
    }

    h2 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
        margin-top: 2rem !important;
        margin-bottom: 0.75rem !important;
    }

    h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        padding: 1.25rem 1.5rem !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        line-height: 1.1 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
        color: #64748b !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #e2e8f0;
        padding: 4px;
        border-radius: 8px;
        width: fit-content;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px !important;
        padding: 0.45rem 1.1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
        border: none !important;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        overflow: hidden !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #ffffff;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        margin-bottom: 0.5rem;
    }

    [data-testid="stExpander"] summary {
        font-weight: 500 !important;
        color: #334155 !important;
        font-size: 0.9rem !important;
        padding: 0.85rem 1rem !important;
    }

    /* Expander arrow — Streamlit uses an SVG here */
    [data-testid="stExpander"] summary svg {
        color: #64748b !important;
        fill: #64748b !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 10px !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6 !important;
    }

    /* ── Buttons (primary/secondary) ── */
    .stButton > button[kind="primary"] {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.5rem !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }

    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #334155 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 7px !important;
        font-weight: 500 !important;
    }

    .stButton > button[kind="secondary"]:hover {
        border-color: #94a3b8 !important;
        background: #f8fafc !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: #f8fafc !important;
        color: #334155 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 7px !important;
        font-weight: 500 !important;
        width: 100% !important;
    }

    .stDownloadButton > button:hover {
        background: #f1f5f9 !important;
        border-color: #94a3b8 !important;
    }

    /* ── Alert / info boxes ── */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
        font-size: 0.875rem !important;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid #e2e8f0 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Page subtitle ── */
    .page-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: -0.1rem;
        margin-bottom: 1.5rem;
    }

    /* ── Stat card (custom) ── */
    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }

    .stat-card-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }

    .stat-card-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
    }

    .stat-card-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.3rem;
    }

    /* ── Info / insight card ── */
    .insight-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        height: 100%;
    }

    .insight-card h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0 0 0.75rem 0;
    }

    .insight-card ul {
        margin: 0;
        padding-left: 1.1rem;
        color: #64748b;
        font-size: 0.85rem;
        line-height: 1.7;
    }

    /* ── Plotly chart container ── */
    [data-testid="stPlotlyChart"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ── Selectbox / inputs ── */
    [data-testid="stSelectbox"] > div > div {
        border-radius: 7px !important;
        border-color: #cbd5e1 !important;
        background: white !important;
    }

    /* ── Progress bar ── */
    [data-testid="stProgress"] > div > div {
        background: #3b82f6 !important;
        border-radius: 99px !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] {
        color: #3b82f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state bootstrap ──────────────────────────────────────────────────
if "data_loaded" not in st.session_state:
    pred, feat, metrics = load_default_data()
    st.session_state.predictions_df = pred
    st.session_state.features_df = feat
    st.session_state.customer_features_df = feat
    st.session_state.metrics = metrics
    # Data is loaded from CSVs only (fast). Models are NOT loaded here.
    st.session_state.data_loaded = not pred.empty
    # Models are intentionally not loaded on startup - load on demand.
    st.session_state.models_loaded = False
    # Whether the user has processed/triggered prediction flow in this session
    st.session_state.is_processed = False
    st.session_state.data_source = "Default dataset" if not pred.empty else "No data"
    st.session_state.upload_info = {
        "rows_uploaded": int(len(pred)),
        "rows_clean": int(len(pred)),
        "customers": int(pred["customerid"].nunique()) if "customerid" in pred.columns else 0,
    }

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# Backward compat
if "features_df" not in st.session_state and "customer_features_df" in st.session_state:
    st.session_state.features_df = st.session_state.customer_features_df
if "customer_features_df" not in st.session_state and "features_df" in st.session_state:
    st.session_state.customer_features_df = st.session_state.features_df


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">📊</div>
        <div>
            <div class="sidebar-logo-text">CustomerIQ</div>
            <div class="sidebar-logo-sub">Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav
    st.markdown('<div class="nav-section-label">Navigation</div>', unsafe_allow_html=True)

    nav_items = [
        ("Home"),
        ("Data Overview"),
        ("Predictions"),
        ("Customer Segments"),
        ("Anomalies and NLP"),
        ("Model Insights")
    ]

    for label in nav_items:
        is_active = st.session_state.current_page == label
        # Wrap in a div that we can style for active state
        if is_active:
            st.markdown('<div class="nav-btn-active">', unsafe_allow_html=True)
        if st.button(f"{label}", key=f"nav_{label}", width='stretch'):
            st.session_state.current_page = label
            st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="nav-section-label">System</div>', unsafe_allow_html=True)

    # Status
    if st.session_state.data_loaded:
        customers = (
            st.session_state.predictions_df["customerid"].nunique()
            if "customerid" in st.session_state.predictions_df.columns
            else 0
        )
        st.markdown(
            '<span class="status-pill loaded">● Data Loaded</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="margin-top:0.6rem;">
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Customers</span>
                    <span class="sidebar-stat-value">{customers:,}</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Source</span>
                    <span class="sidebar-stat-value" style="font-size:0.72rem;max-width:110px;
                        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
                        {st.session_state.get('data_source','Default')}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill empty">⚠ No Data</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("↺  Reset System", key="reset_btn",width='stretch'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ── Page routing ─────────────────────────────────────────────────────────────
page = st.session_state.current_page

pages_dir = APP_ROOT / "pages"

if page == "Home":
    exec(open(pages_dir / "0_Home.py", encoding="utf-8").read())
elif page == "Data Overview":
    exec(open(pages_dir / "1_data_overview.py", encoding="utf-8").read())
elif page == "Predictions":
    exec(open(pages_dir / "2_prediction.py", encoding="utf-8").read())
elif page == "Customer Segments":
    exec(open(pages_dir / "3_Segments.py", encoding="utf-8").read())
elif page == "Anomalies and NLP":
    exec(open(pages_dir / "4_Advanced.py", encoding="utf-8").read())
elif page == "Model Insights":
    exec(open(pages_dir / "5_insights.py", encoding="utf-8").read())
