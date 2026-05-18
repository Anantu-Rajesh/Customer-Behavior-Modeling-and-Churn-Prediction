"""Insights Page - Model Summary, Pipeline Design, and Decision Logic."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines import inference_pipeline as inf

pred_df = st.session_state.get("predictions_df", pd.DataFrame())
metrics = st.session_state.get("metrics", {})

if not metrics:
    metrics = inf._load_metrics()

st.markdown("# Insights")
st.markdown(
    '<p class="page-subtitle">Pipeline design, model decisions, and evaluation rationale.</p>',
    unsafe_allow_html=True,
)

if not st.session_state.get("data_loaded") and pred_df.empty:
    st.info("No data loaded yet. Upload data on Home to view insights with current run context.")

st.markdown("---")

# ── Section 1: Model Summary ──────────────────────────────────────────────────
st.markdown("## Section 1: Model Summary")

churn_m  = metrics.get("churn", {})
value_m  = metrics.get("high_value", {})
risk_m   = metrics.get("high_risk", {})

model_meta = [
    {
        "label": "Churn",
        "model": "Soft-Vote Ensemble",
        "components": "Naive Bayes · SVM · Random Forest · XGBoost",
        "f1": float(churn_m.get("f1_score", 0.687)),
        "roc": float(churn_m.get("roc_auc", 0.735)),
        "threshold": "0.38",
        "note": "Threshold tuned via StratifiedKFold CV for F1 optimisation",
        "accent": "#3b82f6",
    },
    {
        "label": "High Value",
        "model": "XGBoost",
        "components": "Trained on non-churned customers only",
        "f1": float(value_m.get("f1_score", 0.640)),
        "roc": float(value_m.get("roc_auc", 0.858)),
        "threshold": "0.45",
        "note": "High ROC-AUC indicates strong customer ranking quality",
        "accent": "#8b5cf6",
    },
    {
        "label": "High Risk",
        "model": "XGBoost",
        "components": "Trained on baseline features (no unsupervised labels)",
        "f1": float(risk_m.get("f1_score", 0.312)),
        "roc": float(risk_m.get("roc_auc", 0.791)),
        "threshold": "0.30 / 0.60",
        "note": "Low F1 reflects class imbalance, not poor discrimination",
        "accent": "#ef4444",
    },
]

cols = st.columns(3)
for col, m in zip(cols, model_meta):
    with col:
        st.markdown(
            f"""
            <div style='background:white; border:1px solid #e2e8f0;
                        border-top:3px solid {m["accent"]}; border-radius:10px;
                        padding:1rem 1.1rem; margin-bottom:0.5rem;'>
                <div style='font-size:0.75rem; font-weight:600; color:{m["accent"]};
                            text-transform:uppercase; letter-spacing:0.05em;
                            margin-bottom:2px;'>{m["label"]}</div>
                <div style='font-size:1rem; font-weight:700; color:#1e293b;
                            margin-bottom:2px;'>{m["model"]}</div>
                <div style='font-size:0.75rem; color:#94a3b8;
                            margin-bottom:0.75rem;'>{m["components"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("F1", f"{m['f1']:.3f}")
        m2.metric("ROC-AUC", f"{m['roc']:.3f}")
        st.caption(f"Threshold: {m['threshold']} — {m['note']}")

st.caption("F1 and ROC-AUC are prioritised over accuracy due to class imbalance across all three targets.")

st.markdown("---")

# ── Section 2: Pipeline Design ────────────────────────────────────────────────
st.markdown("## Section 2: Pipeline Design")
st.caption("How raw transactions become customer-level predictions.")

st.markdown(
    """
    <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
                padding:1.25rem 1.5rem; font-size:0.875rem; color:#334155;
                line-height:2.2; font-family:monospace;'>
    Raw Transactions (541k rows)
    &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;Clean &amp; normalise &nbsp;(drop nulls, duplicates, invalid prices, standardise cancellation semantics)
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;Temporal split &nbsp;(all features from ≤ reference date · all labels from &gt; reference date)
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;Customer aggregation &nbsp;(23 behavioural + RFM features per customer)
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;NLP product embeddings &nbsp;(MiniLM-L6 → UMAP → KMeans k=14 → 4 diversity/entropy features)
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;Unsupervised layer &nbsp;(KMeans k=3 segments · Isolation Forest anomaly scores)
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;Supervised layer &nbsp;(churn · high-value · high-risk classifiers, trained on separate feature sets)
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;Threshold tuning &nbsp;(CV-optimised per task)
    <br>
    &nbsp;&nbsp;&nbsp;&nbsp;→ &nbsp;Tier assignment &nbsp;(probabilities → business action tiers)
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Section 3: Data Routing ───────────────────────────────────────────────────
st.markdown("## Section 3: What Data Each Model Sees")
st.caption("Feature sets differ by task — this was an intentional, evidence-based design choice.")

routing = pd.DataFrame([
    {
        "Model": "Churn",
        "Training Population": "All customers",
        "Feature Set": "23 behavioural + 4 NLP + cluster label + IF anomaly score",
        "Preprocessing": "Log transform + StandardScaler (linear); unscaled (tree)",
    },
    {
        "Model": "High Value",
        "Training Population": "Non-churned customers only",
        "Feature Set": "23 behavioural + 4 NLP + cluster label + IF anomaly score",
        "Preprocessing": "Unscaled (tree model)",
    },
    {
        "Model": "High Risk",
        "Training Population": "All customers",
        "Feature Set": "23 behavioural features only (baseline)",
        "Preprocessing": "Unscaled (tree model)",
    },
])
st.dataframe(routing, width="stretch", hide_index=True)

st.markdown(
    """
    **Why High Risk uses baseline features only:** A systematic ablation study compared baseline
    vs unsupervised-augmented feature sets across all three tasks. For high-risk prediction,
    the unsupervised labels added noise rather than signal — the cluster and anomaly features
    encode general behavioural deviation, which does not correlate cleanly with future cancellation
    behaviour. The baseline feature set generalised better on held-out data.
    """
)

st.markdown("---")

# ── Section 4: Evaluation Rationale ──────────────────────────────────────────
st.markdown("## Section 4: Evaluation Rationale")

e1, e2, e3 = st.columns(3)

with e1:
    st.markdown("**Why F1 over Accuracy**")
    st.markdown(
        """
        All three targets are class-imbalanced. A model that always predicts the majority class
        achieves high accuracy but is operationally useless. F1 balances precision and recall
        for the minority class and directly reflects business utility.
        """
    )

with e2:
    st.markdown("**Why ROC-AUC alongside F1**")
    st.markdown(
        """
        ROC-AUC measures ranking quality independent of any single threshold. It answers:
        "does the model correctly order customers by risk?" — useful when the operating threshold
        may shift based on business context. A model can have low F1 but high ROC-AUC and still
        be highly actionable via tier-based triage.
        """
    )

with e3:
    st.markdown("**Why threshold tuning**")
    st.markdown(
        """
        The default 0.5 threshold is arbitrary and typically underperforms on imbalanced data.
        Thresholds were tuned via StratifiedKFold cross-validation to maximise F1 on training
        folds, then evaluated on the held-out test set. High-risk tuning showed no improvement,
        which is expected — CV-optimised thresholds tend to overfit under extreme imbalance.
        """
    )

st.markdown("---")

# ── Section 5: Key Design Decisions ──────────────────────────────────────────
st.markdown("## Section 5: Key Design Decisions")

decisions = [
    ("Temporal leakage prevention",
     "A reference-date framework ensures all model inputs are derived from pre-reference transactions and all labels from post-reference activity. This mirrors real-world deployment where the model scores customers before outcomes are known."),
    ("LOF removed from final pipeline",
     "Local Outlier Factor was evaluated alongside Isolation Forest but excluded from the final pipeline. Empirical analysis showed it consistently flagged legitimate high-spend VIP customers as anomalies due to their density separation from the population — a known failure mode of density-based methods when genuine high-value outliers exist."),
    ("Ensemble for churn, single model for others",
     "Churn is the most nuanced prediction task — customers stop purchasing for varied reasons (price sensitivity, lifecycle stage, product fit). An ensemble of diverse model families captures different signal types. High-value and high-risk have cleaner decision boundaries that a single well-tuned XGBoost handles effectively."),
    ("Separate feature sets per task",
     "High-value is trained only on non-churned customers because a churned customer's future value is zero by definition — including them would introduce a confounded label. High-risk uses baseline features because unsupervised labels (cluster, anomaly) added noise rather than signal, confirmed by ablation."),
]

for title, body in decisions:
    with st.expander(title):
        st.markdown(body)