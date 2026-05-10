"""Insights Page - Model Summary, Feature Importance, and Decision Logic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines import inference_pipeline as inf


@st.cache_resource
def load_models():
	"""Load trained models once for feature importance extraction."""
	return inf.load_model()


def _extract_feature_importance(model) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
	"""Return aligned feature names and importance values from a model if available."""
	feature_names = getattr(model, "feature_names_in_", None)

	if hasattr(model, "feature_importances_"):
		values = np.asarray(model.feature_importances_)
	elif hasattr(model, "coef_"):
		values = np.abs(np.asarray(model.coef_)).ravel()
	else:
		return None, None

	if feature_names is None:
		feature_names = np.array([f"feature_{i}" for i in range(len(values))], dtype=object)
	else:
		feature_names = np.asarray(feature_names, dtype=object)

	if len(feature_names) != len(values):
		n = min(len(feature_names), len(values))
		feature_names = feature_names[:n]
		values = values[:n]

	return feature_names, values


def _model_summary_card(col, title: str, f1: float, roc: float):
	with col:
		st.markdown(
			f"""
			<div style='background:white; border:1px solid #e2e8f0; border-radius:12px;
						padding:0.9rem 1rem; margin-bottom:0.75rem;'>
				<div style='font-family:"Space Grotesk",sans-serif; font-size:0.95rem;
							font-weight:600; color:#1e293b; margin-bottom:2px;'>{title}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
		m1, m2 = st.columns(2)
		m1.metric("F1", f"{f1:.3f}")
		m2.metric("ROC", f"{roc:.3f}")


pred_df = st.session_state.get("predictions_df", pd.DataFrame())
metrics = st.session_state.get("metrics", {})

if not metrics:
	metrics = inf._load_metrics()

if not st.session_state.get("data_loaded") and pred_df.empty:
	st.markdown("# Insights")
	st.markdown(
		'<p class="page-subtitle">Model evaluation and decision logic for business interpretation.</p>',
		unsafe_allow_html=True,
	)
	st.info("No data loaded yet. Upload data on Home to view insights with current run context.")

st.markdown("# Insights")
st.markdown(
	'<p class="page-subtitle">Interview-focused view of model quality, interpretability, and threshold design.</p>',
	unsafe_allow_html=True,
)

st.markdown("## Section 1: Model Summary")

churn_m = metrics.get("churn", {})
value_m = metrics.get("high_value", {})
risk_m = metrics.get("high_risk", {})

c1, c2, c3 = st.columns(3)
_model_summary_card(c1, "Churn", float(churn_m.get("f1_score", 0.0)), float(churn_m.get("roc_auc", 0.0)))
_model_summary_card(c2, "High Value", float(value_m.get("f1_score", 0.0)), float(value_m.get("roc_auc", 0.0)))
_model_summary_card(c3, "High Risk", float(risk_m.get("f1_score", 0.0)), float(risk_m.get("roc_auc", 0.0)))

st.caption("Models optimized for F1 due to class imbalance")

st.markdown("---")

st.markdown("## Section 2: Feature Importance")

try:
	loaded = load_models()
except Exception:
	loaded = {}

model_candidates = [
	("high_value", "High Value", loaded.get("high_value") if loaded else None),
	("high_risk", "High Risk", loaded.get("high_risk") if loaded else None),
	("churn", "Churn", loaded.get("churn") if loaded else None),
]

best_row = None
for key, label, model in model_candidates:
	if model is None:
		continue
	names, values = _extract_feature_importance(model)
	if names is None or values is None or len(values) == 0:
		continue
	roc = float(metrics.get(key, {}).get("roc_auc", 0.0))
	f1 = float(metrics.get(key, {}).get("f1_score", 0.0))
	rank_score = roc + f1
	row = (rank_score, key, label, names, values)
	if best_row is None or row[0] > best_row[0]:
		best_row = row

if best_row is None:
	st.warning("Feature importance is not available for the currently loaded models.")
else:
	_, best_key, best_label, feature_names, feature_values = best_row
	fi_df = pd.DataFrame(
		{
			"feature": feature_names,
			"importance": feature_values,
		}
	)
	fi_df = fi_df.sort_values("importance", ascending=False).head(10).sort_values("importance", ascending=True)

	fig = px.bar(
		fi_df,
		x="importance",
		y="feature",
		orientation="h",
		color="importance",
		color_continuous_scale="Blues",
		title=f"Top 10 Features - {best_label} Model",
	)
	fig.update_layout(
		height=420,
		showlegend=False,
		plot_bgcolor="white",
		paper_bgcolor="white",
		margin=dict(t=50, b=25, l=10, r=10),
		font=dict(family="DM Sans"),
		coloraxis_showscale=False,
	)
	fig.update_yaxes(title=None)
	fig.update_xaxes(title="Importance", gridcolor="#f1f5f9")
	st.plotly_chart(fig, width="stretch")

	st.caption(
		f"Interpretability view generated from the {best_label} model ({best_key}) using model-native importance values."
	)

st.markdown("---")