import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from src.models.churn import _ensemble_predict_proba
from sklearn.metrics import roc_auc_score
from src.models import util
from src.pipelines import inference_pipeline as inf


PLOT_BG = '#ffffff'
PAPER_BG = '#fbfdff'
GRID = '#e8eef5'
TEXT = '#1f2937'
MUTED = '#64748b'


def _apply_theme(fig, title, height=460, showlegend=False):
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', x=0.02, xanchor='left'),
        template='plotly_white',
        height=height,
        showlegend=showlegend,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        margin=dict(t=70, b=40, l=15, r=15),
        font=dict(family='Trebuchet MS, sans-serif', color=TEXT, size=13),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def _load_metrics():
    metrics_path = Path('stuff/supervised/results.json')
    if metrics_path.exists():
        with metrics_path.open('r', encoding='utf-8') as handle:
            return json.load(handle)
    return {}


def _prepare_features(customer_features_df: pd.DataFrame, target: str, model_type: str):
    df = customer_features_df.copy()
    drop_cols = [
        'customerid',
        'high_value_customer',
        'high_future_cancellation',
        'first_purchase_date',
        'last_purchase_date',
        'last_cancel_date',
        'cluster_name',
    ]
    label_cols = ['cluster_label', 'if_label', 'lof_label']

    y = df[target] if target in df.columns else None

    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    if target in df.columns:
        df = df.drop(columns=[target])

    models = inf.load_model()

    if model_type == 'linear':
        df = util.prepare_for_inference(df, models, model_key='churn', model_type='linear')

    for col in label_cols:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df, y


def _top_feature_frame(feature_names, values, top_n=10):
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': values})
    fi_df = fi_df.sort_values('importance', ascending=False).head(top_n).sort_values('importance', ascending=True)
    return fi_df


def _plot_top_features(fi_df, title):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fi_df['importance'],
        y=fi_df['feature'],
        orientation='h',
        marker=dict(
            color=fi_df['importance'],
            colorscale='Blues',
            showscale=False,
            line=dict(color='white', width=1),
        ),
        hovertemplate='%{y}<br>Importance: %{x:.5f}<extra></extra>',
    ))
    fig.update_traces(showlegend=False)
    fig.update_xaxes(title='Importance', title_standoff=12, automargin=True)
    fig.update_yaxes(title=None, automargin=True, ticks='outside')

    height = max(420, 42 * len(fi_df) + 120)
    fig.update_layout(margin=dict(t=80, b=50, l=220, r=30))
    return _apply_theme(fig, title, height=height, showlegend=False)


def _tree_feature_importance(X, model_key, target, title, customer_features_df, top_n=10):
    models = inf.load_model()
    model = models[model_key]
    y = customer_features_df[target] if target in customer_features_df.columns else None

    feature_names = getattr(model, 'feature_names_in_', None)
    if feature_names is None:
        feature_names = np.array(X.columns, dtype=object)
    else:
        feature_names = np.asarray(feature_names, dtype=object)

    if hasattr(model, 'feature_importances_'):
        values = np.asarray(model.feature_importances_)
    else:
        if y is None:
            raise ValueError(f'Missing target column: {target}')
        sample_size = min(len(X), 1000)
        sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
        y_sample = y.loc[sample.index]
        result = permutation_importance(model, sample, y_sample, n_repeats=5, random_state=42, scoring='roc_auc')
        values = np.asarray(result.importances_mean)

    if len(feature_names) != len(values):
        n = min(len(feature_names), len(values))
        feature_names = feature_names[:n]
        values = values[:n]

    fi_df = _top_feature_frame(feature_names, values, top_n=top_n)
    return _plot_top_features(fi_df, title)


def plot_feature_importance_churn(customer_features_df, X_tree):    
    # Per-model permutation importance for churn ensemble
    X, y = _prepare_features(customer_features_df, target='churn', model_type='linear')
    models = inf.load_model()
    ensemble = models['churn']

    has_valid_labels = y is not None and pd.Series(y).nunique(dropna=True) >= 2

    sample_size = min(len(X), 1000)
    X = X.reset_index(drop=True)
    X_tree = X_tree.reset_index(drop=True)
    y = y.reset_index(drop=True)
    sample = X.sample(sample_size, random_state=42).sort_index() if len(X) > sample_size else X.copy()
    X_tree_sample = X_tree.loc[sample.index].copy()
    y_sample = y.loc[sample.index].copy()

    model_keys = ['nb', 'svm', 'rf', 'xgb']

    feature_names = np.asarray(sample.columns, dtype=object)

    # If labels are not available (or are single-class), compute label-free importance:
    # mean absolute change in ensemble predicted probability when permuting the feature
    if not has_valid_labels:
        baseline_probs = _ensemble_predict_proba(ensemble, sample, X_tree_sample)
        values = []
        for col in feature_names:
            X_perm = sample.copy()
            X_tree_perm = X_tree_sample.copy()
            permuted = False
            if col in X_perm.columns:
                X_perm[col] = np.random.permutation(X_perm[col].values)
                permuted = True
            if col in X_tree_perm.columns:
                X_tree_perm[col] = np.random.permutation(X_tree_perm[col].values)
                permuted = True

            if not permuted:
                values.append(0.0)
                continue

            perm_probs = _ensemble_predict_proba(ensemble, X_perm, X_tree_perm)
            values.append(np.mean(np.abs(baseline_probs - perm_probs)))

        values = np.array(values)
        fi_df = _top_feature_frame(feature_names, values, top_n=10)
        return _plot_top_features(fi_df, 'Top 10 Churn Drivers (mean |Δprob|)')

    # -- below: when valid labels exist, use per-model AUC-drop implementation --
    # compute per-model baseline AUCs on their respective inputs
    per_model_baseline = {}
    for k in model_keys:
        model = ensemble.get(k)
        if model is None:
            per_model_baseline[k] = 0.0
            continue
        if k in ['nb', 'svm']:
            X_in = sample.copy()
        else:
            X_in = X_tree_sample.copy()

        feature_names_model = getattr(model, 'feature_names_in_', None)
        if feature_names_model is not None:
            for c in feature_names_model:
                if c not in X_in.columns:
                    X_in[c] = 0
            X_in = X_in[list(feature_names_model)]

        try:
            probs = model.predict_proba(X_in)[:, 1]
            per_model_baseline[k] = roc_auc_score(y_sample, probs)
        except Exception:
            per_model_baseline[k] = 0.0

    # compute per-feature, per-model permutation drop (only positive drops)
    per_model_drops = {k: np.zeros(len(feature_names), dtype=float) for k in model_keys}

    for i, col in enumerate(feature_names):
        for k in model_keys:
            model = ensemble.get(k)
            if model is None:
                continue
            if k in ['nb', 'svm']:
                X_in = sample.copy()
            else:
                X_in = X_tree_sample.copy()

            if col not in X_in.columns:
                # feature not used by this input, skip
                per_model_drops[k][i] = 0.0
                continue

            X_perm = X_in.copy()
            X_perm[col] = np.random.permutation(X_perm[col].values)

            feature_names_model = getattr(model, 'feature_names_in_', None)
            if feature_names_model is not None:
                for c in feature_names_model:
                    if c not in X_perm.columns:
                        X_perm[c] = 0
                X_perm = X_perm[list(feature_names_model)]

            try:
                probs_perm = model.predict_proba(X_perm)[:, 1]
                perm_auc = roc_auc_score(y_sample, probs_perm)
                drop = per_model_baseline.get(k, 0.0) - perm_auc
                per_model_drops[k][i] = max(0.0, drop)
            except Exception:
                per_model_drops[k][i] = 0.0

    # aggregate per-model drops (mean across models) to get a single importance per feature
    drops_matrix = np.vstack([per_model_drops[k] for k in model_keys])
    values = drops_matrix.mean(axis=0)

    # build hover strings with per-model contributions
    per_model_labels = { 'nb': 'NB', 'svm': 'SVM', 'rf': 'RF', 'xgb': 'XGB' }
    hover_lines = []
    for i, feat in enumerate(feature_names):
        parts = []
        for k in model_keys:
            parts.append(f"{per_model_labels[k]}: {per_model_drops[k][i]:.5f}")
        hover_lines.append('<br>'.join(parts))

    fi_df = _top_feature_frame(feature_names, values, top_n=10)

    # align hover lines to the top features
    top_feats = fi_df['feature'].astype(str).tolist()
    hover_for_top = []
    feat_to_idx = {str(f): idx for idx, f in enumerate(feature_names)}
    for f in top_feats:
        idx = feat_to_idx.get(f, None)
        hover_for_top.append(hover_lines[idx] if idx is not None else '')

    # attach customdata to show per-model breakdown in hover
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=fi_df['importance'],
        y=fi_df['feature'],
        orientation='h',
        marker=dict(
            color=fi_df['importance'],
            colorscale='Blues',
            showscale=False,
            line=dict(color='white', width=1),
        ),
        customdata=np.array(hover_for_top)[:, None],
        hovertemplate='%{y}<br>Importance (mean AUC drop): %{x:.5f}<br>%{customdata}<extra></extra>',
    ))

    fig.update_traces(showlegend=False)
    fig.update_xaxes(title='Importance (mean AUC drop)', title_standoff=12, automargin=True)
    fig.update_yaxes(title=None, automargin=True, ticks='outside')

    height = max(420, 42 * len(fi_df) + 120)
    fig.update_layout(margin=dict(t=80, b=50, l=220, r=30))
    return _apply_theme(fig, 'Top 10 Churn Drivers', height=height, showlegend=False)

def plot_feature_importance_high_value(X_hv, customer_features_df, top_n=10):
    return _tree_feature_importance(X_hv, 'high_value', 'high_value_customer', 'Top 10 High-Value Drivers', customer_features_df, top_n)

def plot_feature_importance_risk(X_hr, customer_features_df, top_n=10):
    return _tree_feature_importance(X_hr, 'high_risk', 'high_future_cancellation', 'Top 10 High-Risk Drivers', customer_features_df, top_n)


def feature_importance_plots(customer_features_df,X_hr,X_churn_tree,X_hv):
    print("\n CATEGORY 5: Feature Importance Visualizations...")

    fig_churn = plot_feature_importance_churn(customer_features_df, X_churn_tree)
    print("Churn feature importance chart")

    fig_value = plot_feature_importance_high_value(X_hv, customer_features_df)
    print("High value feature importance chart")

    fig_risk = plot_feature_importance_risk(X_hr, customer_features_df)
    print("High risk feature importance chart")
