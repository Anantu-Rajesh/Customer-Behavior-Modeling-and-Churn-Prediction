import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
from src.models.churn import _ensemble_predict_proba
from sklearn.base import BaseEstimator
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
        marker=dict(color=fi_df['importance'], colorscale='Blues', line=dict(color='white', width=1)),
        hovertemplate='%{y}<br>Importance: %{x:.5f}<extra></extra>',
    ))
    fig.update_traces(showlegend=False)
    fig.update_xaxes(title='Importance')
    fig.update_yaxes(title=None)
    return _apply_theme(fig, title, height=440, showlegend=False)


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
    X, y = _prepare_features(customer_features_df, target='churn', model_type='linear')
    models = inf.load_model()
    ensemble = models['churn']

    if y is None:
        raise ValueError('Missing churn target column for feature importance.')

    sample_size = min(len(X), 1000)
    sample = X.sample(sample_size, random_state=42) if len(X) > sample_size else X
    X_tree_sample = X_tree.loc[sample.index]
    y_sample = y.loc[sample.index]

    # manually compute permutation importance
    baseline_score = roc_auc_score(y_sample, _ensemble_predict_proba(ensemble, sample, X_tree_sample))
    values = []
    feature_names = np.asarray(sample.columns, dtype=object)
    
    for col in feature_names:
        X_permuted = sample.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col].values)
        permuted_score = roc_auc_score(y_sample, _ensemble_predict_proba(ensemble, X_permuted, X_tree_sample))
        values.append(baseline_score - permuted_score)
    
    values = np.array(values)
    fi_df = _top_feature_frame(feature_names, values, top_n=10)
    return _plot_top_features(fi_df, 'Top 10 Churn Drivers')

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
