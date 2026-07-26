"""
Model Results Component — ML Evaluation Dashboard
=====================================================
Displays offline evaluation results for the trained XGBoost anomaly
classifier: KPI summary, ROC/AUC curve, confusion matrix, per-attack
recall, feature importance, SHAP attribution, risk score spread,
precision@K, and the full classification report.

Loads already-trained models/artifacts from disk (models/) and
pre-computed evaluation metrics from disk (data/evaluation_metrics.json).
"""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Add project root to path so `config` and `src` are importable from this component
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_DIR, MODELS_DIR

# Fixed categorical color order shared with the rest of the dashboard (never cycled)
PALETTE = ["#00FF88", "#DC2626", "#F59E0B", "#3B82F6", "#8B5CF6", "#EC4899", "#14B8A6", "#6B7280"]

# The 4 baseline ensemble scores appended to the engineered features before the
# classifier was trained (see train_pipeline.py: aug_feature_names)
BASELINE_SCORE_NAMES = [
    "statistical_score", "isolation_forest_score",
    "one_class_svm_score", "baseline_combined_score",
]


@st.cache_data
def _load_metrics():
    """Load the pre-computed evaluation metrics JSON written by the training pipeline."""
    metrics_path = DATA_DIR / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


@st.cache_resource
def _load_models():
    """Load the label encoder, XGBoost classifier, feature engineer, and SHAP explainer from disk."""
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgb_classifier.pkl")
    feature_eng = joblib.load(MODELS_DIR / "feature_engineer.pkl")
    try:
        explainer = joblib.load(MODELS_DIR / "explainer.pkl")
    except FileNotFoundError:
        explainer = None
    return label_encoder, xgb_model, feature_eng, explainer


def render_model_results(alert_df):
    """Main entry point: renders the full Model Results page in order."""
    st.subheader("📊 Model Results")
    st.caption("Offline evaluation of the trained XGBoost anomaly classifier on held-out test data.")

    metrics = _load_metrics()
    try:
        label_encoder, xgb_model, feature_eng, explainer = _load_models()
    except FileNotFoundError:
        st.warning(
            "Model artifacts not found in `models/`. Run the training pipeline first:\n\n"
            "```\npython train_pipeline.py\n```"
        )
        return

    # Numeric code for the "normal" class, e.g. 7 (label_encoder sorts classes alphabetically)
    normal_enc = label_encoder.transform(["normal"])[0]
    y_true = label_encoder.transform(alert_df["label"])            # ground-truth multi-class codes
    y_pred = label_encoder.transform(alert_df["predicted_label"])  # predicted multi-class codes
    y_true_binary = (y_true != normal_enc).astype(int)             # collapse to binary normal(0)/anomaly(1)

    # 1. KPI row — headline accuracy/precision/recall/F1/AUC/FPR tiles
    _render_kpi_row(metrics)
    st.markdown("---")

    # 2. ROC / AUC curve — risk_score as the continuous score vs. binary anomaly labels
    _render_roc_curve(alert_df, y_true_binary)
    st.markdown("---")

    # 3. Confusion matrix — multi-class true vs. predicted label heatmap
    _render_confusion_matrix(y_true, y_pred, label_encoder)
    st.markdown("---")

    # 4. Per-attack detection recall — exact-class recall vs. binary detection rate
    _render_per_attack_recall(alert_df, label_encoder)
    st.markdown("---")

    # 5. Feature importance — top 15 features driving the XGBoost classifier
    _render_feature_importance(xgb_model, feature_eng)
    st.markdown("---")

    # 6. SHAP attribution — top contributing factors for the single highest-risk alert
    _render_shap_attribution(alert_df, feature_eng, explainer)
    st.markdown("---")

    # 7. Risk score by attack type — box plot of risk_score spread per true label
    _render_risk_by_label(alert_df)
    st.markdown("---")

    # 8. Precision@K — precision among the top-K% highest-risk-scored events
    _render_precision_at_k(alert_df)
    st.markdown("---")

    # 9. Full classification report — per-class precision/recall/F1/support table
    _render_classification_report(y_true, y_pred, label_encoder)


def _render_kpi_row(metrics):
    """Show Accuracy, Precision, Recall, F1, AUC-ROC, and FPR as KPI tiles."""
    st.markdown("#### Key Performance Indicators")
    cols = st.columns(6)
    kpis = [
        ("Accuracy", metrics.get("binary_accuracy"), "{:.2%}"),
        ("Precision", metrics.get("binary_precision"), "{:.2%}"),
        ("Recall", metrics.get("binary_recall"), "{:.2%}"),
        ("F1", metrics.get("binary_f1"), "{:.2%}"),
        ("AUC-ROC", metrics.get("auc_roc"), "{:.3f}"),
        ("FPR", metrics.get("false_positive_rate"), "{:.2%}"),
    ]
    for col, (label, value, fmt) in zip(cols, kpis):
        col.metric(label, fmt.format(value) if value is not None else "N/A")


def _render_roc_curve(alert_df, y_true_binary):
    """Plot ROC (FPR vs TPR) using risk_score as the continuous score, with AUC in the legend."""
    st.markdown("#### ROC / AUC Curve")
    fpr, tpr, _ = roc_curve(y_true_binary, alert_df["risk_score"])
    auc_val = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines", name=f"Risk Score (AUC = {auc_val:.3f})",
        line=dict(color="#00FF88", width=2),
        fill="tozeroy", fillcolor="rgba(0, 255, 136, 0.10)",
    ))
    fig.add_trace(go.Scatter(  # diagonal random-classifier baseline
        x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
        line=dict(color="#6B7280", width=1, dash="dash"),
    ))
    fig.update_layout(
        title="ROC Curve — Binary Anomaly Detection (normal vs. anomaly)",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=420, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_confusion_matrix(y_true, y_pred, label_encoder):
    """Render the multi-class confusion matrix as a Blues heatmap with cell counts shown."""
    st.markdown("#### Confusion Matrix (Multi-Class)")
    classes = list(label_encoder.classes_)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    fig = px.imshow(
        cm, x=classes, y=classes, text_auto=True,
        color_continuous_scale=[[0, "#0A0E17"], [0.5, "#064E3B"], [1, "#00FF88"]],
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
    )
    fig.update_layout(title="Confusion Matrix", height=520, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def _render_per_attack_recall(alert_df, label_encoder):
    """Grouped bar chart: exact-class recall vs. binary detection rate, per attack type (skips 'normal')."""
    st.markdown("#### Per-Attack Detection Recall")
    attack_types = [c for c in label_encoder.classes_ if c != "normal"]

    exact_recall, binary_detect = [], []
    for attack in attack_types:
        subset = alert_df[alert_df["label"] == attack]
        if len(subset) == 0:
            exact_recall.append(0.0)
            binary_detect.append(0.0)
            continue
        exact_recall.append((subset["predicted_label"] == attack).mean())      # predicted == true class
        binary_detect.append((subset["predicted_label"] != "normal").mean())   # predicted as any anomaly

    fig = go.Figure()
    fig.add_trace(go.Bar(x=attack_types, y=exact_recall, name="Exact-Class Recall", marker_color="#3B82F6"))
    fig.add_trace(go.Bar(x=attack_types, y=binary_detect, name="Binary Detection Rate", marker_color="#00FF88"))
    fig.update_layout(
        title="Recall by Attack Type", barmode="group",
        yaxis_title="Rate", yaxis_tickformat=".0%",
        height=420, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_feature_importance(xgb_model, feature_eng):
    """Horizontal bar chart of the top 15 features driving the XGBoost classifier."""
    st.markdown("#### Feature Importance")
    aug_feature_names = feature_eng.get_feature_names() + BASELINE_SCORE_NAMES
    importances = xgb_model.feature_importances_

    imp_df = pd.DataFrame({
        "feature": aug_feature_names[:len(importances)],
        "importance": importances,
    })
    imp_df = imp_df.sort_values("importance", ascending=False).head(15)
    imp_df = imp_df.sort_values("importance", ascending=True)  # ascending so the biggest bar plots on top

    fig = go.Figure(go.Bar(
        x=imp_df["importance"], y=imp_df["feature"], orientation="h", marker_color="#00FF88",
    ))
    fig.update_layout(
        title="Top 15 Feature Importances (XGBoost)", xaxis_title="Importance",
        height=500, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_shap_attribution(alert_df, feature_eng, explainer):
    """Horizontal bar chart of SHAP contributing factors for the single highest-risk alert."""
    st.markdown("#### SHAP Attribution — Top Alert")
    if explainer is None:
        st.info("Explainer artifact not found; skipping SHAP attribution.")
        return

    top_event = alert_df.loc[alert_df["risk_score"].idxmax()]  # the single highest risk_score event
    feature_names = feature_eng.get_feature_names()

    # Rebuild the model's scaled input vector for this event; the 4 baseline ensemble
    # scores are padded with zero since they aren't persisted in the alert queue export
    raw_vals = top_event[feature_names].to_numpy(dtype=float).reshape(1, -1)
    scaled_vals = feature_eng.scaler.transform(raw_vals)
    baseline_padding = np.zeros((1, len(BASELINE_SCORE_NAMES)))
    X_single = np.hstack([scaled_vals, baseline_padding])

    explanation = explainer.explain_event(X_single, event_data=top_event, top_n=8)
    factors = explanation["contributing_factors"]
    if not factors:
        st.info("No contributing factors available for this event.")
        return

    factor_df = pd.DataFrame(factors)
    value_col = "shap_value" if "shap_value" in factor_df.columns else "importance"
    factor_df = factor_df.iloc[::-1]  # reverse so the most important factor plots on top

    colors = ["#DC2626" if v > 0 else "#3B82F6" for v in factor_df[value_col]]  # red = pushes toward anomaly
    fig = go.Figure(go.Bar(
        x=factor_df[value_col], y=factor_df["display_name"], orientation="h", marker_color=colors,
    ))
    fig.update_layout(
        title=f"Top Contributing Factors — Entity {top_event.get('entity_id', 'N/A')} "
              f"(risk score {top_event['risk_score']:.1f})",
        xaxis_title="SHAP Impact" if value_col == "shap_value" else "Importance",
        height=420, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Factor Descriptions"):  # plain-language description per contributing factor
        for f in factors:
            st.markdown(f"- **{f['display_name']}**: {f['description']}")


def _render_risk_by_label(alert_df):
    """Box plot of risk_score distribution split out by the true attack label."""
    st.markdown("#### Risk Score Distribution by Attack Type")
    fig = px.box(
        alert_df, x="label", y="risk_score", color="label",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(
        title="Risk Score by True Label", xaxis_title="Label", yaxis_title="Risk Score",
        showlegend=False, height=450, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_precision_at_k(alert_df):
    """Bar chart of precision among the top-K% highest-risk-scored events."""
    st.markdown("#### Precision @ K")
    ranked = alert_df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    k_pcts = [0.5, 1, 2, 5, 10]
    precisions = []
    for k in k_pcts:
        n = max(1, int(len(ranked) * k / 100))  # number of events in the top-K% slice
        top_k = ranked.head(n)
        precisions.append((top_k["label"] != "normal").mean())  # fraction of that slice that's a true anomaly

    fig = go.Figure(go.Bar(
        x=[f"Top {k}%" for k in k_pcts], y=precisions, marker_color="#00FF88",
        text=[f"{p:.1%}" for p in precisions], textposition="outside",
    ))
    fig.update_layout(
        title="Precision at Top-K Ranked Alerts", yaxis_title="Precision",
        yaxis_tickformat=".0%", height=400, margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_classification_report(y_true, y_pred, label_encoder):
    """Render the full per-class precision/recall/F1/support table at the bottom of the page."""
    st.markdown("#### Full Classification Report")
    classes = list(label_encoder.classes_)
    report = classification_report(
        y_true, y_pred, labels=range(len(classes)), target_names=classes,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
