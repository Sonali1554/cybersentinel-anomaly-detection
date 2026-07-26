"""
Utility Functions
==================
Shared helpers for risk scoring, severity classification,
logging, and data I/O across all modules.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RISK_WEIGHTS, SEVERITY_TIERS, DATA_DIR


def compute_risk_score(baseline_scores, lstm_scores, classifier_confidence,
                       entity_risk_history=None):
    """
    Compute composite risk score (0–100) using weighted combination.

    Formula:
      risk = 0.35 × IF_score + 0.35 × LSTM_error + 0.20 × classifier_conf + 0.10 × history
    """
    n = len(baseline_scores.get("isolation_forest_score", []))

    if_score = baseline_scores.get("isolation_forest_score", np.zeros(n))
    lstm_error = lstm_scores if lstm_scores is not None else np.zeros(n)
    clf_conf = classifier_confidence if classifier_confidence is not None else np.zeros(n)
    history = entity_risk_history if entity_risk_history is not None else np.zeros(n)

    # Ensure all arrays are same length
    min_len = min(len(if_score), len(lstm_error), len(clf_conf), len(history))
    if_score = if_score[:min_len]
    lstm_error = lstm_error[:min_len]
    clf_conf = clf_conf[:min_len]
    history = history[:min_len]

    risk = (
        RISK_WEIGHTS["isolation_forest_score"] * if_score +
        RISK_WEIGHTS["lstm_reconstruction_error"] * lstm_error +
        RISK_WEIGHTS["classifier_confidence"] * clf_conf +
        RISK_WEIGHTS["entity_risk_history"] * history
    )

    # Scale to 0–100
    return np.clip(risk * 100, 0, 100)


def assign_severity(risk_scores):
    """Map risk scores to severity tiers."""
    severities = []
    for score in risk_scores:
        assigned = "Low"
        for tier_name, tier_range in SEVERITY_TIERS.items():
            if tier_range["min"] <= score <= tier_range["max"]:
                assigned = tier_name
                break
        if score > 100:
            assigned = "Critical"
        severities.append(assigned)
    return severities


def get_severity_color(severity):
    """Get color for a severity tier."""
    return SEVERITY_TIERS.get(severity, {"color": "#6B7280"})["color"]


def build_alert_queue(df, risk_scores, severities, predictions, explanations=None):
    """
    Build a ranked alert queue from scored events.
    Returns a DataFrame sorted by risk score (highest first).
    """
    alert_data = df.copy()
    # Trim to match df length in case score arrays came out slightly longer (safety guard)
    alert_data["risk_score"] = risk_scores[:len(df)]
    alert_data["severity"] = severities[:len(df)]

    if predictions is not None:
        alert_data["predicted_label"] = predictions[:len(df)]

    if explanations is not None:
        alert_data["explanation"] = [
            json.dumps(e.get("contributing_factors", [])[:3]) if e else "[]"
            for e in explanations[:len(df)]
        ]
        alert_data["risk_summary"] = [
            e.get("risk_summary", "") if e else ""
            for e in explanations[:len(df)]
        ]

    # Sort by risk score descending
    alert_data = alert_data.sort_values("risk_score", ascending=False).reset_index(drop=True)

    return alert_data


def compute_entity_risk_history(df, decay=0.9):
    """
    Compute per-entity historical risk based on past anomaly events.
    More recent anomalies contribute more to the score.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"])

    entity_risk = defaultdict(float)

    for entity_id, group in df.groupby("entity_id"):
        anomaly_events = group[group["label"] != "normal"]
        total_events = len(group)

        if total_events == 0:
            continue

        # Base risk from anomaly ratio
        anomaly_ratio = len(anomaly_events) / total_events

        # Decay-weighted recent anomalies
        if len(anomaly_events) > 0:
            n = len(anomaly_events)
            weights = np.array([decay ** (n - 1 - i) for i in range(n)])
            weighted_risk = weights.sum() / max(total_events, 1)
            entity_risk[entity_id] = min(anomaly_ratio * 0.5 + weighted_risk * 0.5, 1.0)
        else:
            entity_risk[entity_id] = 0.0

    return entity_risk


def format_timestamp(ts):
    """Format timestamp for display."""
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def get_alert_summary_stats(alert_df):
    """Compute summary statistics for the alert queue."""
    stats = {
        "total_alerts": len(alert_df[alert_df["risk_score"] > 20]),
        "critical": len(alert_df[alert_df["severity"] == "Critical"]),
        "high": len(alert_df[alert_df["severity"] == "High"]),
        "medium": len(alert_df[alert_df["severity"] == "Medium"]),
        "low": len(alert_df[alert_df["severity"] == "Low"]),
        "avg_risk_score": alert_df["risk_score"].mean(),
        "max_risk_score": alert_df["risk_score"].max(),
        "unique_entities_flagged": alert_df[alert_df["risk_score"] > 40]["entity_id"].nunique(),
    }
    return stats


def load_generated_data():
    """Load generated synthetic data from disk."""
    access_logs = pd.read_csv(DATA_DIR / "access_logs.csv")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    with open(DATA_DIR / "entity_profiles.json", "r") as f:
        profiles = json.load(f)

    return access_logs, train, test, profiles
