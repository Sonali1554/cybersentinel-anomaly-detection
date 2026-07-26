"""Alert Queue Component — Ranked alerts with severity tiers."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SEVERITY_TIERS


def render_alert_queue(alert_df):
    """Render the ranked alert queue with filters and severity tiers."""
    st.subheader("🚨 Alert Queue")

    # ── Summary metrics ──────────────────────────────────────────────────
    # Show a quick count of alerts in each severity tier
    col1, col2, col3, col4 = st.columns(4)
    for sev, col in zip(["Critical", "High", "Medium", "Low"], [col1, col2, col3, col4]):
        count = len(alert_df[alert_df["severity"] == sev])
        color = SEVERITY_TIERS[sev]["color"]
        col.metric(f"🔴 {sev}" if sev == "Critical" else
                   f"🟠 {sev}" if sev == "High" else
                   f"🔵 {sev}" if sev == "Medium" else
                   f"⚪ {sev}", count)

    # ── Filters ──────────────────────────────────────────────────────────
    # Let the analyst narrow down the alert list by severity, type, and count
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
    with col_f2:
        if "predicted_label" in alert_df.columns:
            anomaly_types = alert_df["predicted_label"].unique().tolist()
            type_filter = st.multiselect("Filter by Anomaly Type", anomaly_types)
        else:
            type_filter = []
    with col_f3:
        max_display = st.slider("Max Alerts Shown", 10, 200, 50)

    # ── Apply filters ────────────────────────────────────────────────────
    filtered = alert_df[alert_df["severity"].isin(severity_filter)] if severity_filter else alert_df
    if type_filter and "predicted_label" in filtered.columns:
        filtered = filtered[filtered["predicted_label"].isin(type_filter)]

    # Show the riskiest alerts first, capped to the number the user wants to see
    filtered = filtered.sort_values("risk_score", ascending=False).head(max_display)

    # ── Severity distribution chart ──────────────────────────────────────
    # Bar chart of how many alerts fall in each severity bucket
    sev_counts = alert_df["severity"].value_counts().reindex(
        ["Critical", "High", "Medium", "Low"], fill_value=0
    )
    fig_sev = go.Figure(data=[go.Bar(
        x=sev_counts.index,
        y=sev_counts.values,
        marker_color=[SEVERITY_TIERS[s]["color"] for s in sev_counts.index],
    )])
    fig_sev.update_layout(
        title="Alert Severity Distribution",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_sev, use_container_width=True)

    # ── Alert table ──────────────────────────────────────────────────────
    # Only show columns that actually exist in the data (keeps it robust to schema changes)
    display_cols = ["entity_id", "entity_type", "risk_score", "severity",
                    "predicted_label", "geo_city", "resource_accessed",
                    "auth_method", "timestamp"]
    available_cols = [c for c in display_cols if c in filtered.columns]

    # Tint each row's background based on its severity color for quick visual scanning
    st.dataframe(
        filtered[available_cols].style.apply(
            lambda row: [
                f"background-color: {SEVERITY_TIERS.get(row['severity'], {}).get('color', '#fff')}20"
            ] * len(row) if "severity" in row.index else [""] * len(row),
            axis=1,
        ),
        use_container_width=True,
        height=400,
    )

    return filtered
