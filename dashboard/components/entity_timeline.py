"""Entity Behavior Timeline Component — Interactive event history view."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SEVERITY_TIERS, ANOMALY_DESCRIPTIONS


def render_entity_timeline(alert_df, full_df=None):
    """Render interactive entity timeline with anomaly highlights."""
    st.subheader("📊 Entity Behavior Timeline")

    # Entity selector: pick one user/device to inspect in detail
    entities = sorted(alert_df["entity_id"].unique())
    selected_entity = st.selectbox("Select Entity", entities, index=0)

    # Get entity data: pull just this entity's rows out of the full/alert datasets
    entity_data = alert_df[alert_df["entity_id"] == selected_entity].copy()
    if full_df is not None:
        entity_full = full_df[full_df["entity_id"] == selected_entity].copy()
    else:
        entity_full = entity_data

    entity_full["timestamp"] = pd.to_datetime(entity_full["timestamp"])
    entity_full = entity_full.sort_values("timestamp")

    if len(entity_full) == 0:
        st.warning(f"No events found for entity {selected_entity}")
        return

    # ── Entity Summary Card ──────────────────────────────────────────────
    # Quick snapshot of this entity's activity and risk before diving into the chart
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", len(entity_full))
    col2.metric("Entity Type", entity_full["entity_type"].iloc[0])
    anomaly_count = len(entity_full[entity_full["label"] != "normal"]) if "label" in entity_full.columns else 0
    col3.metric("Anomalies", anomaly_count)
    avg_risk = entity_data["risk_score"].mean() if "risk_score" in entity_data.columns and len(entity_data) > 0 else 0
    col4.metric("Avg Risk Score", f"{avg_risk:.1f}")

    # ── Timeline Chart ───────────────────────────────────────────────────
    # Three stacked charts sharing one time axis so patterns line up visually
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Risk Score Over Time", "Session Duration", "Resources Accessed"),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3],
    )

    # Risk score timeline: how risky this entity's behaviour looked over time
    if "risk_score" in entity_data.columns:
        ts_data = entity_data.copy()
        ts_data["timestamp"] = pd.to_datetime(ts_data["timestamp"])
        ts_data = ts_data.sort_values("timestamp")

        # Color each point by its severity so spikes stand out
        colors = ts_data["severity"].map(
            lambda s: SEVERITY_TIERS.get(s, {}).get("color", "#6B7280")
        ) if "severity" in ts_data.columns else ["#3B82F6"] * len(ts_data)

        fig.add_trace(go.Scatter(
            x=ts_data["timestamp"],
            y=ts_data["risk_score"],
            mode="markers+lines",
            marker=dict(color=list(colors), size=8),
            line=dict(color="rgba(100,100,100,0.3)", width=1),
            name="Risk Score",
            hovertemplate="Time: %{x}<br>Risk: %{y:.1f}<extra></extra>",
        ), row=1, col=1)

        # Threshold lines: mark where each severity tier begins on the risk axis
        for sev, tier in SEVERITY_TIERS.items():
            fig.add_hline(y=tier["min"], line_dash="dash",
                          line_color=tier["color"], opacity=0.3,
                          annotation_text=sev, row=1, col=1)

    # Session duration: red bars flag sessions tied to non-normal (anomalous) events
    fig.add_trace(go.Bar(
        x=entity_full["timestamp"],
        y=entity_full["session_duration"],
        marker_color=entity_full["label"].apply(
            lambda l: "#DC2626" if l != "normal" else "#00FF88"
        ) if "label" in entity_full.columns else "#00FF88",
        name="Session Duration",
        hovertemplate="Time: %{x}<br>Duration: %{y}s<extra></extra>",
    ), row=2, col=1)

    # Resources (as categorical scatter): plot which resources were touched and when
    if "resource_accessed" in entity_full.columns:
        resources = entity_full["resource_accessed"].unique()
        # Map each resource name to a number so it can sit on a y-axis
        res_map = {r: i for i, r in enumerate(resources)}
        fig.add_trace(go.Scatter(
            x=entity_full["timestamp"],
            y=entity_full["resource_accessed"].map(res_map),
            mode="markers",
            marker=dict(
                size=8,
                color=entity_full["label"].apply(
                    lambda l: 1 if l != "normal" else 0
                ) if "label" in entity_full.columns else 0,
                colorscale=["#00FF88", "#DC2626"],
            ),
            text=entity_full["resource_accessed"],
            hovertemplate="Time: %{x}<br>Resource: %{text}<extra></extra>",
            name="Resources",
        ), row=3, col=1)

        fig.update_yaxes(
            tickvals=list(res_map.values()),
            ticktext=[r.split("/")[-1][:20] for r in res_map.keys()],
            row=3, col=1,
        )

    fig.update_layout(height=700, showlegend=False,
                      margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── Event Details Table ──────────────────────────────────────────────
    # Raw event log so an analyst can drill into the specifics behind the chart
    st.write("**Recent Events:**")
    display_cols = ["timestamp", "resource_accessed", "auth_method", "session_duration",
                    "geo_city", "label", "risk_score", "severity"]
    available = [c for c in display_cols if c in entity_data.columns]
    st.dataframe(entity_data[available].tail(20), use_container_width=True, height=300)
