"""Real-Time Streaming Simulation Component."""

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SEVERITY_TIERS


def render_streaming_simulation(alert_df):
    """
    Simulate real-time event streaming with auto-updating alert view.
    Uses Streamlit's session state to maintain streaming position.
    """
    st.subheader("⚡ Real-Time Streaming Simulation")

    # Initialize session state: remembers stream position across Streamlit reruns
    if "stream_index" not in st.session_state:
        st.session_state.stream_index = 0
    if "stream_running" not in st.session_state:
        st.session_state.stream_running = False
    if "stream_events" not in st.session_state:
        st.session_state.stream_events = []

    # Controls: let the user set playback speed and start/pause/reset the simulation
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        batch_size = st.selectbox("Events per tick", [1, 5, 10, 25], index=1)
    with col2:
        speed = st.selectbox("Speed", ["Slow (2s)", "Normal (1s)", "Fast (0.5s)"], index=1)
    with col3:
        if st.button("▶️ Start / Resume" if not st.session_state.stream_running else "⏸️ Pause"):
            st.session_state.stream_running = not st.session_state.stream_running
    with col4:
        if st.button("🔄 Reset"):
            st.session_state.stream_index = 0
            st.session_state.stream_events = []
            st.session_state.stream_running = False

    # Prepare sorted events: replay events in the order they actually happened
    sorted_df = alert_df.sort_values("timestamp").reset_index(drop=True)
    total = len(sorted_df)

    # Progress bar shows how far through the simulated stream we are
    progress = st.session_state.stream_index / max(total, 1)
    st.progress(min(progress, 1.0),
                text=f"Processed {st.session_state.stream_index}/{total} events")

    # ── Live metrics ─────────────────────────────────────────────────────
    # Only look at events "seen so far" to mimic a real-time feed
    current_events = sorted_df.iloc[:st.session_state.stream_index]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Events Processed", st.session_state.stream_index)
    if len(current_events) > 0 and "severity" in current_events.columns:
        m2.metric("Critical Alerts", len(current_events[current_events["severity"] == "Critical"]))
        m3.metric("High Alerts", len(current_events[current_events["severity"] == "High"]))
        anomaly_rate = len(current_events[current_events.get("risk_score", 0) > 40]) / max(len(current_events), 1) * 100
        m4.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")

    # ── Live feed ────────────────────────────────────────────────────────
    if st.session_state.stream_index > 0:
        # Only keep the most recent 20 events so the feed stays readable
        recent = sorted_df.iloc[max(0, st.session_state.stream_index - 20):st.session_state.stream_index]

        # Risk score rolling chart: track risk trend as new events arrive
        if "risk_score" in recent.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(recent))),
                y=recent["risk_score"].values,
                mode="lines+markers",
                marker=dict(
                    color=recent["severity"].map(
                        lambda s: SEVERITY_TIERS.get(s, {}).get("color", "#3B82F6")
                    ).tolist(),
                    size=8,
                ),
                line=dict(color="rgba(100,100,100,0.4)", width=1),
                hovertemplate="Risk: %{y:.1f}<extra></extra>",
            ))
            fig.update_layout(
                title="Rolling Risk Scores (Last 20 Events)",
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis_title="Risk Score",
                xaxis_title="Recent Events",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Recent events feed, newest first
        st.write("**Live Event Feed:**")
        display_cols = ["timestamp", "entity_id", "risk_score", "severity",
                        "predicted_label", "geo_city"]
        avail = [c for c in display_cols if c in recent.columns]
        st.dataframe(recent[avail].iloc[::-1], use_container_width=True, height=250)

    # ── Auto-advance ─────────────────────────────────────────────────────
    # If running, feed in the next batch of events and refresh the page after a short delay
    if st.session_state.stream_running and st.session_state.stream_index < total:
        st.session_state.stream_index = min(
            st.session_state.stream_index + batch_size, total
        )
        delay = {"Slow (2s)": 2.0, "Normal (1s)": 1.0, "Fast (0.5s)": 0.5}
        time.sleep(delay.get(speed, 1.0))
        st.rerun()
    elif st.session_state.stream_index >= total and st.session_state.stream_running:
        st.session_state.stream_running = False
        st.success("✅ Streaming complete — all events processed!")
