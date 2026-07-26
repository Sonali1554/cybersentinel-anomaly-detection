"""Concept Drift Demo Component — Shows adaptive vs static baseline."""

import streamlit as st
import plotly.graph_objects as go
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import MODELS_DIR, DRIFT_CONFIG
from src.concept_drift import ConceptDriftAdapter


def render_concept_drift_demo():
    """Render concept drift adaptation visualization."""
    st.subheader("🔄 Concept Drift Adaptation Demo")

    st.markdown("""
    **Why this matters:** Legitimate behaviour evolves — new work patterns, new devices, 
    role changes. A static baseline would permanently flag these as anomalies. 
    Our adaptive baseline tracks genuine behaviour shifts while still catching real attacks.
    """)

    # Load or create drift adapter: use the trained one if it exists, else a fresh demo instance
    try:
        adapter = joblib.load(MODELS_DIR / "drift_adapter.pkl")
    except FileNotFoundError:
        adapter = ConceptDriftAdapter()

    # Generate demo data: simulated behaviour that shifts over time to show drift handling
    demo = adapter.demonstrate_adaptation("demo_entity")

    # ── Visualization ────────────────────────────────────────────────────
    fig = go.Figure()

    # Actual values: the raw behaviour readings being tracked
    fig.add_trace(go.Scatter(
        x=demo["event_index"],
        y=demo["actual_values"],
        mode="markers",
        marker=dict(size=5, color="#3B82F6", opacity=0.8),
        name="Actual Behaviour",
    ))

    # Static baseline (doesn't adapt): fixed mean/threshold that never updates
    fig.add_trace(go.Scatter(
        x=demo["event_index"],
        y=demo["static_baseline"],
        mode="lines",
        line=dict(color="#DC2626", width=2, dash="dash"),
        name="Static Baseline (Mean)",
    ))
    fig.add_trace(go.Scatter(
        x=demo["event_index"],
        y=demo["static_upper"],
        mode="lines",
        line=dict(color="#DC2626", width=1, dash="dot"),
        name="Static Threshold (±2σ)",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=demo["event_index"],
        y=demo["static_lower"],
        mode="lines",
        line=dict(color="#DC2626", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(220, 38, 38, 0.08)",
        showlegend=False,
    ))

    # Adaptive baseline: recalculates its mean/threshold as new behaviour comes in
    fig.add_trace(go.Scatter(
        x=demo["event_index"],
        y=demo["adaptive_baseline"],
        mode="lines",
        line=dict(color="#00FF88", width=2),
        name="Adaptive Baseline",
    ))
    fig.add_trace(go.Scatter(
        x=demo["event_index"],
        y=demo["adaptive_upper"],
        mode="lines",
        line=dict(color="#00FF88", width=1, dash="dot"),
        name="Adaptive Threshold (±2σ)",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=demo["event_index"],
        y=demo["adaptive_lower"],
        mode="lines",
        line=dict(color="#00FF88", width=1, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(0, 255, 136, 0.08)",
        showlegend=False,
    ))

    # Phase annotations: label the stages of the demo (e.g. before/after behaviour change)
    for phase in demo["phases"]:
        fig.add_vrect(
            x0=phase["start"], x1=phase["end"],
            annotation_text=phase["label"],
            annotation_position="top left",
            fillcolor="rgba(0,0,0,0.03)",
            line_width=1, line_color="gray", line_dash="dash",
        )

    # Drift detection points: mark exactly when the system noticed behaviour had shifted
    for pt in demo.get("drift_detected_at", []):
        fig.add_vline(
            x=pt, line_dash="dash", line_color="orange",
            annotation_text="Drift Detected",
        )

    fig.update_layout(
        title="Static vs Adaptive Baseline — Concept Drift Response",
        xaxis_title="Event Index",
        yaxis_title="Behavioural Feature Value",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(yanchor="top", y=-0.15, xanchor="center", x=0.5, orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Explanation ──────────────────────────────────────────────────────
    # Plain-language summary of why the adaptive approach beats a static one
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**❌ Static Baseline (Red)**")
        st.markdown("""
        - Fixed mean and thresholds from initial training
        - After behaviour shifts, legitimate events fall outside bounds
        - Results in **persistent false positives**
        """)
    with col2:
        st.markdown("**✅ Adaptive Baseline (Green)**")
        st.markdown(f"""
        - Exponential decay weighting (factor: {DRIFT_CONFIG['decay_factor']})
        - Sliding window of {DRIFT_CONFIG['window_size']} events
        - Automatically adjusts to genuine behaviour changes
        - Page-Hinkley drift detection triggers recalibration
        """)
