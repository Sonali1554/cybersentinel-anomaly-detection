"""
DELIVERABLE 6: Analyst-Facing Dashboard
=========================================
Streamlit dashboard with:
- Alert queue ranked by risk score with severity tiers (Critical/High/Medium/Low)
- Real-time streaming simulation mode
- World map for impossible travel visualization
- Entity behavior timeline
- Concept drift adaptation demo
- Cold-start handling via peer group profiling display
- Model comparison toggle (Isolation Forest / LSTM / XGBoost)
- PDF report export button

Run: streamlit run dashboard/app.py
"""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, MODELS_DIR, SEVERITY_TIERS, ANOMALY_TYPES, ANOMALY_DESCRIPTIONS

# Import dashboard components
from dashboard.components.alert_queue import render_alert_queue
from dashboard.components.world_map import render_world_map
from dashboard.components.entity_timeline import render_entity_timeline
from dashboard.components.streaming_sim import render_streaming_simulation
from dashboard.components.concept_drift_demo import render_concept_drift_demo
from dashboard.components.model_comparison import render_model_comparison
from dashboard.components.report_export import render_report_export
from dashboard.components.model_results import render_model_results

# ─── Dark Plotly Template ────────────────────────────────────────────────────
import plotly.io as pio

CYBER_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10, 14, 23, 0.8)",
        font=dict(color="#E5E7EB", family="monospace"),
        title=dict(font=dict(color="#00FF88")),
        xaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937", color="#9CA3AF"),
        yaxis=dict(gridcolor="#1F2937", zerolinecolor="#1F2937", color="#9CA3AF"),
        colorway=["#00FF88", "#3B82F6", "#F59E0B", "#DC2626", "#8B5CF6", "#14B8A6", "#EC4899", "#6B7280"],
        legend=dict(font=dict(color="#D1D5DB")),
        polar=dict(
            bgcolor="rgba(10, 14, 23, 0.8)",
            radialaxis=dict(gridcolor="#1F2937", color="#9CA3AF"),
            angularaxis=dict(gridcolor="#1F2937", color="#9CA3AF"),
        ),
    )
)
pio.templates["cyber_dark"] = CYBER_TEMPLATE
pio.templates.default = "cyber_dark"

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberSentinel — Anomaly Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Cyber Theme CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(180deg, #0A0E17 0%, #0D1321 50%, #0A0E17 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #111827 100%);
    border-right: 1px solid #1F2937;
}

/* Header / title */
.stApp h1 {
    color: #00FF88 !important;
    text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
}
.stApp h2, .stApp h3, .stApp h4 {
    color: #00FF88 !important;
}

/* Metric cards glow */
[data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.8);
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 0 15px rgba(0, 255, 136, 0.05);
}
[data-testid="stMetricValue"] {
    color: #00FF88 !important;
}
[data-testid="stMetricLabel"] {
    color: #9CA3AF !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #1F2937;
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #064E3B, #065F46) !important;
    color: #00FF88 !important;
    border: 1px solid #00FF88 !important;
    border-radius: 6px;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
    transform: translateY(-1px);
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(17, 24, 39, 0.6);
    border: 1px solid #1F2937;
    border-radius: 8px;
}

/* Multiselect chips */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #064E3B !important;
    color: #00FF88 !important;
}

/* Sidebar nav radio */
[data-testid="stSidebar"] .stRadio label {
    color: #D1D5DB !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #00FF88 !important;
}

/* Horizontal rules */
hr {
    border-color: #1F2937 !important;
}

/* Caption text */
.stCaption, .stApp .stCaption p {
    color: #6B7280 !important;
}

/* Markdown text */
.stMarkdown p {
    color: #D1D5DB;
}

/* Scanline overlay for cyber feel */
.stApp::after {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 255, 136, 0.01) 2px,
        rgba(0, 255, 136, 0.01) 4px
    );
    z-index: 9999;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load alert queue and full access log data."""
    alert_path = DATA_DIR / "alert_queue.csv"
    full_path = DATA_DIR / "access_logs.csv"

    # Stop the app early if the training pipeline hasn't been run yet
    if not alert_path.exists():
        st.error(
            "⚠️ No data found! Run the training pipeline first:\n\n"
            "```\npython train_pipeline.py\n```"
        )
        st.stop()

    alert_df = pd.read_csv(alert_path)
    # Fall back to alert_df if the full access log isn't available
    full_df = pd.read_csv(full_path) if full_path.exists() else alert_df

    return alert_df, full_df


def load_cold_start_info():
    """Load cold-start handler for display."""
    try:
        import joblib
        cold_start = joblib.load(MODELS_DIR / "cold_start.pkl")
        return cold_start
    except FileNotFoundError:
        return None


def render_cold_start_panel(alert_df, cold_start_handler):
    """Render cold-start handling display."""
    st.subheader("🆕 Cold-Start Entity Handling")

    st.markdown("""
    **Problem:** New users/devices have no behavioural history — how do we score them?  
    **Solution:** Peer group profiling assigns a temporary baseline from similar entities.
    """)

    if cold_start_handler is None:
        st.warning("Cold-start handler not available. Run training pipeline first.")
        return

    # Find entities with fewest events (simulating cold-start)
    entity_counts = alert_df.groupby("entity_id").size().reset_index(name="event_count")
    entity_counts = entity_counts.sort_values("event_count")

    # Anything with under 20 events is treated as "new" and lacking full history
    cold_entities = entity_counts[entity_counts["event_count"] < 20].head(10)

    if len(cold_entities) == 0:
        st.info("No cold-start entities found in current dataset (all entities have sufficient history).")
        # Show demo anyway
        cold_entities = entity_counts.head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Cold-Start Entities (< 20 events):**")
        for _, row in cold_entities.iterrows():
            eid = row["entity_id"]
            count = row["event_count"]
            etype = alert_df[alert_df["entity_id"] == eid]["entity_type"].iloc[0]
            info = cold_start_handler.get_cold_start_info(eid, etype, count)

            progress = info["progress_pct"]
            status = "🔴" if progress < 30 else "🟡" if progress < 70 else "🟢"
            st.markdown(
                f"{status} **{eid}** ({etype}) — "
                f"{count} events ({progress}% to full profile) — "
                f"Peer: {info['peer_weight']:.0%} | Individual: {info['individual_weight']:.0%}"
            )

    with col2:
        st.markdown("**Peer Group Profiles:**")
        for etype, group in cold_start_handler.peer_groups.items():
            st.markdown(f"**{etype}** ({group['count']} entities in group)")
            st.markdown(
                f"- Avg login hour: {group['hour_mean']:.1f} (±{group['hour_std']:.1f})\n"
                f"- Avg session: {group['duration_mean']:.0f}s\n"
                f"- Auth success rate: {group['auth_success_rate']:.1%}"
            )

    # Transition visualization
    st.markdown("---")
    st.markdown("**Profile Transition: Peer → Individual**")

    # Show how trust shifts from the peer group average to the entity's own behaviour over time
    fig = go.Figure()
    x_events = list(range(1, 25))
    min_hist = cold_start_handler.config["min_history_events"]
    peer_weights = [max(0, 1 - e / min_hist) for e in x_events]
    ind_weights = [min(1, e / min_hist) for e in x_events]

    fig.add_trace(go.Scatter(
        x=x_events, y=peer_weights,
        mode="lines+markers", name="Peer Group Weight",
        line=dict(color="#F59E0B", width=2),
        fill="tozeroy", fillcolor="rgba(245, 158, 11, 0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=x_events, y=ind_weights,
        mode="lines+markers", name="Individual Profile Weight",
        line=dict(color="#00FF88", width=2),
        fill="tozeroy", fillcolor="rgba(0, 255, 136, 0.1)",
    ))
    fig.add_vline(x=min_hist, line_dash="dash", line_color="gray",
                  annotation_text=f"Full profile ({min_hist} events)")

    fig.update_layout(
        title="Weight Transition Over Event Accumulation",
        xaxis_title="Entity Event Count",
        yaxis_title="Weight",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_overview(alert_df):
    """Render dashboard overview/home page."""
    st.markdown("### 📈 System Overview")

    # Top metrics row: quick headline numbers for the whole dataset
    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(alert_df)
    # Anything scoring above 40 is flagged as an anomaly
    anomalies = len(alert_df[alert_df["risk_score"] > 40]) if "risk_score" in alert_df.columns else 0
    col1.metric("Total Events", f"{total:,}")
    col2.metric("Anomalies Detected", anomalies)
    col3.metric("Anomaly Rate", f"{anomalies/max(total,1)*100:.1f}%")
    col4.metric("Unique Entities", alert_df["entity_id"].nunique())
    if "risk_score" in alert_df.columns:
        col5.metric("Avg Risk Score", f"{alert_df['risk_score'].mean():.1f}")

    # Charts row
    c1, c2 = st.columns(2)

    with c1:
        # Anomaly type distribution
        if "predicted_label" in alert_df.columns:
            type_counts = alert_df[alert_df["predicted_label"] != "normal"]["predicted_label"].value_counts()
            if len(type_counts) > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=type_counts.index,
                    values=type_counts.values,
                    hole=0.4,
                    marker=dict(colors=["#00FF88", "#3B82F6", "#F59E0B",
                                        "#DC2626", "#8B5CF6", "#14B8A6", "#EC4899"]),
                )])
                fig.update_layout(title="Anomaly Type Distribution", height=350,
                                  margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Risk score histogram
        if "risk_score" in alert_df.columns:
            fig = go.Figure(data=[go.Histogram(
                x=alert_df["risk_score"],
                nbinsx=50,
                marker_color="#00FF88",
                opacity=0.85,
            )])
            fig.update_layout(title="Risk Score Distribution", height=350,
                              xaxis_title="Risk Score", yaxis_title="Count",
                              margin=dict(l=20, r=20, t=40, b=20))

            # Add severity threshold lines
            for sev, tier in SEVERITY_TIERS.items():
                fig.add_vline(x=tier["min"], line_dash="dash",
                              line_color=tier["color"],
                              annotation_text=sev)

            st.plotly_chart(fig, use_container_width=True)

    # Timeline of anomalies: group events by day to spot spikes over time
    if "timestamp" in alert_df.columns and "risk_score" in alert_df.columns:
        ts_df = alert_df.copy()
        ts_df["timestamp"] = pd.to_datetime(ts_df["timestamp"])
        ts_df["date"] = ts_df["timestamp"].dt.date
        daily = ts_df.groupby("date").agg(
            events=("entity_id", "count"),
            anomalies=("risk_score", lambda x: (x > 40).sum()),
            avg_risk=("risk_score", "mean"),
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["events"],
            name="Total Events", marker_color="#1F2937", opacity=0.5,
        ))
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["anomalies"],
            name="Anomalies", marker_color="#DC2626",
        ))
        fig.update_layout(
            title="Daily Event & Anomaly Volume",
            height=300, barmode="overlay",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 5px 0;">
        <h1 style="font-size: 2.8rem; margin-bottom: 0; letter-spacing: 3px;
                    color: #00FF88; text-shadow: 0 0 30px rgba(0,255,136,0.4), 0 0 60px rgba(0,255,136,0.1);">
            &#x1f6e1;&#xfe0f; CYBERSENTINEL
        </h1>
        <p style="color: #4B5563; font-size: 0.85rem; letter-spacing: 2px; margin-top: 4px;">
            AI-POWERED BEHAVIORAL ANOMALY DETECTION FOR CYBERSECURITY &nbsp;|&nbsp; HONEYWELL HACKATHON 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    alert_df, full_df = load_data()

    # ── Sidebar Navigation ───────────────────────────────────────────────
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select View",
        [
            "📈 Overview",
            "🚨 Alert Queue",
            "🌍 World Map",
            "📊 Entity Timeline",
            "⚡ Streaming Simulation",
            "🔄 Concept Drift",
            "🆕 Cold-Start",
            "🔬 Model Comparison",
            "📊 Model Results",
            "📄 Export Report",
        ],
    )

    # Sidebar stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")
    st.sidebar.metric("Total Events", f"{len(alert_df):,}")
    if "severity" in alert_df.columns:
        critical = len(alert_df[alert_df["severity"] == "Critical"])
        st.sidebar.metric("Critical Alerts", critical)
    st.sidebar.metric("Entities", alert_df["entity_id"].nunique())

    # Anomaly type legend: quick reference so analysts know what each label means
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Anomaly Types")
    for atype, desc in ANOMALY_DESCRIPTIONS.items():
        st.sidebar.markdown(f"**{atype}**")
        st.sidebar.caption(desc)

    # ── Render selected page ─────────────────────────────────────────────
    # Show only the page the user picked in the sidebar menu
    if page == "📈 Overview":
        render_overview(alert_df)
    elif page == "🚨 Alert Queue":
        render_alert_queue(alert_df)
    elif page == "🌍 World Map":
        render_world_map(alert_df)
    elif page == "📊 Entity Timeline":
        render_entity_timeline(alert_df, full_df)
    elif page == "⚡ Streaming Simulation":
        render_streaming_simulation(alert_df)
    elif page == "🔄 Concept Drift":
        render_concept_drift_demo()
    elif page == "🆕 Cold-Start":
        cold_start = load_cold_start_info()
        render_cold_start_panel(alert_df, cold_start)
    elif page == "🔬 Model Comparison":
        render_model_comparison(alert_df)
    elif page == "📊 Model Results":
        render_model_results(alert_df)
    elif page == "📄 Export Report":
        render_report_export(alert_df)


if __name__ == "__main__":
    main()
