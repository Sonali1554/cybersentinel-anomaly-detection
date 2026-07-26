"""World Map Component — Impossible travel visualization using Plotly geo."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SEVERITY_TIERS


def _parse_geo(geo_str):
    """Parse 'lat,lon' string."""
    try:
        parts = str(geo_str).split(",")
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        # Bad or missing coordinates are dropped later rather than crashing the app
        return None, None


def render_world_map(alert_df):
    """Render interactive world map showing anomaly locations and impossible travel paths."""
    st.subheader("🌍 Geographic Anomaly Map")

    # Parse coordinates: split the raw "lat,lon" string into separate numeric columns
    df = alert_df.copy()
    geo_parsed = df["geo_location"].apply(lambda x: _parse_geo(x))
    df["lat"] = geo_parsed.apply(lambda x: x[0])
    df["lon"] = geo_parsed.apply(lambda x: x[1])
    df = df.dropna(subset=["lat", "lon"])

    if len(df) == 0:
        st.warning("No events with valid geo-coordinates found.")
        return

    # Toggle to include Low-severity events
    show_low = st.checkbox("Show Low-severity events", value=False,
                           help="Low-severity events are normal behavior — enable to see all activity on the map")

    sevs_to_show = ["Critical", "High", "Medium"]
    if show_low:
        sevs_to_show.append("Low")

    anomaly_df = df[df["severity"].isin(sevs_to_show)].copy()

    # Color map by severity
    color_map = {
        "Critical": "#FF0000",
        "High": "#FF8C00",
        "Medium": "#00BFFF",
        "Low": "#888888",
    }

    fig = go.Figure()

    # ── Plot all anomaly points ──────────────────────────────────────────
    # Draw lower severity first (bottom), higher on top
    draw_order = ["Low", "Critical", "High", "Medium"] if show_low else ["Critical", "High", "Medium"]
    for sev in draw_order:
        sev_df = anomaly_df[anomaly_df["severity"] == sev].copy()
        if len(sev_df) == 0:
            continue

        if sev == "Low":
            sev_df = sev_df.sample(n=min(200, len(sev_df)), random_state=42)
            dot_size = 5
            dot_opacity = 0.4
            border_width = 0
        elif sev == "Critical":
            dot_size = 14
            dot_opacity = 1.0
            border_width = 2
        elif sev == "High":
            dot_size = 10
            dot_opacity = 0.95
            border_width = 1
        else:
            dot_size = 7
            dot_opacity = 0.8
            border_width = 1

        rng = np.random.RandomState(42)
        jitter_range = 4.0 if sev == "Low" else 1.5
        jitter_lat = sev_df["lat"].values + rng.uniform(-jitter_range, jitter_range, len(sev_df))
        jitter_lon = sev_df["lon"].values + rng.uniform(-jitter_range, jitter_range, len(sev_df))

        fig.add_trace(go.Scattergeo(
            lat=jitter_lat,
            lon=jitter_lon,
            mode="markers",
            marker=dict(
                size=dot_size,
                color=color_map[sev],
                opacity=dot_opacity,
                line=dict(width=border_width, color="#00FF88"),
                sizemode="diameter",
            ),
            text=sev_df.apply(
                lambda r: f"Entity: {r['entity_id']}<br>"
                          f"Type: {r.get('predicted_label', 'N/A')}<br>"
                          f"Risk: {r['risk_score']:.1f}<br>"
                          f"City: {r.get('geo_city', 'N/A')}<br>"
                          f"Time: {r['timestamp']}",
                axis=1,
            ),
            hoverinfo="text",
            name=sev,
        ))

    # ── Draw impossible travel lines ─────────────────────────────────────
    # Connect consecutive logins for the same entity to show suspiciously fast location jumps
    if "predicted_label" in df.columns:
        travel_df = df[df["predicted_label"] == "impossible_travel"].copy()
        if len(travel_df) > 0:
            travel_df = travel_df.sort_values(["entity_id", "timestamp"])
            for entity_id, group in travel_df.groupby("entity_id"):
                if len(group) >= 2:
                    for i in range(len(group) - 1):
                        row1 = group.iloc[i]
                        row2 = group.iloc[i + 1]
                        fig.add_trace(go.Scattergeo(
                            lat=[row1["lat"], row2["lat"]],
                            lon=[row1["lon"], row2["lon"]],
                            mode="lines",
                            line=dict(width=3, color="#DC2626", dash="dash"),
                            name=f"Travel: {entity_id}",
                            showlegend=False,
                            hoverinfo="text",
                            text=f"Impossible Travel: {entity_id}<br>"
                                 f"From: {row1.get('geo_city', 'N/A')}<br>"
                                 f"To: {row2.get('geo_city', 'N/A')}",
                        ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="#111827",
        showocean=True, oceancolor="#0A0E17",
        showcountries=True, countrycolor="#1F2937",
        showcoastlines=True, coastlinecolor="#374151",
        showframe=False,
        bgcolor="rgba(0,0,0,0)",
        showlakes=True, lakecolor="#0A0E17",
    )
    fig.update_layout(
        title="Anomaly Geo-Distribution & Impossible Travel Paths",
        height=550,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            font=dict(size=13),
        ),
        legend_itemclick="toggle",
        legend_itemdoubleclick="toggleothers",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats: headline numbers about impossible travel for quick reference below the map
    if "predicted_label" in df.columns:
        travel_count = len(df[df["predicted_label"] == "impossible_travel"])
        st.info(f"Impossible travel events detected: **{travel_count}** | "
                f"Unique entities involved: **{df[df['predicted_label'] == 'impossible_travel']['entity_id'].nunique()}**")
