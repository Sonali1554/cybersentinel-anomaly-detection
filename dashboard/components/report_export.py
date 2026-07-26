"""PDF Report Export Component — Generate downloadable PDF summary."""

import io
import json
from datetime import datetime

import streamlit as st
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_DIR, SEVERITY_TIERS


def _generate_pdf(alert_df, metrics=None):
    """Generate a PDF report using FPDF2."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Title Page ───────────────────────────────────────────────────────
    # Cover page with the product name and when/how much data this report covers
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "CyberSentinel", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "AI-Powered Behavioral Anomaly Detection", ln=True, align="C")
    pdf.cell(0, 10, "Security Analysis Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 8, f"Total Events Analyzed: {len(alert_df)}", ln=True, align="C")

    # ── Executive Summary ────────────────────────────────────────────────
    # A plain-language paragraph summarizing the findings, for non-technical readers
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_draw_color(59, 130, 246)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 10)
    sev_counts = {}
    for sev in ["Critical", "High", "Medium", "Low"]:
        sev_counts[sev] = len(alert_df[alert_df["severity"] == sev]) if "severity" in alert_df.columns else 0

    summary_text = (
        f"This report summarizes the anomaly detection analysis across {len(alert_df)} access events. "
        f"The system identified {sev_counts['Critical']} critical alerts, {sev_counts['High']} high-priority alerts, "
        f"{sev_counts['Medium']} medium alerts, and {sev_counts['Low']} low-priority alerts. "
        f"The analysis covers brute force attacks, impossible travel, credential stuffing, lateral movement, "
        f"device spoofing, low-and-slow exfiltration, and insider drift patterns."
    )
    pdf.multi_cell(0, 6, summary_text)

    # ── Alert Severity Breakdown ─────────────────────────────────────────
    # Table showing count and share of alerts at each severity level
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Alert Severity Breakdown", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, "Severity", 1)
    pdf.cell(40, 8, "Count", 1)
    pdf.cell(50, 8, "Percentage", 1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    total = max(len(alert_df), 1)
    for sev in ["Critical", "High", "Medium", "Low"]:
        pdf.cell(60, 8, sev, 1)
        pdf.cell(40, 8, str(sev_counts[sev]), 1)
        pdf.cell(50, 8, f"{sev_counts[sev]/total*100:.1f}%", 1)
        pdf.ln()

    # ── Top Alerts ───────────────────────────────────────────────────────
    # Detailed list of the most dangerous individual events, for follow-up investigation
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Top 15 Critical Alerts", ln=True)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    top_alerts = alert_df.sort_values("risk_score", ascending=False).head(15)

    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(25, 7, "Entity", 1)
    pdf.cell(15, 7, "Score", 1)
    pdf.cell(18, 7, "Severity", 1)
    pdf.cell(30, 7, "Type", 1)
    pdf.cell(25, 7, "City", 1)
    pdf.cell(40, 7, "Resource", 1)
    pdf.cell(35, 7, "Timestamp", 1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for _, row in top_alerts.iterrows():
        pdf.cell(25, 6, str(row.get("entity_id", ""))[:12], 1)
        pdf.cell(15, 6, f"{row.get('risk_score', 0):.1f}", 1)
        pdf.cell(18, 6, str(row.get("severity", "")), 1)
        pdf.cell(30, 6, str(row.get("predicted_label", ""))[:18], 1)
        pdf.cell(25, 6, str(row.get("geo_city", ""))[:15], 1)
        pdf.cell(40, 6, str(row.get("resource_accessed", ""))[:25], 1)
        pdf.cell(35, 6, str(row.get("timestamp", ""))[:19], 1)
        pdf.ln()

    # ── Anomaly Type Distribution ────────────────────────────────────────
    # Breakdown of which attack/anomaly categories showed up and how often
    if "predicted_label" in alert_df.columns:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Anomaly Type Distribution", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        type_counts = alert_df[alert_df["predicted_label"] != "normal"]["predicted_label"].value_counts()

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(60, 8, "Anomaly Type", 1)
        pdf.cell(40, 8, "Count", 1)
        pdf.cell(50, 8, "Percentage of Anomalies", 1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 10)
        total_anom = max(type_counts.sum(), 1)
        for atype, count in type_counts.items():
            pdf.cell(60, 8, str(atype), 1)
            pdf.cell(40, 8, str(count), 1)
            pdf.cell(50, 8, f"{count/total_anom*100:.1f}%", 1)
            pdf.ln()

    # ── Model Performance ────────────────────────────────────────────────
    # Include the model's evaluation numbers (accuracy, F1, etc.) if they were computed
    if metrics:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Model Performance Metrics", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 10)
        for key, val in metrics.items():
            if val is not None:
                pdf.cell(80, 7, key.replace("_", " ").title(), 1)
                pdf.cell(40, 7, str(val), 1)
                pdf.ln()

    # ── Footer ───────────────────────────────────────────────────────────
    pdf.ln(15)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Report generated by CyberSentinel - AI-Powered Behavioral Anomaly Detection System", ln=True, align="C")
    pdf.cell(0, 5, "Honeywell Hackathon 2026", ln=True, align="C")

    return bytes(pdf.output())


def render_report_export(alert_df):
    """Render the PDF report export section."""
    st.subheader("📄 Report Export")

    st.markdown("Generate a downloadable PDF report with current alert analysis and metrics.")

    # Load metrics if available, to show a preview and include them in the PDF
    metrics = {}
    metrics_path = DATA_DIR / "evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Report includes:**")
        st.markdown("""
        - Executive summary
        - Alert severity breakdown
        - Top 15 critical alerts with details
        - Anomaly type distribution
        - Model performance metrics
        """)

    with col2:
        st.write("**Current Statistics:**")
        st.write(f"- Total events: {len(alert_df)}")
        if "severity" in alert_df.columns:
            st.write(f"- Critical alerts: {len(alert_df[alert_df['severity'] == 'Critical'])}")
            st.write(f"- High alerts: {len(alert_df[alert_df['severity'] == 'High'])}")
        if metrics:
            st.write(f"- AUC-ROC: {metrics.get('auc_roc', 'N/A')}")
            st.write(f"- F1 Score: {metrics.get('binary_f1', 'N/A')}")

    # Build the PDF only when the button is clicked, since it can take a moment to generate
    if st.button("📥 Generate PDF Report", type="primary"):
        with st.spinner("Generating report..."):
            try:
                pdf_bytes = _generate_pdf(alert_df, metrics)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"CyberSentinel_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
                st.success("Report generated successfully!")
            except ImportError:
                st.error("FPDF2 not installed. Run: `pip install fpdf2`")
            except Exception as e:
                st.error(f"Error generating report: {e}")
