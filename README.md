# 🛡️ CyberSentinel — AI-Powered Behavioral Anomaly Detection for Cybersecurity

## Honeywell Hackathon 2026 | Q4

An AI/ML system that models "normal" access and connection behaviour for users and devices, detects intrusions or compromised-credential activity in near real-time, and classifies the type of anomaly with an explainable risk score.

---

## Quick Start

## 🚀 Live Demo

🌍 **Experience CyberSentinel AI here:**

👉 **https://cybersentinel-anomaly-detection.onrender.com**

## 🎥 Project Demo

📺 **Watch the complete project walkthrough and live demonstration**

▶️ **Demo Video:**  
https://drive.google.com/file/d/1pWk3mUrOpDxayyCU1uWudkfTgAxAWX5S/view?usp=sharing

💡 *The demo showcases the complete workflow—from anomaly detection and AI models to the interactive Streamlit dashboard and explainable risk analysis.*


### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline

This generates synthetic data, trains all models, and produces the alert queue:

```bash
python train_pipeline.py
```

### 3. Evaluate Models

```bash
python evaluate.py
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## Deliverables

| # | Deliverable | Location | Status |
|---|-------------|----------|--------|
| 1 | Synthetic Data Generator | `src/data_generator.py` | ✅ |
| 2 | Baseline Profiling Model | `src/baseline_profiler.py` | ✅ |
| 3 | Detection Model (LSTM Autoencoder) | `src/detection_model.py` | ✅ |
| 4 | Anomaly Classifier (XGBoost) | `src/anomaly_classifier.py` | ✅ |
| 5 | Explainability Layer (SHAP) | `src/explainability.py` | ✅ |
| 6 | Analyst Dashboard (Streamlit) | `dashboard/app.py` | ✅ |
| 7 | Report | `report/report.md` | ✅ |

### Extra Features

- ✅ Real-time streaming simulation mode
- ✅ World map for impossible travel visualization
- ✅ Entity behavior timeline
- ✅ Concept drift adaptation demo
- ✅ Cold-start handling via peer group profiling
- ✅ Alert severity tiers (Critical/High/Medium/Low)
- ✅ Model comparison toggle (IF / LSTM / XGBoost)
- ✅ PDF report export button

---

## Project Structure

```
cybersecurity_anomaly_detection/
├── config.py                 # Central configuration
├── requirements.txt          # Dependencies
├── train_pipeline.py         # End-to-end training
├── evaluate.py               # Evaluation & metrics
├── src/
│   ├── data_generator.py     # Deliverable 1
│   ├── feature_engineering.py
│   ├── baseline_profiler.py  # Deliverable 2
│   ├── detection_model.py    # Deliverable 3
│   ├── anomaly_classifier.py # Deliverable 4
│   ├── explainability.py     # Deliverable 5
│   ├── cold_start.py
│   ├── concept_drift.py
│   └── utils.py
├── dashboard/
│   ├── app.py                # Deliverable 6
│   └── components/
│       ├── alert_queue.py
│       ├── world_map.py
│       ├── entity_timeline.py
│       ├── streaming_sim.py
│       ├── concept_drift_demo.py
│       ├── model_comparison.py
│       └── report_export.py
└── report/
    └── report.md             # Deliverable 7
```

---

## Attack Types Detected

| Attack | Description |
|--------|-------------|
| Brute Force | Rapid repeated failed-auth attempts from one source |
| Impossible Travel | Logins from distant locations in implausible time |
| Credential Stuffing | Few IPs targeting many entities with high failure rate |
| Lateral Movement | Entity accessing unusual resources never touched before |
| Device Spoofing | Device with mismatched OS/MAC fingerprint |
| Low-and-Slow Exfiltration | Gradual off-hours data access over weeks |
| Insider Drift | Slowly expanding privilege footprint (edge case) |

---

## Tech Stack

- **Data**: NumPy, Pandas, Faker
- **ML**: scikit-learn, TensorFlow/Keras, XGBoost
- **Explainability**: SHAP
- **Dashboard**: Streamlit, Plotly
- **PDF Export**: FPDF2

---

## Team

Honeywell Hackathon 2026 — Question 4
