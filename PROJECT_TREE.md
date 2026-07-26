# Project Tree Structure

```
cybersecurity_anomaly_detection/
│
├── PLAN.md                          # Project plan document (this file)
├── PROJECT_TREE.md                  # File/folder structure reference
├── README.md                        # Project documentation & setup guide
├── requirements.txt                 # Python dependencies
├── config.py                        # Central configuration (paths, hyperparams, constants)
├── train_pipeline.py                # End-to-end training orchestrator
├── evaluate.py                      # Evaluation & metrics computation
│
├── data/                            # Generated synthetic data (created at runtime)
│   ├── access_logs.csv              # Full synthetic dataset
│   ├── train.csv                    # Training split
│   ├── test.csv                     # Test split
│   └── entity_profiles.json         # Per-entity behavioral profiles
│
├── models/                          # Saved trained models (created at runtime)
│   ├── isolation_forest.pkl         # Baseline Isolation Forest
│   ├── oneclasssvm.pkl              # Baseline One-Class SVM
│   ├── lstm_autoencoder.h5          # LSTM Autoencoder weights
│   ├── xgb_classifier.pkl           # XGBoost anomaly classifier
│   ├── scaler.pkl                   # Feature scaler
│   └── label_encoder.pkl            # Label encoder for anomaly types
│
├── src/                             # Core source code
│   ├── __init__.py                  # Package initializer
│   ├── data_generator.py            # [Deliverable 1] Synthetic data generator
│   ├── feature_engineering.py       # Feature extraction & encoding pipeline
│   ├── baseline_profiler.py         # [Deliverable 2] Statistical + Isolation Forest profiler
│   ├── detection_model.py           # [Deliverable 3] LSTM Autoencoder detector
│   ├── anomaly_classifier.py        # [Deliverable 4] XGBoost multi-class classifier
│   ├── explainability.py            # [Deliverable 5] SHAP-based explanation engine
│   ├── cold_start.py                # Cold-start handling via peer group profiling
│   ├── concept_drift.py             # Concept drift detection & adaptation
│   └── utils.py                     # Shared utilities (logging, I/O, scoring)
│
├── dashboard/                       # Streamlit analyst dashboard
│   ├── app.py                       # [Deliverable 6] Main Streamlit application
│   └── components/                  # Dashboard sub-components
│       ├── __init__.py              # Components package init
│       ├── alert_queue.py           # Ranked alert queue with severity tiers
│       ├── world_map.py             # Impossible travel world map (Plotly geo)
│       ├── entity_timeline.py       # Entity behavior timeline view
│       ├── streaming_sim.py         # Real-time streaming simulation
│       ├── concept_drift_demo.py    # Concept drift visualization
│       ├── model_comparison.py      # Model comparison toggle panel
│       └── report_export.py         # PDF report export functionality
│
└── report/
    └── report.md                    # [Deliverable 7] Assumptions, metrics, limitations
```

## File Count: 27 files across 6 directories
## Deliverable Coverage: All 7 required + 8 extras
