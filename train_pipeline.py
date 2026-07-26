"""
Training Pipeline
==================
End-to-end orchestrator that:
1. Generates synthetic data (Deliverable 1)
2. Engineers features
3. Trains baseline profiler — IF + SVM (Deliverable 2)
4. Trains LSTM Autoencoder (Deliverable 3)
4b. Trains Transformer Detector (Deliverable 3 — comparison)
5. Trains XGBoost classifier (Deliverable 4)
5b. Trains Random Forest classifier (Deliverable 4 — comparison)
6. Initializes explainability engine (Deliverable 5)
7. Sets up cold-start and concept drift handlers

6 models total: IF, SVM, LSTM, Transformer, XGBoost, Random Forest

Run: python train_pipeline.py
"""

import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, MODELS_DIR, ANOMALY_TYPES
from src.data_generator import SyntheticDataGenerator
from src.feature_engineering import FeatureEngineer
from src.baseline_profiler import BaselineProfiler
from src.detection_model import LSTMAutoencoder, TransformerDetector
from src.anomaly_classifier import AnomalyClassifier
from src.explainability import ExplainabilityEngine
from src.cold_start import ColdStartHandler
from src.concept_drift import ConceptDriftAdapter
from src.utils import compute_risk_score, assign_severity, compute_entity_risk_history


def main():
    start_time = time.time()
    print("=" * 70)
    print("  CyberSentinel — Training Pipeline")
    print("  AI-Powered Behavioral Anomaly Detection for Cybersecurity")
    print("=" * 70)

    # ── Step 1: Generate Synthetic Data ──────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1: Generating Synthetic Data (Deliverable 1)")
    print("─" * 70)

    generator = SyntheticDataGenerator()
    df = generator.generate()
    train_df, test_df = generator.save(df)

    # ── Step 2: Feature Engineering ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2: Feature Engineering")
    print("─" * 70)

    feature_eng = FeatureEngineer()
    X_train, y_train, train_df_feat = feature_eng.fit_transform(train_df)
    X_test, y_test, test_df_feat = feature_eng.transform(test_df)

    # Save feature engineer
    joblib.dump(feature_eng, MODELS_DIR / "feature_engineer.pkl")

    print(f"  Feature matrix shape: train={X_train.shape}, test={X_test.shape}")
    print(f"  Feature names: {feature_eng.get_feature_names()[:5]}... ({len(feature_eng.get_feature_names())} total)")

    # ── Step 3: Baseline Profiler ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 3: Training Baseline Profiler (Deliverable 2)")
    print("─" * 70)

    profiler = BaselineProfiler()
    profiler.fit(X_train, train_df_feat)

    # Score training data
    train_baseline_scores = profiler.score(X_train, train_df_feat)
    test_baseline_scores = profiler.score(X_test, test_df_feat)

    # Save profiler so evaluate.py can reload it for model comparison
    joblib.dump(profiler, MODELS_DIR / "baseline_profiler.pkl")

    print(f"  Baseline scoring complete.")
    print(f"  Mean IF score (train): {train_baseline_scores['isolation_forest_score'].mean():.4f}")
    print(f"  Mean IF score (test): {test_baseline_scores['isolation_forest_score'].mean():.4f}")

    # ── Step 4: LSTM Autoencoder ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 4: Training LSTM Autoencoder (Deliverable 3)")
    print("─" * 70)

    lstm = LSTMAutoencoder()
    entity_ids_train = train_df_feat["entity_id"].values
    entity_ids_test = test_df_feat["entity_id"].values

    lstm.fit(X_train, entity_ids_train, y_train)

    # Score
    lstm_train_scores = lstm.score(X_train, entity_ids_train)
    lstm_test_scores = lstm.score(X_test, entity_ids_test)
    print(f"  Mean LSTM score (train): {lstm_train_scores.mean():.4f}")
    print(f"  Mean LSTM score (test): {lstm_test_scores.mean():.4f}")

    # ── Step 4b: Transformer Detector ──────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 4b: Training Transformer Detector (Deliverable 3 — comparison)")
    print("─" * 70)

    transformer = TransformerDetector()
    transformer.fit(X_train, entity_ids_train, y_train)

    # Score
    transformer_train_scores = transformer.score(X_train, entity_ids_train)
    transformer_test_scores = transformer.score(X_test, entity_ids_test)
    print(f"  Mean Transformer score (train): {transformer_train_scores.mean():.4f}")
    print(f"  Mean Transformer score (test): {transformer_test_scores.mean():.4f}")

    # ── Step 5: Anomaly Classifier ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 5: Training Anomaly Classifier — XGBoost (Deliverable 4)")
    print("─" * 70)

    classifier = AnomalyClassifier(label_encoder=feature_eng.label_encoder)
    classifier.fit(X_train, y_train, baseline_scores=train_baseline_scores)

    # Evaluate
    results = classifier.evaluate(X_test, y_test, baseline_scores=test_baseline_scores)

    # ── Step 5b: Random Forest Classifier ──────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 5b: Training Random Forest Classifier (Deliverable 4 — comparison)")
    print("─" * 70)

    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE as SMOTE_RF
    from config import CLASSIFIER_CONFIG

    # Same augmented features as XGBoost
    X_train_aug = classifier._augment_features(X_train, train_baseline_scores)
    X_test_aug = classifier._augment_features(X_test, test_baseline_scores)

    # SMOTE for class balance
    smote_rf = SMOTE_RF(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote_rf.fit_resample(X_train_aug, y_train)
    print(f"  After SMOTE: {len(X_train_sm)} samples")

    rf_model = RandomForestClassifier(**CLASSIFIER_CONFIG["random_forest"])
    print(f"  Training Random Forest ({CLASSIFIER_CONFIG['random_forest']['n_estimators']} trees)...")
    rf_model.fit(X_train_sm, y_train_sm)

    # Evaluate RF
    rf_preds = rf_model.predict(X_test_aug)
    from sklearn.metrics import classification_report as clf_report
    present_labels_rf = sorted(set(y_test) | set(rf_preds))
    target_names_rf = [
        feature_eng.label_encoder.inverse_transform([i])[0] if i < len(feature_eng.label_encoder.classes_) else f"class_{i}"
        for i in present_labels_rf
    ]
    print("\n" + "=" * 70)
    print("RANDOM FOREST CLASSIFICATION REPORT")
    print("=" * 70)
    print(clf_report(y_test, rf_preds, labels=present_labels_rf,
                     target_names=target_names_rf, zero_division=0))

    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    print("[RF] Model saved.")

    # ── Step 6: Explainability Engine ────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 6: Initializing Explainability Engine (Deliverable 5)")
    print("─" * 70)

    explainer = ExplainabilityEngine()

    # Build augmented feature names (original + baseline scores)
    aug_feature_names = feature_eng.get_feature_names() + [
        "statistical_score", "isolation_forest_score",
        "one_class_svm_score", "baseline_combined_score"
    ]
    explainer.initialize_shap(classifier.model, aug_feature_names)

    # Generate sample explanations
    print("\n  Sample explanations (first 3 anomalies from test set):")
    normal_enc = feature_eng.label_encoder.transform(["normal"])[0]
    anomaly_mask = y_test != normal_enc
    if anomaly_mask.any():
        anomaly_indices = np.where(anomaly_mask)[0][:3]
        X_test_aug = classifier._augment_features(X_test, test_baseline_scores)
        for idx in anomaly_indices:
            explanation = explainer.explain_event(
                X_test_aug[idx],
                event_data=test_df_feat.iloc[idx],
                top_n=3,
            )
            label = test_df_feat.iloc[idx]["label"]
            print(f"\n  Event {idx} (label: {label}):")
            print(f"    Summary: {explanation['risk_summary']}")
            for factor in explanation["contributing_factors"][:3]:
                print(f"    → {factor['description']}")

    # Save explainer
    joblib.dump(explainer, MODELS_DIR / "explainer.pkl")

    # ── Step 7: Cold-Start Handler ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 7: Setting Up Cold-Start Handler")
    print("─" * 70)

    cold_start = ColdStartHandler()
    cold_start.build_peer_groups(profiler.stat_profiler)
    joblib.dump(cold_start, MODELS_DIR / "cold_start.pkl")

    # ── Step 8: Concept Drift Adapter ────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 8: Setting Up Concept Drift Adapter")
    print("─" * 70)

    drift_adapter = ConceptDriftAdapter()
    demo_data = drift_adapter.demonstrate_adaptation("demo_entity")
    print(f"  Drift demo generated: {len(demo_data['actual_values'])} events across 3 phases")
    joblib.dump(drift_adapter, MODELS_DIR / "drift_adapter.pkl")

    # ── Step 9: Compute Risk Scores & Build Alert Queue ──────────────────
    print("\n" + "─" * 70)
    print("STEP 9: Computing Risk Scores & Alert Queue")
    print("─" * 70)

    # Entity risk history: how risky has this entity looked in the past (learned from training data)
    entity_risk = compute_entity_risk_history(train_df_feat)
    # Entities never seen in training (brand new) default to 0 prior risk
    test_entity_risk = np.array([
        entity_risk.get(eid, 0.0) for eid in test_df_feat["entity_id"]
    ])

    # Classifier confidence
    test_confidence = classifier.get_confidence(X_test, test_baseline_scores)

    # Risk scores
    risk_scores = compute_risk_score(
        test_baseline_scores, lstm_test_scores, test_confidence, test_entity_risk
    )
    severities = assign_severity(risk_scores)

    # Save risk data for dashboard
    alert_data = test_df_feat.copy()
    alert_data["risk_score"] = risk_scores[:len(test_df_feat)]
    alert_data["severity"] = severities[:len(test_df_feat)]
    alert_data["predicted_label_enc"] = results["predictions"][:len(test_df_feat)]

    # Map predicted labels back to names
    label_enc = feature_eng.label_encoder
    alert_data["predicted_label"] = [
        label_enc.inverse_transform([p])[0] if p < len(label_enc.classes_) else "unknown"
        for p in results["predictions"][:len(test_df_feat)]
    ]

    alert_data.to_csv(DATA_DIR / "alert_queue.csv", index=False)

    # Print summary
    print(f"\n  Alert Summary:")
    for sev in ["Critical", "High", "Medium", "Low"]:
        count = sum(1 for s in severities if s == sev)
        print(f"    {sev}: {count}")

    print(f"\n  Top 5 alerts:")
    top_alerts = alert_data.nlargest(5, "risk_score")
    for _, row in top_alerts.iterrows():
        print(f"    Score: {row['risk_score']:.1f} | {row['severity']} | "
              f"{row['entity_id']} | {row['predicted_label']} | {row.get('geo_city', 'N/A')}")

    # ── Complete ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Training Pipeline Complete! ({elapsed:.1f} seconds)")
    print("=" * 70)
    print(f"\n  Generated files in {DATA_DIR}:")
    for f in sorted(DATA_DIR.iterdir()):
        size = f.stat().st_size / 1024
        print(f"    {f.name}: {size:.1f} KB")

    print(f"\n  Saved models in {MODELS_DIR}:")
    for f in sorted(MODELS_DIR.iterdir()):
        size = f.stat().st_size / 1024
        print(f"    {f.name}: {size:.1f} KB")

    print("\n  Next: Run the dashboard with 'streamlit run dashboard/app.py'")


if __name__ == "__main__":
    main()
