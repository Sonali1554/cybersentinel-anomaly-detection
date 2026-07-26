"""
Evaluation Pipeline
====================
Computes comprehensive metrics for all 6 models:
- Isolation Forest, One-Class SVM (unsupervised baselines)
- LSTM Autoencoder, Transformer Detector (sequence-based detection)
- XGBoost, Random Forest (supervised classification)

Metrics: detection accuracy, anomaly-type classification, FPR,
Precision@K, Recall, F1, AUC-ROC, per-class breakdown, model comparison.
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, MODELS_DIR, ANOMALY_TYPES


def evaluate_all():
    """Run full evaluation suite."""
    print("=" * 70)
    print("  CyberSentinel — Evaluation Report")
    print("=" * 70)

    # Load data — alert_queue.csv has both true labels and predictions (aligned)
    alert_df = pd.read_csv(DATA_DIR / "alert_queue.csv")

    # Load label encoder
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    normal_enc = label_encoder.transform(["normal"])[0]

    # True labels from alert_queue (aligned with predictions after feature engineering reorder)
    y_true = label_encoder.transform(alert_df["label"])
    y_true_binary = (y_true != normal_enc).astype(int)

    # Predicted labels
    y_pred = label_encoder.transform(alert_df["predicted_label"].fillna("normal"))
    y_pred_binary = (y_pred != normal_enc).astype(int)
    risk_scores = alert_df["risk_score"].values

    min_len = len(alert_df)

    # ── 1. Binary Detection Metrics ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("1. BINARY ANOMALY DETECTION (Normal vs Anomaly)")
    print("─" * 70)

    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true_binary, risk_scores)
        print(f"  AUC-ROC:   {auc:.4f}")
    except Exception:
        auc = None
        print(f"  AUC-ROC:   N/A (insufficient classes)")

    # ── 2. Precision@K (top 1% of events) ────────────────────────────────
    print("\n" + "─" * 70)
    print("2. PRECISION@K (Analyst Alert Budget)")
    print("─" * 70)

    # Simulates a real SOC analyst who can only review the top K% highest-risk alerts
    for k_pct in [0.01, 0.02, 0.05]:
        k = max(1, int(len(risk_scores) * k_pct))
        top_k_indices = np.argsort(risk_scores)[-k:]
        top_k_true = y_true_binary[top_k_indices]
        precision_at_k = top_k_true.sum() / k
        print(f"  Precision@{k_pct*100:.0f}% (top {k} events): {precision_at_k:.4f}")

    # ── 3. Multi-Class Classification ────────────────────────────────────
    print("\n" + "─" * 70)
    print("3. MULTI-CLASS ANOMALY CLASSIFICATION")
    print("─" * 70)

    present_labels = sorted(set(y_true) | set(y_pred))
    target_names = [
        label_encoder.inverse_transform([i])[0] if i < len(label_encoder.classes_) else f"class_{i}"
        for i in present_labels
    ]

    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=target_names,
        zero_division=0,
    ))

    # F1 macro
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"  F1-macro:    {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")

    # ── 4. Confusion Matrix ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("4. CONFUSION MATRIX")
    print("─" * 70)

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df.to_string())

    # ── 5. False Positive Analysis ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("5. FALSE POSITIVE ANALYSIS")
    print("─" * 70)

    # FP: predicted anomaly but actually normal
    fp_mask = (y_pred_binary == 1) & (y_true_binary == 0)
    fn_mask = (y_pred_binary == 0) & (y_true_binary == 1)
    tn_mask = (y_pred_binary == 0) & (y_true_binary == 0)
    tp_mask = (y_pred_binary == 1) & (y_true_binary == 1)

    print(f"  True Positives:  {tp_mask.sum()}")
    print(f"  True Negatives:  {tn_mask.sum()}")
    print(f"  False Positives: {fp_mask.sum()}")
    print(f"  False Negatives: {fn_mask.sum()}")
    fpr = fp_mask.sum() / max(fp_mask.sum() + tn_mask.sum(), 1)
    print(f"  False Positive Rate: {fpr:.4f}")

    # ── 6. Per-Attack-Type Recall ────────────────────────────────────────
    print("\n" + "─" * 70)
    print("6. PER-ATTACK-TYPE DETECTION RECALL")
    print("─" * 70)

    for name in label_encoder.classes_:
        if name == "normal":
            continue
        enc_val = label_encoder.transform([name])[0]
        mask = y_true == enc_val
        if mask.sum() > 0:
            recall = (y_pred[mask] == enc_val).sum() / mask.sum()
            detected = (y_pred_binary[mask] == 1).sum() / mask.sum()
            print(f"  {name:25s} | Exact recall: {recall:.4f} | Binary detected: {detected:.4f} | Count: {mask.sum()}")

    # ── 7. Model Comparison (all 6 models) ─────────────────────────────
    print("\n" + "─" * 70)
    print("7. MODEL COMPARISON (ALL MODELS)")
    print("─" * 70)

    comparison = {}

    # --- Unsupervised baselines (binary: score > threshold = anomaly) ---
    try:
        from src.baseline_profiler import BaselineProfiler
        profiler = joblib.load(MODELS_DIR / "baseline_profiler.pkl") if (MODELS_DIR / "baseline_profiler.pkl").exists() else None
        if profiler is None:
            profiler = BaselineProfiler()
            profiler_data = profiler  # fallback

        # Load feature engineer to rebuild test features
        feature_eng = joblib.load(MODELS_DIR / "feature_engineer.pkl")
        test_path = DATA_DIR / "test.csv"
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            X_test_feat, y_test_feat, test_df_feat = feature_eng.transform(test_df)
            test_baseline = profiler.score(X_test_feat, test_df_feat)

            # IF binary: top 2.5% scores = anomaly
            if_scores = test_baseline["isolation_forest_score"]
            if_threshold = np.percentile(if_scores, 97.5)
            if_binary = (if_scores > if_threshold).astype(int)

            # SVM binary
            svm_scores = test_baseline["one_class_svm_score"]
            svm_threshold = np.percentile(svm_scores, 97.5)
            svm_binary = (svm_scores > svm_threshold).astype(int)

            comparison["Isolation Forest"] = {
                "Precision": precision_score(y_true_binary, if_binary, zero_division=0),
                "Recall": recall_score(y_true_binary, if_binary, zero_division=0),
                "F1-Score": f1_score(y_true_binary, if_binary, zero_division=0),
                "Accuracy": accuracy_score(y_true_binary, if_binary),
            }
            comparison["One-Class SVM"] = {
                "Precision": precision_score(y_true_binary, svm_binary, zero_division=0),
                "Recall": recall_score(y_true_binary, svm_binary, zero_division=0),
                "F1-Score": f1_score(y_true_binary, svm_binary, zero_division=0),
                "Accuracy": accuracy_score(y_true_binary, svm_binary),
            }
    except Exception as e:
        print(f"  Skipping IF/SVM comparison: {e}")

    # --- LSTM Autoencoder ---
    try:
        from src.detection_model import LSTMAutoencoder
        lstm = LSTMAutoencoder()
        lstm.load()
        entity_ids_test = alert_df["entity_id"].values
        lstm_preds = lstm.predict(
            feature_eng.scaler.transform(
                alert_df[feature_eng.get_feature_names()].values.astype(float)
            ) if hasattr(feature_eng, 'scaler') else np.zeros((len(alert_df), 28)),
            entity_ids_test
        )
        comparison["LSTM Autoencoder"] = {
            "Precision": precision_score(y_true_binary, lstm_preds, zero_division=0),
            "Recall": recall_score(y_true_binary, lstm_preds, zero_division=0),
            "F1-Score": f1_score(y_true_binary, lstm_preds, zero_division=0),
            "Accuracy": accuracy_score(y_true_binary, lstm_preds),
        }
    except Exception as e:
        print(f"  Skipping LSTM comparison: {e}")

    # --- Transformer Detector ---
    try:
        from src.detection_model import TransformerDetector
        transformer = TransformerDetector()
        transformer.load()
        transformer_preds = transformer.predict(
            feature_eng.scaler.transform(
                alert_df[feature_eng.get_feature_names()].values.astype(float)
            ) if hasattr(feature_eng, 'scaler') else np.zeros((len(alert_df), 28)),
            entity_ids_test
        )
        comparison["Transformer"] = {
            "Precision": precision_score(y_true_binary, transformer_preds, zero_division=0),
            "Recall": recall_score(y_true_binary, transformer_preds, zero_division=0),
            "F1-Score": f1_score(y_true_binary, transformer_preds, zero_division=0),
            "Accuracy": accuracy_score(y_true_binary, transformer_preds),
        }
    except Exception as e:
        print(f"  Skipping Transformer comparison: {e}")

    # --- XGBoost (already computed above) ---
    comparison["XGBoost"] = {
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Accuracy": round(acc, 4),
    }

    # --- Random Forest ---
    try:
        rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
        # Rebuild augmented features for RF
        from src.anomaly_classifier import AnomalyClassifier
        temp_clf = AnomalyClassifier(label_encoder=label_encoder)

        if test_path.exists():
            test_baseline_scores = profiler.score(X_test_feat, test_df_feat)
            X_test_aug = temp_clf._augment_features(X_test_feat, test_baseline_scores)
            rf_preds = rf_model.predict(X_test_aug)
            rf_binary = (rf_preds != normal_enc).astype(int)

            comparison["Random Forest"] = {
                "Precision": precision_score(y_true_binary, rf_binary, zero_division=0),
                "Recall": recall_score(y_true_binary, rf_binary, zero_division=0),
                "F1-Score": f1_score(y_true_binary, rf_binary, zero_division=0),
                "Accuracy": accuracy_score(y_true_binary, rf_binary),
            }
    except Exception as e:
        print(f"  Skipping RF comparison: {e}")

    # Print comparison table
    if comparison:
        comp_df = pd.DataFrame(comparison).T
        comp_df = comp_df.round(4)
        print("\n  MODEL COMPARISON TABLE:")
        print(comp_df.to_string())

        comp_df.to_csv(DATA_DIR / "model_comparison_metrics.csv")
        print(f"\n  Comparison saved to {DATA_DIR / 'model_comparison_metrics.csv'}")

    # ── 8. Severity Distribution ─────────────────────────────────────────
    print("\n" + "─" * 70)
    print("8. SEVERITY DISTRIBUTION")
    print("─" * 70)

    if "severity" in alert_df.columns:
        for sev in ["Critical", "High", "Medium", "Low"]:
            sev_mask = alert_df["severity"].iloc[:min_len] == sev
            count = sev_mask.sum()
            if count > 0:
                true_anomaly_rate = y_true_binary[sev_mask.values[:min_len]].mean()
                print(f"  {sev:10s} | Count: {count:5d} | True anomaly rate: {true_anomaly_rate:.4f}")

    # ── 9. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    metrics = {
        "binary_accuracy": round(acc, 4),
        "binary_precision": round(prec, 4),
        "binary_recall": round(rec, 4),
        "binary_f1": round(f1, 4),
        "auc_roc": round(auc, 4) if auc else None,
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "false_positive_rate": round(fpr, 4),
        "total_test_events": int(min_len),
        "total_anomalies": int(y_true_binary.sum()),
        "anomaly_rate": round(y_true_binary.mean(), 4),
    }

    for key, val in metrics.items():
        print(f"  {key:30s}: {val}")

    # Save metrics
    with open(DATA_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to {DATA_DIR / 'evaluation_metrics.json'}")

    return metrics


if __name__ == "__main__":
    evaluate_all()
