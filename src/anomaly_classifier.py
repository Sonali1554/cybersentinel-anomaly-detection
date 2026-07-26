"""
DELIVERABLE 4: Anomaly Classification
=======================================
Not just "anomalous" — classifies WHICH attack category an event resembles.

Uses XGBoost multi-class classifier with:
- SMOTE oversampling to handle extreme class imbalance
- Feature augmentation with baseline model scores
- Per-class confidence scores for explainability

Classes: normal, brute_force, impossible_travel, credential_stuffing,
         lateral_movement, device_spoofing, low_and_slow, insider_drift
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASSIFIER_CONFIG, MODELS_DIR, ANOMALY_TYPES


class AnomalyClassifier:
    """
    Multi-class anomaly type classifier.
    Classifies flagged events into specific attack categories.
    """

    def __init__(self, config=None, label_encoder=None):
        self.config = config or CLASSIFIER_CONFIG
        # XGBoost multi-class model: predicts WHICH attack type, not just anomaly/normal
        self.model = XGBClassifier(
            **self.config["xgboost"],
            num_class=len(ANOMALY_TYPES),
        )
        self.fitted = False
        self.label_encoder = label_encoder
        self.class_names = list(label_encoder.classes_) if label_encoder else ANOMALY_TYPES

    def fit(self, X_train, y_train, baseline_scores=None):
        """
        Train the classifier on labeled data.
        X_train: feature matrix
        y_train: encoded labels (0=normal, 1=brute_force, etc.)
        baseline_scores: optional dict of baseline model scores to augment features
        """
        # Augment features with baseline scores if available
        X_aug = self._augment_features(X_train, baseline_scores)

        # Handle class imbalance with SMOTE
        print("[Classifier] Applying SMOTE for class imbalance...")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(**self.config["smote"])
            X_resampled, y_resampled = smote.fit_resample(X_aug, y_train)
            print(f"  Before SMOTE: {len(X_aug)} samples")
            print(f"  After SMOTE: {len(X_resampled)} samples")
        except Exception as e:
            print(f"  SMOTE failed ({e}), using original data with class weights")
            X_resampled, y_resampled = X_aug, y_train
            # Compute sample weights for imbalanced classes
            unique, counts = np.unique(y_train, return_counts=True)
            total = len(y_train)
            class_weights = {c: total / (len(unique) * cnt) for c, cnt in zip(unique, counts)}
            sample_weights = np.array([class_weights[y] for y in y_train])
            self.model.set_params(sample_weight=sample_weights)

        # Print class distribution
        print(f"\n[Classifier] Class distribution (after resampling):")
        unique, counts = np.unique(y_resampled, return_counts=True)
        for c, cnt in zip(unique, counts):
            name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            print(f"  {name}: {cnt}")

        # Train
        print(f"\n[Classifier] Training XGBoost ({self.config['xgboost']['n_estimators']} trees)...")
        self.model.fit(X_resampled, y_resampled, verbose=True)
        self.fitted = True

        # Save model
        self._save()
        print("[Classifier] Model trained and saved.")
        return self

    def predict(self, X, baseline_scores=None):
        """Predict anomaly type for each event."""
        assert self.fitted, "Must call fit() first"
        X_aug = self._augment_features(X, baseline_scores)
        return self.model.predict(X_aug)

    def predict_proba(self, X, baseline_scores=None):
        """Get probability distribution over anomaly types."""
        assert self.fitted, "Must call fit() first"
        X_aug = self._augment_features(X, baseline_scores)
        return self.model.predict_proba(X_aug)

    def get_confidence(self, X, baseline_scores=None):
        """Get anomaly confidence: 1 - P(normal). Higher = more likely anomalous."""
        proba = self.predict_proba(X, baseline_scores)
        classes = list(self.model.classes_)
        # "normal" is encoded as the highest label value (alphabetically last) — find its column
        normal_class = max(classes)
        normal_idx = classes.index(normal_class)
        return 1.0 - proba[:, normal_idx]

    def evaluate(self, X_test, y_test, baseline_scores=None):
        """Evaluate classifier and print metrics."""
        predictions = self.predict(X_test, baseline_scores)
        proba = self.predict_proba(X_test, baseline_scores)

        # Classification report
        present_labels = sorted(set(y_test) | set(predictions))
        target_names = [
            self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            for i in present_labels
        ]

        print("\n" + "=" * 70)
        print("ANOMALY CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(
            y_test, predictions,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        ))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions, labels=present_labels)
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, index=target_names, columns=target_names))

        # Per-class metrics
        results = {
            "predictions": predictions,
            "probabilities": proba,
            "classification_report": classification_report(
                y_test, predictions,
                labels=present_labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": cm,
        }

        return results

    def _augment_features(self, X, baseline_scores=None):
        """Augment feature matrix with baseline model scores."""
        # Give XGBoost the baseline models' opinions (IF, SVM, stats) as extra input columns
        if baseline_scores is not None:
            extra_features = []
            for key in ["statistical_score", "isolation_forest_score",
                        "one_class_svm_score", "baseline_combined_score"]:
                if key in baseline_scores:
                    extra_features.append(baseline_scores[key].reshape(-1, 1))

            if extra_features:
                extras = np.hstack(extra_features)
                return np.hstack([X, extras])
        return X

    def _save(self):
        """Save classifier to disk."""
        joblib.dump(self.model, MODELS_DIR / "xgb_classifier.pkl")

    def load(self):
        """Load classifier from disk."""
        model_path = MODELS_DIR / "xgb_classifier.pkl"
        if model_path.exists():
            self.model = joblib.load(model_path)
            self.fitted = True
            print("[Classifier] Model loaded from disk.")
        else:
            print("[Classifier] No saved model found.")
