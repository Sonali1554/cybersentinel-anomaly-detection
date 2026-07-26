"""
DELIVERABLE 5: Explainability Layer
=====================================
Provides human-readable explanations for every anomaly alert.
A SOC analyst needs to know WHY an event was flagged, not just a score.

Implements:
- SHAP TreeExplainer for XGBoost classification decisions
- Feature attribution: top-N contributing factors per alert
- Natural language explanation generation
- Risk factor breakdown for dashboard display
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ANOMALY_DESCRIPTIONS


# Human-readable feature name mapping — turns technical column names into
# plain-English labels a SOC analyst can read on the dashboard
FEATURE_DISPLAY_NAMES = {
    "hour": "Login Hour",
    "day_of_week": "Day of Week",
    "is_weekend": "Weekend Access",
    "is_night": "Nighttime Access",
    "minute_of_day": "Time of Day",
    "hour_sin": "Hour (cyclic)",
    "hour_cos": "Hour (cyclic)",
    "dow_sin": "Day (cyclic)",
    "dow_cos": "Day (cyclic)",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "session_duration": "Session Duration",
    "log_session_duration": "Session Duration (log)",
    "cmd_count": "Command Count",
    "has_suspicious_cmd": "Suspicious Commands Detected",
    "fingerprint_parts": "Device Fingerprint Completeness",
    "auth_success_int": "Authentication Success",
    "hour_deviation": "Hour Deviation from Normal",
    "duration_deviation": "Duration Deviation from Normal",
    "time_gap_seconds": "Time Since Last Event",
    "log_time_gap": "Time Gap (log)",
    "geo_distance_km": "Geographic Distance from Last Login",
    "geo_velocity_kmh": "Geographic Velocity (km/h)",
    "cumulative_unique_resources": "Unique Resources Accessed",
    "entity_event_count": "Total Event Count",
    "entity_type_enc": "Entity Type",
    "auth_method_enc": "Auth Method",
    "resource_enc": "Resource Accessed",
    "statistical_score": "Statistical Anomaly Score",
    "isolation_forest_score": "Isolation Forest Score",
    "one_class_svm_score": "One-Class SVM Score",
    "baseline_combined_score": "Combined Baseline Score",
}


class ExplainabilityEngine:
    """
    Generates human-readable explanations for anomaly detections.
    Uses SHAP for feature attribution and rule-based natural language generation.
    """

    def __init__(self, classifier_model=None, feature_names=None):
        self.classifier = classifier_model
        self.feature_names = feature_names or []
        self.shap_explainer = None
        self.shap_values = None

    def initialize_shap(self, classifier_model, feature_names=None):
        """Initialize SHAP explainer for the classifier."""
        self.classifier = classifier_model
        if feature_names:
            self.feature_names = feature_names

        try:
            import shap
            self.shap_explainer = shap.TreeExplainer(classifier_model)
            print("[Explainability] SHAP TreeExplainer initialized.")
        except Exception as e:
            print(f"[Explainability] SHAP initialization failed: {e}")
            print("  Falling back to feature-importance-based explanations.")
            self.shap_explainer = None

    def explain_event(self, X_single, event_data=None, top_n=5):
        """
        Generate explanation for a single event.
        X_single: feature vector (1D or 2D array with 1 row)
        event_data: optional dict/Series with raw event data
        top_n: number of top contributing factors to return
        """
        X_single = np.atleast_2d(X_single)
        explanation = {
            "contributing_factors": [],
            "risk_summary": "",
            "anomaly_type_explanation": "",
        }

        # Get SHAP values if available
        if self.shap_explainer is not None:
            try:
                import shap
                shap_vals = self.shap_explainer.shap_values(X_single)
                # For multi-class, shap_vals is a list of arrays (one per class)
                if isinstance(shap_vals, list):
                    # Use the SHAP values for the predicted class
                    pred = self.classifier.predict(X_single)[0]
                    if pred < len(shap_vals):
                        event_shap = shap_vals[pred][0]
                    else:
                        event_shap = shap_vals[0][0]
                else:
                    event_shap = shap_vals[0]

                explanation["contributing_factors"] = self._format_shap_factors(
                    event_shap, X_single[0], top_n
                )
            except Exception as e:
                explanation["contributing_factors"] = self._fallback_explanation(
                    X_single[0], top_n
                )
        else:
            explanation["contributing_factors"] = self._fallback_explanation(
                X_single[0], top_n
            )

        # Add context from raw event data
        if event_data is not None:
            explanation["risk_summary"] = self._generate_risk_summary(event_data)
            if "label" in event_data:
                label = event_data["label"] if isinstance(event_data, dict) else event_data.get("label", "")
                if label in ANOMALY_DESCRIPTIONS:
                    explanation["anomaly_type_explanation"] = ANOMALY_DESCRIPTIONS[label]

        return explanation

    def explain_batch(self, X, event_df=None, top_n=3):
        """Generate explanations for a batch of events."""
        explanations = []
        for i in range(len(X)):
            event_data = event_df.iloc[i] if event_df is not None else None
            explanations.append(self.explain_event(X[i], event_data, top_n))
        return explanations

    def _format_shap_factors(self, shap_values, feature_values, top_n):
        """Format SHAP values into human-readable contributing factors."""
        factors = []
        # Get indices sorted by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]

        for idx in top_indices:
            if idx < len(self.feature_names):
                fname = self.feature_names[idx]
            else:
                fname = f"feature_{idx}"

            display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
            shap_val = shap_values[idx]
            feat_val = feature_values[idx]
            direction = "increases" if shap_val > 0 else "decreases"

            factors.append({
                "feature": fname,
                "display_name": display_name,
                "shap_value": float(shap_val),
                "feature_value": float(feat_val),
                "impact": direction,
                "importance": float(abs_shap[idx]),
                "description": self._feature_description(fname, feat_val, shap_val),
            })

        return factors

    def _fallback_explanation(self, feature_values, top_n):
        """Generate explanation using feature importance when SHAP unavailable."""
        factors = []
        if hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]

            for idx in top_indices:
                if idx < len(self.feature_names):
                    fname = self.feature_names[idx]
                else:
                    fname = f"feature_{idx}"

                display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
                feat_val = feature_values[idx] if idx < len(feature_values) else 0

                factors.append({
                    "feature": fname,
                    "display_name": display_name,
                    "importance": float(importances[idx]),
                    "feature_value": float(feat_val),
                    "description": self._feature_description(fname, feat_val, importances[idx]),
                })
        else:
            # Last resort: report top features by absolute value
            top_indices = np.argsort(np.abs(feature_values))[-top_n:][::-1]
            for idx in top_indices:
                fname = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
                factors.append({
                    "feature": fname,
                    "display_name": display_name,
                    "feature_value": float(feature_values[idx]),
                    "description": self._feature_description(fname, feature_values[idx], 0),
                })

        return factors

    def _feature_description(self, feature_name, value, shap_val):
        """Generate a natural language description for a feature contribution."""
        # Custom, plain-English phrasing for the features analysts care about most
        desc_map = {
            "geo_velocity_kmh": lambda v: f"Travel speed of {v:.0f} km/h detected (impossible travel indicator)" if v > 500 else f"Geographic velocity: {v:.0f} km/h",
            "geo_distance_km": lambda v: f"Login from {v:.0f} km away from previous location" if v > 100 else f"Geographic distance: {v:.0f} km",
            "hour_deviation": lambda v: f"Login hour deviates {v:.1f} standard deviations from normal pattern",
            "duration_deviation": lambda v: f"Session duration deviates {v:.1f} standard deviations from normal",
            "is_night": lambda v: "Nighttime access detected (unusual for this entity)" if v > 0 else "Daytime access (normal)",
            "is_weekend": lambda v: "Weekend access detected" if v > 0 else "Weekday access",
            "has_suspicious_cmd": lambda v: "Suspicious command patterns detected in session" if v > 0 else "Normal command patterns",
            "auth_success_int": lambda v: "Authentication FAILED" if v == 0 else "Authentication succeeded",
            "cumulative_unique_resources": lambda v: f"Entity has accessed {v:.0f} unique resources (resource breadth check)",
            "session_duration": lambda v: f"Session lasted {v:.0f} seconds",
            "time_gap_seconds": lambda v: f"Only {v:.0f} seconds since previous event" if v < 60 else f"{v/3600:.1f} hours since previous event",
            "cmd_count": lambda v: f"{v:.0f} commands executed in session",
        }

        if feature_name in desc_map:
            return desc_map[feature_name](value)

        display = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
        if shap_val > 0:
            return f"{display} value ({value:.2f}) contributed to anomaly detection"
        else:
            return f"{display}: {value:.2f}"

    def _generate_risk_summary(self, event_data):
        """Generate a one-line risk summary from raw event data."""
        parts = []

        if isinstance(event_data, pd.Series):
            event_dict = event_data.to_dict()
        else:
            event_dict = event_data

        entity_id = event_dict.get("entity_id", "Unknown")
        entity_type = event_dict.get("entity_type", "unknown")
        resource = event_dict.get("resource_accessed", "unknown resource")
        geo_city = event_dict.get("geo_city", "unknown location")
        auth = event_dict.get("auth_method", "unknown")
        auth_success = event_dict.get("auth_success", True)

        parts.append(f"Entity {entity_id} ({entity_type})")
        parts.append(f"accessed {resource}")
        parts.append(f"from {geo_city}")

        if not auth_success:
            parts.append(f"with FAILED {auth} authentication")
        else:
            parts.append(f"via {auth}")

        return " | ".join(parts)

    def get_global_feature_importance(self):
        """Get global feature importance from the classifier."""
        if self.classifier is not None and hasattr(self.classifier, "feature_importances_"):
            importances = self.classifier.feature_importances_
            feature_imp = []
            for i, imp in enumerate(importances):
                fname = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                feature_imp.append({
                    "feature": fname,
                    "display_name": FEATURE_DISPLAY_NAMES.get(fname, fname),
                    "importance": float(imp),
                })
            return sorted(feature_imp, key=lambda x: x["importance"], reverse=True)
        return []
