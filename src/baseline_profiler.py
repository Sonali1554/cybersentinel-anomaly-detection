"""
DELIVERABLE 2: Baseline Profiling Model
=========================================
Per-entity "normal" behaviour representation using:
1. Statistical Profiles — mean/std/percentiles for key behavioural features
2. Isolation Forest — unsupervised anomaly scoring
3. One-Class SVM — alternative baseline for model comparison

The profiler learns what "normal" looks like for each entity and scores
new events by deviation from the learned baseline.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASELINE_CONFIG, MODELS_DIR


class StatisticalProfiler:
    """
    Builds per-entity statistical profiles from normal behaviour data.
    Captures: mean, std, percentiles for login hour, session duration,
    geo-distance, resource diversity, and event frequency.
    """

    def __init__(self):
        self.profiles = {}

    def fit(self, df):
        """Build statistical profiles from training data."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Only learn from normal events
        normal_df = df[df["label"] == "normal"]

        for entity_id, group in normal_df.groupby("entity_id"):
            profile = {
                "entity_id": entity_id,
                "entity_type": group["entity_type"].iloc[0],
                "event_count": len(group),
                # Login hour stats
                "hour_mean": group["hour"].mean() if "hour" in group.columns else 12,
                "hour_std": max(group["hour"].std(), 0.1) if "hour" in group.columns else 4,
                # Session duration stats
                "duration_mean": group["session_duration"].mean(),
                "duration_std": max(group["session_duration"].std(), 1.0),
                "duration_p95": group["session_duration"].quantile(0.95),
                # Geo stats
                "geo_distance_mean": group["geo_distance_km"].mean() if "geo_distance_km" in group.columns else 0,
                "geo_distance_p95": group["geo_distance_km"].quantile(0.95) if "geo_distance_km" in group.columns else 100,
                # Resource diversity
                "unique_resources": group["resource_accessed"].nunique(),
                "typical_resources": group["resource_accessed"].value_counts().head(10).index.tolist(),
                # Auth patterns
                "primary_auth": group["auth_method"].mode().iloc[0] if len(group) > 0 else "password",
                "auth_success_rate": (group["auth_success"].astype(str).str.lower() == "true").mean() if "auth_success" in group.columns else 1.0,
            }
            self.profiles[entity_id] = profile

        print(f"[StatProfiler] Built profiles for {len(self.profiles)} entities")
        return self

    def score(self, df):
        """Score events against entity profiles. Higher = more anomalous."""
        df = df.copy()
        scores = np.zeros(len(df))

        for i, (idx, row) in enumerate(df.iterrows()):
            entity_id = row["entity_id"]
            if entity_id not in self.profiles:
                scores[i] = 0.5
                continue

            profile = self.profiles[entity_id]
            s = 0.0

            # Hour deviation
            if "hour" in row.index:
                hour_z = abs(row["hour"] - profile["hour_mean"]) / max(profile["hour_std"], 0.1)
                s += min(hour_z / 4.0, 1.0) * 0.2

            # Session duration deviation
            dur_z = abs(row["session_duration"] - profile["duration_mean"]) / max(profile["duration_std"], 1.0)
            s += min(dur_z / 4.0, 1.0) * 0.15

            # Geo distance anomaly
            if "geo_distance_km" in row.index and profile["geo_distance_p95"] > 0:
                geo_ratio = row["geo_distance_km"] / max(profile["geo_distance_p95"], 1.0)
                s += min(geo_ratio, 1.0) * 0.25

            # Unusual resource
            if row["resource_accessed"] not in profile["typical_resources"]:
                s += 0.2

            # Auth method mismatch
            if row["auth_method"] != profile["primary_auth"]:
                s += 0.1

            # Auth failure
            if "auth_success" in row.index and str(row["auth_success"]).lower() == "false":
                s += 0.1

            scores[i] = min(s, 1.0)

        return scores

    def get_profile(self, entity_id):
        """Retrieve profile for a specific entity."""
        return self.profiles.get(entity_id, None)


class BaselineProfiler:
    """
    Combined baseline profiling using Statistical Profiles,
    Isolation Forest, and One-Class SVM.
    """

    def __init__(self):
        # Combine 3 different "what does normal look like" techniques so no single blind spot dominates
        self.stat_profiler = StatisticalProfiler()
        self.isolation_forest = IsolationForest(**BASELINE_CONFIG["isolation_forest"])
        self.one_class_svm = OneClassSVM(**BASELINE_CONFIG["one_class_svm"])
        self.fitted = False

    def fit(self, X_train, df_train):
        """
        Fit all baseline models.
        X_train: scaled feature matrix (from FeatureEngineer)
        df_train: original DataFrame with labels and entity info
        """
        # 1. Statistical profiles (uses raw DataFrame)
        self.stat_profiler.fit(df_train)

        # 2. Isolation Forest (uses feature matrix, trained on normal data only)
        normal_mask = df_train["label"] == "normal"
        X_normal = X_train[normal_mask.values]

        print(f"[BaselineProfiler] Training Isolation Forest on {len(X_normal)} normal events...")
        self.isolation_forest.fit(X_normal)

        # 3. One-Class SVM (on subset for efficiency — SVM doesn't scale well)
        svm_sample_size = min(5000, len(X_normal))
        indices = np.random.choice(len(X_normal), svm_sample_size, replace=False)
        X_svm = X_normal[indices]

        print(f"[BaselineProfiler] Training One-Class SVM on {svm_sample_size} normal events...")
        self.one_class_svm.fit(X_svm)

        self.fitted = True
        self._save_models()
        print("[BaselineProfiler] All baseline models fitted and saved.")

    def score(self, X, df):
        """
        Score events using all baseline models.
        Returns dict with individual and combined scores.
        """
        assert self.fitted, "Must call fit() first"

        # Statistical profile scores (0–1, higher = more anomalous)
        stat_scores = self.stat_profiler.score(df)

        # Isolation Forest: decision_function returns negative for anomalies
        if_raw = self.isolation_forest.decision_function(X)
        # Normalize to 0–1 (invert so higher = more anomalous)
        if_min, if_max = if_raw.min(), if_raw.max()
        if if_max > if_min:
            if_scores = 1 - (if_raw - if_min) / (if_max - if_min)
        else:
            if_scores = np.full_like(if_raw, 0.5)

        # One-Class SVM: decision_function
        svm_raw = self.one_class_svm.decision_function(X)
        svm_min, svm_max = svm_raw.min(), svm_raw.max()
        if svm_max > svm_min:
            svm_scores = 1 - (svm_raw - svm_min) / (svm_max - svm_min)
        else:
            svm_scores = np.full_like(svm_raw, 0.5)

        # Combined baseline score (weighted average)
        combined = (0.3 * stat_scores + 0.4 * if_scores + 0.3 * svm_scores)

        return {
            "statistical_score": stat_scores,
            "isolation_forest_score": if_scores,
            "one_class_svm_score": svm_scores,
            "baseline_combined_score": combined,
        }

    def predict(self, X):
        """Binary anomaly prediction from Isolation Forest."""
        # Returns 1 for normal, -1 for anomaly
        return self.isolation_forest.predict(X)

    def _save_models(self):
        """Save trained models to disk."""
        joblib.dump(self.isolation_forest, MODELS_DIR / "isolation_forest.pkl")
        joblib.dump(self.one_class_svm, MODELS_DIR / "oneclasssvm.pkl")
        joblib.dump(self.stat_profiler, MODELS_DIR / "stat_profiler.pkl")

    def load_models(self):
        """Load trained models from disk."""
        self.isolation_forest = joblib.load(MODELS_DIR / "isolation_forest.pkl")
        self.one_class_svm = joblib.load(MODELS_DIR / "oneclasssvm.pkl")
        self.stat_profiler = joblib.load(MODELS_DIR / "stat_profiler.pkl")
        self.fitted = True
        print("[BaselineProfiler] Models loaded from disk.")
