"""
Feature Engineering Pipeline
=============================
Transforms raw access log events into ML-ready feature vectors.
Handles temporal, behavioral, and categorical features for all model layers.
"""

import json
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS_DIR, ANOMALY_TYPES


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometers."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


class FeatureEngineer:
    """Extracts and engineers features from raw access log data."""

    def __init__(self):
        # LabelEncoders turn text categories (auth method, resource, etc.) into numbers for ML
        self.entity_encoder = LabelEncoder()
        self.auth_encoder = LabelEncoder()
        self.resource_encoder = LabelEncoder()
        self.entity_type_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        # StandardScaler puts all features on the same scale so no single feature dominates
        self.scaler = StandardScaler()
        self.fitted = False

    def _parse_geo(self, geo_str):
        """Parse 'lat,lon' string to floats."""
        try:
            parts = str(geo_str).split(",")
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            return 0.0, 0.0

    def _parse_commands(self, cmd_str):
        """Parse command sequence JSON string."""
        try:
            cmds = json.loads(str(cmd_str))
            if isinstance(cmds, list):
                return cmds
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def _extract_temporal_features(self, df):
        """Extract time-based features from timestamp."""
        df = df.copy()
        ts = pd.to_datetime(df["timestamp"])
        # Break the timestamp into simple signals like hour, weekday, night/weekend flags
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        df["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype(int)
        df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
        df["day_of_month"] = ts.dt.day
        # Cyclical encoding for hour so 11pm and 12am are treated as close, not far apart
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        # Cyclical encoding for day of week (Sunday and Monday are next to each other too)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    def _extract_geo_features(self, df):
        """Extract geo-location features."""
        df = df.copy()
        geo_parsed = df["geo_location"].apply(self._parse_geo)
        df["latitude"] = geo_parsed.apply(lambda x: x[0])
        df["longitude"] = geo_parsed.apply(lambda x: x[1])
        return df

    def _extract_behavioral_features(self, df):
        """Extract per-entity behavioral deviation features."""
        df = df.copy()

        # Command sequence features
        df["cmd_count"] = df["command_sequence"].apply(lambda x: len(self._parse_commands(x)))
        # Flag sessions containing known attack-style commands (e.g. dump, exfiltrate, disable_logging)
        df["has_suspicious_cmd"] = df["command_sequence"].apply(
            lambda x: int(any(
                cmd in str(x).lower()
                for cmd in ["escalate", "dump", "exfiltrate", "disable_logging",
                            "scan_network", "override", "bulk_export", "delete_logs"]
            ))
        )

        # Device fingerprint features
        df["fingerprint_parts"] = df["device_fingerprint"].apply(
            lambda x: len(str(x).split("|")) if pd.notna(x) else 0
        )

        # Auth success as integer
        df["auth_success_int"] = df["auth_success"].astype(int)

        # Log-transform session duration so a few very long sessions don't skew the model
        df["log_session_duration"] = np.log1p(df["session_duration"].clip(lower=0))

        return df

    def _extract_entity_stats(self, df):
        """Compute rolling entity-level statistics as features."""
        df = df.copy()
        df = df.sort_values(["entity_id", "timestamp"])

        entity_groups = df.groupby("entity_id")

        # Per-entity: deviation from mean hour
        entity_mean_hour = entity_groups["hour"].transform("mean")
        entity_std_hour = entity_groups["hour"].transform("std").fillna(1.0)
        df["hour_deviation"] = (df["hour"] - entity_mean_hour).abs() / entity_std_hour.clip(lower=0.1)

        # Per-entity: deviation from mean session duration
        entity_mean_dur = entity_groups["session_duration"].transform("mean")
        entity_std_dur = entity_groups["session_duration"].transform("std").fillna(1.0)
        df["duration_deviation"] = (df["session_duration"] - entity_mean_dur).abs() / entity_std_dur.clip(lower=0.1)

        # Per-entity: inter-event time gap
        df["prev_timestamp"] = entity_groups["timestamp"].shift(1)
        df["time_gap_seconds"] = (
            pd.to_datetime(df["timestamp"]) - pd.to_datetime(df["prev_timestamp"])
        ).dt.total_seconds().fillna(0).clip(lower=0)
        df["log_time_gap"] = np.log1p(df["time_gap_seconds"])
        df.drop(columns=["prev_timestamp"], inplace=True)

        # Per-entity: geo distance from previous event
        df["prev_lat"] = entity_groups["latitude"].shift(1)
        df["prev_lon"] = entity_groups["longitude"].shift(1)
        df["geo_distance_km"] = df.apply(
            lambda row: haversine_km(
                row["prev_lat"], row["prev_lon"],
                row["latitude"], row["longitude"]
            ) if pd.notna(row["prev_lat"]) else 0.0,
            axis=1,
        )
        # Geo-velocity: km per hour
        df["geo_velocity_kmh"] = np.where(
            df["time_gap_seconds"] > 0,
            df["geo_distance_km"] / (df["time_gap_seconds"] / 3600),
            0.0,
        )
        df.drop(columns=["prev_lat", "prev_lon"], inplace=True)

        # Per-entity: unique resources accessed (cumulative count) — helps catch lateral movement /
        # insider drift, where an entity's "footprint" keeps growing over time
        def _cumulative_nunique(series):
            seen = set()
            counts = []
            for val in series:
                seen.add(val)
                counts.append(len(seen))
            return pd.Series(counts, index=series.index)

        df["cumulative_unique_resources"] = entity_groups["resource_accessed"].transform(
            _cumulative_nunique
        )

        # Per-entity: event count (rolling)
        df["entity_event_count"] = entity_groups.cumcount() + 1

        return df

    def fit_transform(self, df):
        """Fit encoders/scaler and transform the dataframe into features."""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract all feature groups
        df = self._extract_temporal_features(df)
        df = self._extract_geo_features(df)
        df = self._extract_behavioral_features(df)
        df = self._extract_entity_stats(df)

        # Encode categoricals
        df["entity_type_enc"] = self.entity_type_encoder.fit_transform(df["entity_type"])
        df["auth_method_enc"] = self.auth_encoder.fit_transform(df["auth_method"])
        df["resource_enc"] = self.resource_encoder.fit_transform(df["resource_accessed"])

        # Encode labels
        self.label_encoder.fit(ANOMALY_TYPES)
        df["label_enc"] = self.label_encoder.transform(df["label"])

        # Select numeric features for ML
        feature_cols = self._get_feature_columns()
        existing_cols = [c for c in feature_cols if c in df.columns]

        X = df[existing_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Fit and transform scaler
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        self.feature_columns = existing_cols

        # Save encoders and scaler
        joblib.dump(self.scaler, MODELS_DIR / "scaler.pkl")
        joblib.dump(self.label_encoder, MODELS_DIR / "label_encoder.pkl")

        print(f"[FeatureEng] Extracted {len(existing_cols)} features from {len(df)} events")
        return X_scaled, df["label_enc"].values, df

    def transform(self, df):
        """Transform new data using fitted encoders/scaler."""
        assert self.fitted, "Must call fit_transform first"
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = self._extract_temporal_features(df)
        df = self._extract_geo_features(df)
        df = self._extract_behavioral_features(df)
        df = self._extract_entity_stats(df)

        # Encode categoricals (handle unseen labels)
        df["entity_type_enc"] = self._safe_transform(self.entity_type_encoder, df["entity_type"])
        df["auth_method_enc"] = self._safe_transform(self.auth_encoder, df["auth_method"])
        df["resource_enc"] = self._safe_transform(self.resource_encoder, df["resource_accessed"])
        df["label_enc"] = self._safe_transform(self.label_encoder, df["label"])

        existing_cols = [c for c in self.feature_columns if c in df.columns]
        X = df[existing_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        return X_scaled, df["label_enc"].values, df

    def _safe_transform(self, encoder, series):
        """Transform with fallback for unseen labels."""
        result = []
        known = set(encoder.classes_)
        for val in series:
            if val in known:
                result.append(encoder.transform([val])[0])
            else:
                result.append(-1)
        return result

    def _get_feature_columns(self):
        """Return the list of numeric feature column names."""
        # This is the master feature list every downstream model (baseline, LSTM, classifier) is trained on
        return [
            "hour", "day_of_week", "is_weekend", "is_night", "minute_of_day",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "latitude", "longitude",
            "session_duration", "log_session_duration",
            "cmd_count", "has_suspicious_cmd",
            "fingerprint_parts", "auth_success_int",
            "hour_deviation", "duration_deviation",
            "time_gap_seconds", "log_time_gap",
            "geo_distance_km", "geo_velocity_kmh",
            "cumulative_unique_resources", "entity_event_count",
            "entity_type_enc", "auth_method_enc", "resource_enc",
        ]

    def get_feature_names(self):
        """Return feature names for explainability."""
        if hasattr(self, "feature_columns"):
            return self.feature_columns
        return self._get_feature_columns()
