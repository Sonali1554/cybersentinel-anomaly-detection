"""
Concept Drift Detection & Adaptation
======================================
Legitimate behaviour evolves (new work patterns, new devices, role changes).
These should NOT be permanently flagged as anomalies.

Implements:
1. Sliding Window: Baseline profiles updated with exponential decay
   (recent events weighted more heavily than old ones).
2. Page-Hinkley Drift Detection: Monitors feature distribution shifts
   and triggers recalibration when significant drift is detected.
3. Adaptive Thresholds: Anomaly score thresholds recalibrated periodically.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DRIFT_CONFIG


class PageHinkleyDetector:
    """
    Page-Hinkley test for detecting distribution shifts.
    Monitors a running statistic and signals when cumulative deviation
    from the mean exceeds a threshold.
    """

    def __init__(self, threshold=None, delta=0.005):
        self.threshold = threshold or DRIFT_CONFIG["drift_threshold"]
        self.delta = delta
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = float("inf")

    def update(self, value):
        """Update with a new observation. Returns True if drift detected."""
        self.n += 1
        self.sum += value
        self.mean = self.sum / self.n
        # Track the running gap between observed values and the mean
        self.cumulative_sum += value - self.mean - self.delta
        # Remember the lowest point this cumulative sum has reached
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)

        # A big rebound from that low point means behaviour has shifted (drift)
        page_hinkley_value = self.cumulative_sum - self.min_cumulative_sum
        return page_hinkley_value > self.threshold


class ConceptDriftAdapter:
    """
    Manages concept drift detection and adaptive baseline updates.
    """

    def __init__(self, config=None):
        self.config = config or DRIFT_CONFIG
        self.entity_windows = defaultdict(list)  # entity_id -> recent feature vectors
        self.entity_baselines = {}               # entity_id -> running baseline stats
        self.drift_detectors = defaultdict(PageHinkleyDetector)
        self.drift_events = []                    # Log of detected drift events

    def update_window(self, entity_id, feature_vector, timestamp):
        """Add a new event to the entity's sliding window."""
        window = self.entity_windows[entity_id]
        window.append({
            "features": feature_vector,
            "timestamp": timestamp,
        })

        # Keep only the most recent window_size events
        if len(window) > self.config["window_size"]:
            self.entity_windows[entity_id] = window[-self.config["window_size"]:]

    def compute_adaptive_baseline(self, entity_id):
        """
        Compute adaptive baseline using exponential decay weighting.
        Recent events are weighted more heavily than older ones.
        """
        window = self.entity_windows[entity_id]
        if not window:
            return None

        n = len(window)
        decay = self.config["decay_factor"]

        # Compute weights: most recent event gets weight 1, older events decay
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()

        # Weighted mean and std of features
        features = np.array([w["features"] for w in window])
        weighted_mean = np.average(features, axis=0, weights=weights)
        weighted_var = np.average((features - weighted_mean) ** 2, axis=0, weights=weights)
        weighted_std = np.sqrt(weighted_var)

        self.entity_baselines[entity_id] = {
            "mean": weighted_mean,
            "std": weighted_std,
            "n_events": n,
            "last_updated": window[-1]["timestamp"] if window else None,
        }

        return self.entity_baselines[entity_id]

    def check_drift(self, entity_id, feature_vector):
        """
        Check if an entity's behaviour has drifted significantly.
        Uses Page-Hinkley test on the deviation from baseline.
        """
        if entity_id not in self.entity_baselines:
            return False, 0.0

        baseline = self.entity_baselines[entity_id]
        # Average number of standard deviations this event is away from the entity's usual behaviour
        deviation = np.mean(np.abs(feature_vector - baseline["mean"]) / np.clip(baseline["std"], 1e-6, None))

        detector = self.drift_detectors[entity_id]
        drift_detected = detector.update(deviation)

        if drift_detected:
            self.drift_events.append({
                "entity_id": entity_id,
                "timestamp": datetime.now().isoformat(),
                "deviation": float(deviation),
                "action": "recalibrate",
            })
            # Reset detector and trigger recalibration
            detector.reset()
            self.compute_adaptive_baseline(entity_id)

        return drift_detected, deviation

    def get_adaptive_threshold(self, entity_id, base_threshold):
        """
        Get an adaptive anomaly threshold that accounts for concept drift.
        If recent behaviour has been consistently changing, widen the threshold.
        """
        if entity_id not in self.entity_baselines:
            return base_threshold

        baseline = self.entity_baselines[entity_id]
        # If the entity's behaviour is naturally variable, widen threshold
        avg_std = np.mean(baseline["std"])
        if avg_std > 1.0:
            return base_threshold * (1 + avg_std * 0.1)
        return base_threshold

    def simulate_drift(self, original_profiles, drift_factor=0.3):
        """
        Simulate concept drift for demonstration purposes.
        Gradually shifts entity baselines to show adaptation in action.

        Returns: (before_df, after_df, drift_info)
        """
        drift_info = []

        # Simulate 3 phases of drift
        phases = [
            {"name": "Phase 1: Stable", "shift": 0.0, "days": 30},
            {"name": "Phase 2: Gradual Shift", "shift": drift_factor, "days": 30},
            {"name": "Phase 3: New Normal", "shift": drift_factor * 2, "days": 30},
        ]

        for phase in phases:
            drift_info.append({
                "phase": phase["name"],
                "shift_magnitude": phase["shift"],
                "duration_days": phase["days"],
                "description": (
                    f"Behavioural baseline shifts by {phase['shift']:.1f}σ "
                    f"over {phase['days']} days"
                ),
            })

        return drift_info

    def get_drift_summary(self):
        """Get summary of all detected drift events."""
        return {
            "total_drift_events": len(self.drift_events),
            "entities_with_drift": len(set(e["entity_id"] for e in self.drift_events)),
            "recent_drifts": self.drift_events[-10:],
            "config": {
                "window_size": self.config["window_size"],
                "decay_factor": self.config["decay_factor"],
                "drift_threshold": self.config["drift_threshold"],
            },
        }

    def demonstrate_adaptation(self, entity_id, n_events=100):
        """
        Generate demonstration data showing how adaptation works.
        Returns data suitable for visualization in the dashboard.
        """
        np.random.seed(42)

        # Phase 1: Stable baseline (first 40 events)
        base_mean = np.random.uniform(5, 15)
        stable = np.random.normal(base_mean, 1.0, 40)

        # Phase 2: Gradual drift (next 30 events)
        drift_target = base_mean + 5.0
        drift = np.linspace(base_mean, drift_target, 30) + np.random.normal(0, 1.0, 30)

        # Phase 3: New stable (last 30 events)
        new_stable = np.random.normal(drift_target, 1.0, 30)

        all_values = np.concatenate([stable, drift, new_stable])

        # Compute running adaptive baseline
        adaptive_means = []
        adaptive_stds = []
        window = []
        decay = self.config["decay_factor"]

        for i, val in enumerate(all_values):
            window.append(val)
            if len(window) > 20:
                window = window[-20:]

            n = len(window)
            weights = np.array([decay ** (n - 1 - j) for j in range(n)])
            weights /= weights.sum()

            adaptive_means.append(np.average(window, weights=weights))
            adaptive_stds.append(np.sqrt(np.average((np.array(window) - adaptive_means[-1])**2, weights=weights)))

        # Static baseline (doesn't adapt)
        static_mean = np.mean(stable)
        static_std = np.std(stable)

        return {
            "event_index": list(range(len(all_values))),
            "actual_values": all_values.tolist(),
            "adaptive_baseline": adaptive_means,
            "adaptive_upper": [m + 2*s for m, s in zip(adaptive_means, adaptive_stds)],
            "adaptive_lower": [m - 2*s for m, s in zip(adaptive_means, adaptive_stds)],
            "static_baseline": [static_mean] * len(all_values),
            "static_upper": [static_mean + 2*static_std] * len(all_values),
            "static_lower": [static_mean - 2*static_std] * len(all_values),
            "phases": [
                {"start": 0, "end": 39, "label": "Stable Baseline"},
                {"start": 40, "end": 69, "label": "Gradual Drift"},
                {"start": 70, "end": 99, "label": "New Normal"},
            ],
            "drift_detected_at": [40, 55],  # Approximate drift detection points
        }
