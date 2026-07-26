"""
Cold-Start Handler
===================
Solves the cold-start problem: how to score a brand-new user or device
with no behavioural history.

Approach: Peer Group Profiling
1. When a new entity appears with < min_history_events, identify its peer group
   (entities of the same type with similar attributes).
2. Use the aggregate baseline profile of the peer group as a temporary reference.
3. Apply a lower anomaly threshold (elevated monitoring) for the first N days.
4. Gradually transition to the entity's own individual profile as history accumulates.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COLD_START_CONFIG


class ColdStartHandler:
    """
    Handles cold-start entities using peer group profiling.
    """

    def __init__(self, config=None):
        self.config = config or COLD_START_CONFIG
        self.peer_groups = {}   # entity_type -> aggregated profile
        self.entity_history_count = defaultdict(int)

    def build_peer_groups(self, stat_profiler):
        """
        Build aggregate peer group profiles from existing entity profiles.
        Groups entities by entity_type and computes aggregate statistics.
        """
        type_profiles = defaultdict(list)
        for entity_id, profile in stat_profiler.profiles.items():
            etype = profile["entity_type"]
            type_profiles[etype].append(profile)

        for etype, profiles in type_profiles.items():
            self.peer_groups[etype] = {
                "entity_type": etype,
                "count": len(profiles),
                "hour_mean": np.mean([p["hour_mean"] for p in profiles]),
                "hour_std": np.mean([p["hour_std"] for p in profiles]) * 1.5,  # Wider tolerance
                "duration_mean": np.mean([p["duration_mean"] for p in profiles]),
                "duration_std": np.mean([p["duration_std"] for p in profiles]) * 1.5,
                "duration_p95": np.percentile([p["duration_p95"] for p in profiles], 75),
                "geo_distance_mean": np.mean([p["geo_distance_mean"] for p in profiles]),
                "geo_distance_p95": np.percentile([p["geo_distance_p95"] for p in profiles], 90),
                "unique_resources_mean": np.mean([p["unique_resources"] for p in profiles]),
                "auth_success_rate": np.mean([p["auth_success_rate"] for p in profiles]),
            }

        print(f"[ColdStart] Built peer groups for {len(self.peer_groups)} entity types:")
        for etype, group in self.peer_groups.items():
            print(f"  {etype}: {group['count']} entities, avg hour={group['hour_mean']:.1f}")

    def is_cold_start(self, entity_id, event_count=None):
        """Check if an entity is in cold-start phase."""
        if event_count is not None:
            return event_count < self.config["min_history_events"]
        return self.entity_history_count[entity_id] < self.config["min_history_events"]

    def update_history(self, entity_id):
        """Increment event count for an entity."""
        self.entity_history_count[entity_id] += 1

    def get_peer_profile(self, entity_type):
        """Get the peer group profile for a given entity type."""
        return self.peer_groups.get(entity_type, None)

    def score_cold_start_event(self, event_data, entity_type):
        """
        Score a cold-start entity's event using peer group profile.
        Returns a risk score with elevated sensitivity.
        """
        peer = self.get_peer_profile(entity_type)
        if peer is None:
            return 0.5  # Default moderate risk for unknown types

        score = 0.0
        multiplier = self.config["threshold_multiplier"]

        # Hour deviation (compared to peer group)
        if "hour" in event_data:
            hour_z = abs(event_data["hour"] - peer["hour_mean"]) / max(peer["hour_std"], 0.1)
            score += min(hour_z / 3.0, 1.0) * 0.25  # Stricter threshold

        # Session duration
        if "session_duration" in event_data:
            dur_z = abs(event_data["session_duration"] - peer["duration_mean"]) / max(peer["duration_std"], 1.0)
            score += min(dur_z / 3.0, 1.0) * 0.2

        # Geo distance
        if "geo_distance_km" in event_data and peer["geo_distance_p95"] > 0:
            geo_ratio = event_data["geo_distance_km"] / max(peer["geo_distance_p95"], 1.0)
            score += min(geo_ratio * 1.2, 1.0) * 0.25  # Elevated geo sensitivity

        # Auth failure
        if "auth_success" in event_data and str(event_data["auth_success"]).lower() == "false":
            score += 0.2

        # Suspicious commands
        if "has_suspicious_cmd" in event_data and event_data["has_suspicious_cmd"]:
            score += 0.1

        # Apply elevated monitoring multiplier (lower threshold = more sensitive)
        adjusted_score = min(score / multiplier, 1.0)

        return adjusted_score

    def get_transition_weight(self, entity_id, event_count=None):
        """
        Get blending weight between peer profile and individual profile.
        Returns (peer_weight, individual_weight) that sum to 1.0.
        Weight transitions linearly from peer to individual over min_history_events.
        """
        count = event_count or self.entity_history_count[entity_id]
        min_events = self.config["min_history_events"]

        if count >= min_events:
            return 0.0, 1.0  # Fully individual
        else:
            individual_weight = count / min_events
            peer_weight = 1.0 - individual_weight
            return peer_weight, individual_weight

    def get_cold_start_info(self, entity_id, entity_type, event_count=None):
        """Get a summary of cold-start status for dashboard display."""
        count = event_count or self.entity_history_count[entity_id]
        is_cold = self.is_cold_start(entity_id, count)
        peer_w, ind_w = self.get_transition_weight(entity_id, count)

        return {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "is_cold_start": is_cold,
            "event_count": count,
            "min_required": self.config["min_history_events"],
            # How far along this entity is toward having enough history for its own profile
            "progress_pct": min(100, int(count / self.config["min_history_events"] * 100)),
            "peer_weight": round(peer_w, 2),
            "individual_weight": round(ind_w, 2),
            "monitoring_level": "Elevated" if is_cold else "Standard",
            "peer_group": entity_type,
        }
