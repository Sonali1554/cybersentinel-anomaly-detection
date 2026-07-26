"""
DELIVERABLE 1: Synthetic Data Generator
========================================
Generates realistic access-log data with per-entity behavioural profiles
and injected attack patterns at controlled rates (0.5–3% of sessions).

Behavioural Assumptions:
- Each entity has a habitual pattern: preferred login hours (±2h noise),
  consistent geo-location (1-2 base cities), typical resource set (3-6 resources),
  consistent device fingerprint, and preferred auth method.
- Normal session durations follow log-normal distribution per entity.
- Command sequences are drawn from entity-type-specific pools.
- Noise is added to all features to simulate real-world variance.

Attack Taxonomy:
- Brute force: rapid failed-auth bursts from single source
- Impossible travel: distant geo logins within implausible timeframe
- Credential stuffing: few IPs targeting many entities with high failure
- Lateral movement: sudden access to never-before-seen resources
- Device spoofing: fingerprint mismatch on known device
- Low-and-slow exfiltration: gradual off-hours access escalation
- Insider drift: slow privilege/resource expansion (edge case)
"""

import json
import random
import string
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
from faker import Faker

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_CONFIG, ATTACK_RATES, RESOURCES, AUTH_METHODS,
    NORMAL_COMMANDS, SUSPICIOUS_COMMANDS, GEO_LOCATIONS,
    OS_OPTIONS, PROTOCOLS, DATA_DIR
)

# Fix all random seeds so the same "random" data is generated every run (reproducibility)
fake = Faker()
Faker.seed(DATA_CONFIG["random_seed"])
np.random.seed(DATA_CONFIG["random_seed"])
random.seed(DATA_CONFIG["random_seed"])


class EntityProfile:
    """Represents a single entity's normal behavioural baseline."""

    def __init__(self, entity_id, entity_type):
        self.entity_id = entity_id
        self.entity_type = entity_type

        # Habitual login hours (24h clock, ±2h noise)
        if entity_type == "user":
            self.preferred_hour = np.random.choice(range(7, 20))  # 7 AM – 7 PM
            self.hour_std = np.random.uniform(1.0, 2.5)
        elif entity_type == "service_account":
            self.preferred_hour = np.random.choice(range(0, 24))  # 24/7
            self.hour_std = np.random.uniform(0.5, 1.5)
        else:  # edge_device
            self.preferred_hour = np.random.choice(range(0, 24))
            self.hour_std = np.random.uniform(1.0, 3.0)

        # Base geo-locations (1-2 consistent locations)
        num_locations = np.random.choice([1, 2], p=[0.7, 0.3])
        self.base_locations = random.sample(GEO_LOCATIONS, num_locations)

        # Typical resources (3-6 from entity-type pool)
        resource_pool = RESOURCES[entity_type]
        num_resources = min(np.random.randint(3, 7), len(resource_pool))
        self.typical_resources = random.sample(resource_pool, num_resources)

        # Preferred auth method
        auth_weights = AUTH_METHODS[entity_type]
        methods = list(auth_weights.keys())
        weights = list(auth_weights.values())
        self.preferred_auth = np.random.choice(methods, p=weights)

        # Session duration distribution (log-normal)
        if entity_type == "user":
            self.session_mean = np.random.uniform(300, 3600)  # 5 min – 1 hour
        elif entity_type == "service_account":
            self.session_mean = np.random.uniform(5, 120)  # 5 sec – 2 min
        else:
            self.session_mean = np.random.uniform(60, 600)  # 1 min – 10 min
        self.session_std = self.session_mean * 0.3

        # Device fingerprint
        self.os = random.choice(OS_OPTIONS[entity_type])
        self.mac = fake.mac_address()
        self.protocol = random.choice(PROTOCOLS[entity_type])

        # Source IP (consistent base)
        self.base_ip = fake.ipv4_public()

        # Events per day (activity level)
        if entity_type == "user":
            self.events_per_day = np.random.uniform(3, 15)
        elif entity_type == "service_account":
            self.events_per_day = np.random.uniform(20, 100)
        else:
            self.events_per_day = np.random.uniform(10, 50)

    def to_dict(self):
        """Serialize profile for JSON export."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "preferred_hour": int(self.preferred_hour),
            "hour_std": round(self.hour_std, 2),
            "base_locations": [loc["city"] for loc in self.base_locations],
            "typical_resources": self.typical_resources,
            "preferred_auth": self.preferred_auth,
            "session_mean": round(self.session_mean, 1),
            "os": self.os,
            "mac": self.mac,
            "protocol": self.protocol,
            "events_per_day": round(self.events_per_day, 1),
        }


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points in km."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


class SyntheticDataGenerator:
    """
    Generates synthetic access logs with realistic behavioural patterns
    and injected attack scenarios.
    """

    def __init__(self, config=None):
        self.config = config or DATA_CONFIG
        self.profiles = {}
        self.events = []
        self.start_date = datetime(2026, 4, 1)  # 90-day window start
        self.end_date = self.start_date + timedelta(days=self.config["time_window_days"])

    def create_entity_profiles(self):
        """Create behavioural profiles for all entities."""
        # Build a list of entity types matching the configured mix (e.g. 60% users, 25% service accounts)
        entity_types = []
        weights = self.config["entity_type_weights"]
        for etype, weight in weights.items():
            count = int(self.config["num_entities"] * weight)
            entity_types.extend([etype] * count)

        # Fill remaining to reach exact count
        while len(entity_types) < self.config["num_entities"]:
            entity_types.append("user")

        random.shuffle(entity_types)

        # Give each entity a readable ID like USR_0001, SVC_0002, DEV_0003
        prefixes = {"user": "USR", "service_account": "SVC", "edge_device": "DEV"}
        for i, etype in enumerate(entity_types):
            eid = f"{prefixes[etype]}_{i:04d}"
            self.profiles[eid] = EntityProfile(eid, etype)

        print(f"[DataGen] Created {len(self.profiles)} entity profiles")
        print(f"  - Users: {sum(1 for p in self.profiles.values() if p.entity_type == 'user')}")
        print(f"  - Service Accounts: {sum(1 for p in self.profiles.values() if p.entity_type == 'service_account')}")
        print(f"  - Edge Devices: {sum(1 for p in self.profiles.values() if p.entity_type == 'edge_device')}")

    def _generate_normal_event(self, profile, timestamp=None):
        """Generate a single normal access event for an entity."""
        if timestamp is None:
            # Random timestamp within window, biased toward entity's preferred hours
            day_offset = np.random.randint(0, self.config["time_window_days"])
            hour = int(np.clip(np.random.normal(profile.preferred_hour, profile.hour_std), 0, 23))
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            timestamp = self.start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        # Geo-location: pick from base locations with small noise
        loc = random.choice(profile.base_locations)
        lat_noise = np.random.normal(0, 0.05)
        lon_noise = np.random.normal(0, 0.05)

        # Resource: from typical set
        resource = random.choice(profile.typical_resources)

        # Auth method: preferred with 85% probability, else random
        if random.random() < 0.85:
            auth = profile.preferred_auth
        else:
            auth_opts = list(AUTH_METHODS[profile.entity_type].keys())
            auth = random.choice(auth_opts)

        # Session duration: log-normal
        duration = max(1, int(np.random.lognormal(
            np.log(profile.session_mean), 0.3
        )))

        # Command sequence: from normal pool
        cmd_pool = NORMAL_COMMANDS[profile.entity_type]
        commands = random.choice(cmd_pool)

        # Device fingerprint: consistent
        fingerprint = f"{profile.os}|{profile.mac}|{profile.protocol}"

        # Source IP: base with small variation
        ip = profile.base_ip

        return {
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": ip,
            "geo_location": f"{loc['lat'] + lat_noise:.4f},{loc['lon'] + lon_noise:.4f}",
            "geo_city": loc["city"],
            "resource_accessed": resource,
            "auth_method": auth,
            "session_duration": duration,
            "command_sequence": json.dumps(commands),
            "device_fingerprint": fingerprint,
            "auth_success": True,
            "label": "normal",
        }

    def _inject_brute_force(self, profile):
        """Inject brute-force attack: rapid failed-auth attempts."""
        events = []
        day_offset = np.random.randint(0, self.config["time_window_days"])
        base_time = self.start_date + timedelta(
            days=day_offset,
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )
        num_attempts = np.random.randint(10, 51)  # 10–50 rapid attempts
        attacker_ip = fake.ipv4_public()

        for i in range(num_attempts):
            ts = base_time + timedelta(seconds=np.random.randint(1, 300))  # Within 5 min
            loc = random.choice(GEO_LOCATIONS)
            events.append({
                "entity_id": profile.entity_id,
                "entity_type": profile.entity_type,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "source_ip": attacker_ip,
                "geo_location": f"{loc['lat']:.4f},{loc['lon']:.4f}",
                "geo_city": loc["city"],
                "resource_accessed": "/auth/login",
                "auth_method": "password",
                "session_duration": np.random.randint(1, 5),
                "command_sequence": json.dumps(["login_attempt_failed"]),
                "device_fingerprint": f"Unknown|{fake.mac_address()}|HTTPS",
                "auth_success": False,
                "label": "brute_force",
            })
        return events

    def _inject_impossible_travel(self, profile):
        """Inject impossible travel: logins from distant locations in short time."""
        events = []
        day_offset = np.random.randint(0, self.config["time_window_days"])
        base_time = self.start_date + timedelta(
            days=day_offset,
            hours=np.random.randint(8, 20),
        )

        # First login: base location
        loc1 = profile.base_locations[0]
        # Second login: distant location (>500km away)
        distant_locs = [
            g for g in GEO_LOCATIONS
            if haversine_km(loc1["lat"], loc1["lon"], g["lat"], g["lon"]) > 500
        ]
        loc2 = random.choice(distant_locs) if distant_locs else random.choice(GEO_LOCATIONS)

        # First event
        event1 = self._generate_normal_event(profile, base_time)
        event1["label"] = "impossible_travel"
        event1["geo_location"] = f"{loc1['lat']:.4f},{loc1['lon']:.4f}"
        event1["geo_city"] = loc1["city"]
        events.append(event1)

        # Second event: 10–45 minutes later from distant location
        gap_minutes = np.random.randint(10, 46)
        ts2 = base_time + timedelta(minutes=gap_minutes)
        event2 = self._generate_normal_event(profile, ts2)
        event2["label"] = "impossible_travel"
        event2["source_ip"] = fake.ipv4_public()
        event2["geo_location"] = f"{loc2['lat']:.4f},{loc2['lon']:.4f}"
        event2["geo_city"] = loc2["city"]
        events.append(event2)

        return events

    def _inject_credential_stuffing(self):
        """Inject credential stuffing: few IPs targeting many entities."""
        events = []
        # 1–3 attacker IPs
        num_ips = np.random.randint(1, 4)
        attacker_ips = [fake.ipv4_public() for _ in range(num_ips)]

        # Target 20–40 random entities
        target_entities = random.sample(
            list(self.profiles.values()),
            min(np.random.randint(20, 41), len(self.profiles))
        )

        day_offset = np.random.randint(0, self.config["time_window_days"])
        base_time = self.start_date + timedelta(days=day_offset, hours=np.random.randint(1, 6))
        loc = random.choice(GEO_LOCATIONS)

        for entity in target_entities:
            ts = base_time + timedelta(seconds=np.random.randint(0, 1800))
            success = random.random() < 0.15  # 85% failure rate
            events.append({
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "source_ip": random.choice(attacker_ips),
                "geo_location": f"{loc['lat']:.4f},{loc['lon']:.4f}",
                "geo_city": loc["city"],
                "resource_accessed": "/auth/login",
                "auth_method": "password",
                "session_duration": np.random.randint(1, 10),
                "command_sequence": json.dumps(
                    ["login_attempt_failed"] if not success
                    else ["login", "enumerate_accounts"]
                ),
                "device_fingerprint": f"Linux|{fake.mac_address()}|HTTPS",
                "auth_success": success,
                "label": "credential_stuffing",
            })
        return events

    def _inject_lateral_movement(self, profile):
        """Inject lateral movement: accessing unusual resources."""
        events = []
        all_resources = []
        for rtype in RESOURCES.values():
            all_resources.extend(rtype)
        unusual = [r for r in all_resources if r not in profile.typical_resources]

        day_offset = np.random.randint(0, self.config["time_window_days"])
        base_time = self.start_date + timedelta(
            days=day_offset,
            hours=np.random.randint(0, 24),
        )

        num_accesses = np.random.randint(5, 11)  # 5–10 unusual resources
        accessed = random.sample(unusual, min(num_accesses, len(unusual)))

        for i, resource in enumerate(accessed):
            ts = base_time + timedelta(minutes=i * np.random.randint(2, 15))
            event = self._generate_normal_event(profile, ts)
            event["resource_accessed"] = resource
            event["command_sequence"] = json.dumps(random.choice(SUSPICIOUS_COMMANDS))
            event["label"] = "lateral_movement"
            events.append(event)

        return events

    def _inject_device_spoofing(self, profile):
        """Inject device spoofing: mismatched fingerprint."""
        events = []
        day_offset = np.random.randint(0, self.config["time_window_days"])
        ts = self.start_date + timedelta(
            days=day_offset,
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
        )

        # Different OS and MAC than historical
        spoofed_os_pool = []
        for os_list in OS_OPTIONS.values():
            spoofed_os_pool.extend(os_list)
        spoofed_os = random.choice([o for o in spoofed_os_pool if o != profile.os])
        spoofed_mac = fake.mac_address()

        event = self._generate_normal_event(profile, ts)
        event["device_fingerprint"] = f"{spoofed_os}|{spoofed_mac}|{profile.protocol}"
        event["label"] = "device_spoofing"
        events.append(event)

        return events

    def _inject_low_and_slow(self, profile):
        """Inject low-and-slow exfiltration: gradual off-hours access."""
        events = []
        start_day = np.random.randint(0, max(1, self.config["time_window_days"] - 21))

        # Gradually increasing access over 2–3 weeks
        for week in range(3):
            accesses_this_week = 2 + week * 2  # 2, 4, 6
            for _ in range(accesses_this_week):
                day = start_day + week * 7 + np.random.randint(0, 7)
                if day >= self.config["time_window_days"]:
                    continue
                # Off-hours: 2–5 AM
                hour = np.random.randint(2, 6)
                ts = self.start_date + timedelta(days=day, hours=hour, minutes=np.random.randint(0, 60))

                event = self._generate_normal_event(profile, ts)
                # Sensitive resources
                sensitive = ["/files/docs/contracts", "/api/v1/data/export",
                             "/files/reports/quarterly.xlsx", "/api/v1/db/query"]
                event["resource_accessed"] = random.choice(sensitive)
                event["session_duration"] = np.random.randint(60, 300)  # Short sessions
                event["label"] = "low_and_slow"
                events.append(event)

        return events

    def _inject_insider_drift(self, profile):
        """Inject insider drift: gradually expanding resource access (edge case)."""
        events = []
        all_resources = RESOURCES[profile.entity_type]
        new_resources = [r for r in all_resources if r not in profile.typical_resources]

        if not new_resources:
            return events

        start_day = np.random.randint(0, max(1, self.config["time_window_days"] - 30))

        # Add 1–2 new resources per week over 4 weeks
        for week in range(4):
            if not new_resources:
                break
            new_this_week = random.sample(new_resources, min(np.random.randint(1, 3), len(new_resources)))
            for res in new_this_week:
                new_resources.remove(res)
                day = start_day + week * 7 + np.random.randint(0, 7)
                if day >= self.config["time_window_days"]:
                    continue
                ts = self.start_date + timedelta(
                    days=day,
                    hours=int(np.clip(np.random.normal(profile.preferred_hour, profile.hour_std), 0, 23)),
                    minutes=np.random.randint(0, 60),
                )
                event = self._generate_normal_event(profile, ts)
                event["resource_accessed"] = res
                event["label"] = "insider_drift"
                events.append(event)

        return events

    def generate(self):
        """Generate the complete synthetic dataset."""
        print("[DataGen] Starting synthetic data generation...")
        self.create_entity_profiles()

        # ── Step 1: Generate normal events ───────────────────────────────
        print("[DataGen] Generating normal baseline events...")
        total_normal = int(self.config["total_events"] * (1 - sum(ATTACK_RATES.values())))

        for profile in self.profiles.values():
            # Proportional events based on entity activity level
            total_entity_activity = sum(p.events_per_day for p in self.profiles.values())
            entity_share = profile.events_per_day / total_entity_activity
            num_events = max(5, int(total_normal * entity_share))

            for _ in range(num_events):
                self.events.append(self._generate_normal_event(profile))

        print(f"  Normal events: {len(self.events)}")

        # ── Step 2: Inject attack patterns ───────────────────────────────
        print("[DataGen] Injecting attack patterns...")
        entity_list = list(self.profiles.values())

        # Brute force
        n_brute = max(1, int(self.config["total_events"] * ATTACK_RATES["brute_force"] / 20))
        for _ in range(n_brute):
            target = random.choice(entity_list)
            self.events.extend(self._inject_brute_force(target))

        # Impossible travel
        n_travel = max(1, int(self.config["total_events"] * ATTACK_RATES["impossible_travel"] / 2))
        for _ in range(n_travel):
            target = random.choice([e for e in entity_list if e.entity_type == "user"])
            self.events.extend(self._inject_impossible_travel(target))

        # Credential stuffing (bulk events)
        n_stuffing = max(1, int(self.config["total_events"] * ATTACK_RATES["credential_stuffing"] / 25))
        for _ in range(n_stuffing):
            self.events.extend(self._inject_credential_stuffing())

        # Lateral movement
        n_lateral = max(1, int(self.config["total_events"] * ATTACK_RATES["lateral_movement"] / 7))
        for _ in range(n_lateral):
            target = random.choice(entity_list)
            self.events.extend(self._inject_lateral_movement(target))

        # Device spoofing
        n_spoof = max(1, int(self.config["total_events"] * ATTACK_RATES["device_spoofing"]))
        device_entities = [e for e in entity_list if e.entity_type == "edge_device"]
        if not device_entities:
            device_entities = entity_list
        for _ in range(n_spoof):
            target = random.choice(device_entities)
            self.events.extend(self._inject_device_spoofing(target))

        # Low-and-slow exfiltration
        n_slow = max(1, int(self.config["total_events"] * ATTACK_RATES["low_and_slow"] / 10))
        for _ in range(n_slow):
            target = random.choice(entity_list)
            self.events.extend(self._inject_low_and_slow(target))

        # Insider drift
        n_drift = max(1, int(self.config["total_events"] * ATTACK_RATES["insider_drift"] / 5))
        for _ in range(n_drift):
            target = random.choice([e for e in entity_list if e.entity_type == "user"])
            self.events.extend(self._inject_insider_drift(target))

        # ── Step 3: Create DataFrame and sort by timestamp ───────────────
        df = pd.DataFrame(self.events)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Print summary
        total = len(df)
        anomaly_count = len(df[df["label"] != "normal"])
        print(f"\n[DataGen] Generation complete!")
        print(f"  Total events: {total}")
        print(f"  Normal events: {total - anomaly_count} ({(total - anomaly_count)/total*100:.1f}%)")
        print(f"  Anomaly events: {anomaly_count} ({anomaly_count/total*100:.1f}%)")
        print(f"\n  Attack breakdown:")
        for label in df["label"].unique():
            if label != "normal":
                count = len(df[df["label"] == label])
                print(f"    {label}: {count} ({count/total*100:.2f}%)")

        return df

    def save(self, df, split_ratio=0.8):
        """Save dataset and profiles to disk."""
        # Save full dataset
        df.to_csv(DATA_DIR / "access_logs.csv", index=False)

        # Train/test split (temporal: earlier events for training)
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        train_df.to_csv(DATA_DIR / "train.csv", index=False)
        test_df.to_csv(DATA_DIR / "test.csv", index=False)

        # Save entity profiles
        profiles_dict = {eid: p.to_dict() for eid, p in self.profiles.items()}
        with open(DATA_DIR / "entity_profiles.json", "w") as f:
            json.dump(profiles_dict, f, indent=2)

        print(f"\n[DataGen] Saved to {DATA_DIR}/")
        print(f"  access_logs.csv: {len(df)} events")
        print(f"  train.csv: {len(train_df)} events")
        print(f"  test.csv: {len(test_df)} events")
        print(f"  entity_profiles.json: {len(profiles_dict)} profiles")

        return train_df, test_df


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    df = generator.generate()
    train_df, test_df = generator.save(df)
    print("\nSample events:")
    print(df.head(3).to_string())
