"""
CyberSentinel — Central Configuration
All paths, hyperparameters, constants, and schema definitions.
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ─── Data Generation Config ───────────────────────────────────────────────────
DATA_CONFIG = {
    "num_entities": 200,              # Total unique entities
    "entity_type_weights": {          # Distribution of entity types
        "user": 0.60,
        "service_account": 0.25,
        "edge_device": 0.15,
    },
    "total_events": 50000,            # Total access log events to generate
    "time_window_days": 90,           # 90-day observation period
    "random_seed": 42,
}

# ─── Attack Injection Rates ───────────────────────────────────────────────────
# Total anomaly rate: ~2.5% (within 0.5–3% spec)
ATTACK_RATES = {
    "brute_force": 0.004,             # 0.4%
    "impossible_travel": 0.003,       # 0.3%
    "credential_stuffing": 0.003,     # 0.3%
    "lateral_movement": 0.004,        # 0.4%
    "device_spoofing": 0.003,         # 0.3%
    "low_and_slow": 0.003,            # 0.3%
    "insider_drift": 0.005,           # 0.5% (edge case)
}

# ─── Synthetic Data Schema ────────────────────────────────────────────────────
SCHEMA_FIELDS = [
    "entity_id",
    "entity_type",
    "timestamp",
    "source_ip",
    "geo_location",       # stored as "lat,lon"
    "geo_city",           # human-readable city name
    "resource_accessed",
    "auth_method",
    "session_duration",
    "command_sequence",
    "device_fingerprint",
    "auth_success",       # boolean - whether auth succeeded
    "label",              # normal / anomaly_type
]

# ─── Resource Pools (by entity type) ─────────────────────────────────────────
RESOURCES = {
    "user": [
        "/files/reports/quarterly.xlsx", "/files/docs/policy.pdf",
        "/email/inbox", "/email/sent", "/apps/crm/dashboard",
        "/apps/erp/inventory", "/apps/hr/timesheet", "/apps/slack",
        "/files/shared/presentations", "/apps/jira/board",
        "/files/docs/contracts", "/apps/confluence/wiki",
    ],
    "service_account": [
        "/api/v1/auth/token", "/api/v1/data/export", "/api/v1/users/list",
        "/api/v1/logs/audit", "/api/v1/config/update", "/api/v1/db/query",
        "/api/v1/backup/trigger", "/api/v1/metrics/collect",
        "/api/v1/deploy/status", "/api/v1/certs/renew",
    ],
    "edge_device": [
        "/telemetry/temperature", "/telemetry/pressure",
        "/telemetry/vibration", "/control/valve/set",
        "/control/pump/start", "/control/pump/stop",
        "/firmware/check_update", "/config/network",
        "/diagnostics/self_test", "/logs/system",
    ],
}

# ─── Auth Methods (by entity type) ───────────────────────────────────────────
AUTH_METHODS = {
    "user": {"password": 0.50, "token": 0.30, "biometric": 0.15, "certificate": 0.05},
    "service_account": {"token": 0.60, "certificate": 0.35, "password": 0.05, "biometric": 0.0},
    "edge_device": {"certificate": 0.55, "token": 0.40, "password": 0.05, "biometric": 0.0},
}

# ─── Command Sequences (for privileged sessions) ─────────────────────────────
NORMAL_COMMANDS = {
    "user": [
        ["login", "read_file", "logout"],
        ["login", "search", "read_file", "download", "logout"],
        ["login", "edit_file", "save", "logout"],
        ["login", "view_dashboard", "export_report", "logout"],
    ],
    "service_account": [
        ["authenticate", "query_db", "return_results"],
        ["authenticate", "collect_metrics", "store_metrics"],
        ["authenticate", "run_backup", "verify_backup"],
        ["authenticate", "check_health", "report_status"],
    ],
    "edge_device": [
        ["handshake", "send_telemetry", "ack"],
        ["handshake", "receive_config", "apply_config", "ack"],
        ["handshake", "run_diagnostics", "send_results", "ack"],
    ],
}

SUSPICIOUS_COMMANDS = [
    ["login", "escalate_privileges", "dump_credentials", "exfiltrate_data"],
    ["login", "disable_logging", "access_admin_panel", "modify_config"],
    ["login", "scan_network", "enumerate_hosts", "lateral_move", "access_db"],
    ["authenticate", "override_permissions", "bulk_export", "delete_logs"],
]

# ─── Geo Locations (city, lat, lon, typical timezone offset) ──────────────────
GEO_LOCATIONS = [
    {"city": "New York", "lat": 40.7128, "lon": -74.0060, "tz_offset": -5},
    {"city": "London", "lat": 51.5074, "lon": -0.1278, "tz_offset": 0},
    {"city": "Tokyo", "lat": 35.6762, "lon": 139.6503, "tz_offset": 9},
    {"city": "San Francisco", "lat": 37.7749, "lon": -122.4194, "tz_offset": -8},
    {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777, "tz_offset": 5.5},
    {"city": "Berlin", "lat": 52.5200, "lon": 13.4050, "tz_offset": 1},
    {"city": "Sydney", "lat": -33.8688, "lon": 151.2093, "tz_offset": 10},
    {"city": "São Paulo", "lat": -23.5505, "lon": -46.6333, "tz_offset": -3},
    {"city": "Toronto", "lat": 43.6532, "lon": -79.3832, "tz_offset": -5},
    {"city": "Singapore", "lat": 1.3521, "lon": 103.8198, "tz_offset": 8},
    {"city": "Dubai", "lat": 25.2048, "lon": 55.2708, "tz_offset": 4},
    {"city": "Seoul", "lat": 37.5665, "lon": 126.9780, "tz_offset": 9},
    {"city": "Chicago", "lat": 41.8781, "lon": -87.6298, "tz_offset": -6},
    {"city": "Paris", "lat": 48.8566, "lon": 2.3522, "tz_offset": 1},
    {"city": "Beijing", "lat": 39.9042, "lon": 116.4074, "tz_offset": 8},
]

# ─── Device Fingerprints ─────────────────────────────────────────────────────
OS_OPTIONS = {
    "user": ["Windows 11", "Windows 10", "macOS 14", "macOS 13", "Ubuntu 22.04", "ChromeOS"],
    "service_account": ["Ubuntu 22.04 Server", "RHEL 9", "Debian 12", "Alpine 3.18"],
    "edge_device": ["FreeRTOS 10.5", "Zephyr 3.4", "Linux 5.15-rt", "VxWorks 7"],
}

PROTOCOLS = {
    "user": ["HTTPS", "SSH", "RDP", "VPN"],
    "service_account": ["HTTPS", "gRPC", "AMQP", "SSH"],
    "edge_device": ["MQTT", "CoAP", "HTTPS", "Modbus-TCP"],
}

# ─── Model Hyperparameters ────────────────────────────────────────────────────
BASELINE_CONFIG = {
    "isolation_forest": {
        "n_estimators": 200,
        "contamination": 0.025,       # Expected anomaly rate
        "random_state": 42,
        "max_samples": "auto",
    },
    "one_class_svm": {
        "kernel": "rbf",
        "gamma": "scale",
        "nu": 0.025,
    },
}

LSTM_CONFIG = {
    "sequence_length": 10,            # Events per sequence window
    "encoding_dim": 32,               # Latent space dimension
    "lstm_units_encoder": 64,
    "lstm_units_decoder": 64,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "validation_split": 0.15,
    "reconstruction_threshold_percentile": 97.5,  # Top 2.5% = anomaly
}

CLASSIFIER_CONFIG = {
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "use_label_encoder": False,
        "random_state": 42,
        "scale_pos_weight": 1,        # Handled via SMOTE instead
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 10,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    },
    "smote": {
        "random_state": 42,
        "k_neighbors": 3,
    },
}

# ─── Transformer Detector Hyperparameters ────────────────────────────────────
TRANSFORMER_CONFIG = {
    "sequence_length": 10,            # Must match LSTM for fair comparison
    "d_model": 64,                    # Embedding / internal dimension
    "num_heads": 4,                   # Multi-head attention heads
    "num_layers": 2,                  # Transformer encoder blocks
    "d_ff": 128,                      # Feed-forward hidden dim
    "dropout": 0.1,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.001,
    "validation_split": 0.15,
    "reconstruction_threshold_percentile": 97.5,
}

# ─── Risk Score Weights ───────────────────────────────────────────────────────
# How much each model contributes to the final 0-100 risk score (weights add up to 1.0)
RISK_WEIGHTS = {
    "isolation_forest_score": 0.35,
    "lstm_reconstruction_error": 0.35,
    "classifier_confidence": 0.20,
    "entity_risk_history": 0.10,
}

# ─── Severity Tiers ──────────────────────────────────────────────────────────
# Buckets the 0-100 risk score into labels + dashboard colors an analyst can scan quickly
SEVERITY_TIERS = {
    "Critical": {"min": 85, "max": 100, "color": "#DC2626"},
    "High":     {"min": 65, "max": 84,  "color": "#F59E0B"},
    "Medium":   {"min": 40, "max": 64,  "color": "#3B82F6"},
    "Low":      {"min": 20, "max": 39,  "color": "#6B7280"},
}

# ─── Cold-Start Config ────────────────────────────────────────────────────────
COLD_START_CONFIG = {
    "min_history_events": 20,         # Below this = cold-start entity
    "peer_group_size": 10,            # Number of peers for profiling
    "elevated_monitoring_days": 7,    # Days of elevated monitoring
    "threshold_multiplier": 0.7,      # Lower thresholds during cold-start
}

# ─── Concept Drift Config ────────────────────────────────────────────────────
DRIFT_CONFIG = {
    "window_size": 500,               # Events per sliding window
    "decay_factor": 0.95,             # Exponential decay for older events
    "drift_threshold": 0.05,          # Page-Hinkley drift detection threshold
    "recalibration_interval_days": 30,
}

# ─── Dashboard Config ─────────────────────────────────────────────────────────
DASHBOARD_CONFIG = {
    "streaming_interval_ms": 1000,    # Real-time simulation refresh rate
    "max_alerts_display": 100,        # Max alerts shown in queue
    "map_zoom": 2,                    # Default world map zoom level
    "timeline_events": 50,            # Events shown in entity timeline
}

# ─── Anomaly Type Labels ─────────────────────────────────────────────────────
ANOMALY_TYPES = [
    "normal",
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow",
    "insider_drift",
]

ANOMALY_DESCRIPTIONS = {
    "brute_force": "Rapid repeated failed-auth attempts from one source in a short window",
    "impossible_travel": "Login from geographically distant locations within implausible time gap",
    "credential_stuffing": "Many entity IDs targeted from few source IPs with high failure rate",
    "lateral_movement": "Entity accessing unusual sequence/breadth of resources never touched before",
    "device_spoofing": "Device reappearing with mismatched fingerprint (different OS/MAC than history)",
    "low_and_slow": "Gradual, small, off-hours resource access building up over days/weeks",
    "insider_drift": "Legitimate entity slowly expanding privilege or resource footprint",
}
