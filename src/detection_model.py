"""
DELIVERABLE 3: Detection Model — LSTM Autoencoder + Transformer Detector
==========================================================================
Sequence-aware anomaly detection using two approaches:

1. LSTM Autoencoder:
   - Architecture: Encoder (LSTM 64→32) → Latent Space → Decoder (LSTM 32→64)
   - Learns to reconstruct normal sequences; high error = anomaly

2. Transformer Detector:
   - Architecture: Positional Encoding → Transformer Encoder (multi-head attention) → Dense reconstruction
   - Self-attention captures non-local temporal dependencies that RNNs can miss

Both models take sliding windows of N events per entity as input and
flag deviations via reconstruction error.
"""

import numpy as np
import pandas as pd
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LSTM_CONFIG, TRANSFORMER_CONFIG, MODELS_DIR


def _build_sequences(X, entity_ids, seq_length):
    """
    Build sequences of consecutive events per entity.
    Returns: X_seq (n_sequences, seq_length, n_features), seq_entity_ids
    """
    sequences = []
    seq_entities = []

    unique_entities = np.unique(entity_ids)
    for entity in unique_entities:
        mask = entity_ids == entity
        entity_features = X[mask]

        if len(entity_features) < seq_length:
            # Pad short sequences with zeros
            padded = np.zeros((seq_length, entity_features.shape[1]))
            padded[-len(entity_features):] = entity_features
            sequences.append(padded)
            seq_entities.append(entity)
        else:
            # Sliding window
            for i in range(len(entity_features) - seq_length + 1):
                sequences.append(entity_features[i:i + seq_length])
                seq_entities.append(entity)

    return np.array(sequences), np.array(seq_entities)


class LSTMAutoencoder:
    """
    LSTM Autoencoder for sequence-based anomaly detection.
    Trains on normal event sequences and detects anomalies via
    high reconstruction error.
    """

    def __init__(self, config=None):
        self.config = config or LSTM_CONFIG
        self.model = None
        self.threshold = None
        self.history = None

    def _build_model(self, n_features):
        """Build the LSTM Autoencoder architecture."""
        # Import tensorflow here to avoid import errors if not installed
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        seq_length = self.config["sequence_length"]
        encoding_dim = self.config["encoding_dim"]
        lstm_enc = self.config["lstm_units_encoder"]
        lstm_dec = self.config["lstm_units_decoder"]

        # Encoder
        inputs = keras.Input(shape=(seq_length, n_features))
        encoded = layers.LSTM(lstm_enc, activation="tanh", return_sequences=True)(inputs)
        encoded = layers.LSTM(encoding_dim, activation="tanh", return_sequences=False)(encoded)

        # Latent space
        latent = layers.RepeatVector(seq_length)(encoded)

        # Decoder
        decoded = layers.LSTM(encoding_dim, activation="tanh", return_sequences=True)(latent)
        decoded = layers.LSTM(lstm_dec, activation="tanh", return_sequences=True)(decoded)
        outputs = layers.TimeDistributed(layers.Dense(n_features))(decoded)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss="mse",
        )

        print(f"[LSTM-AE] Model built: input_shape=({seq_length}, {n_features})")
        model.summary()
        return model

    def fit(self, X_train, entity_ids_train, labels_train):
        """
        Train the LSTM Autoencoder on normal event sequences.
        X_train: scaled feature matrix
        entity_ids_train: entity IDs for sequence building
        labels_train: labels (only normal events used for training)
        """
        import tensorflow as tf

        # Filter to normal events only (LabelEncoder sorts alphabetically, normal != 0)
        normal_enc = max(np.unique(labels_train))
        normal_mask = labels_train == normal_enc
        X_normal = X_train[normal_mask]
        entities_normal = entity_ids_train[normal_mask]

        n_features = X_normal.shape[1]
        seq_length = self.config["sequence_length"]

        # Build sequences
        print(f"[LSTM-AE] Building sequences (window={seq_length})...")
        X_seq, seq_entities = _build_sequences(X_normal, entities_normal, seq_length)
        print(f"[LSTM-AE] Created {len(X_seq)} sequences from {len(np.unique(entities_normal))} entities")

        if len(X_seq) == 0:
            print("[LSTM-AE] WARNING: No sequences created. Skipping training.")
            return self

        # Build and train model
        self.model = self._build_model(n_features)

        # Callbacks: stop early once validation loss stops improving, and slow the
        # learning rate down when progress stalls — both help avoid overfitting
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]

        print(f"[LSTM-AE] Training for up to {self.config['epochs']} epochs...")
        self.history = self.model.fit(
            X_seq, X_seq,  # Autoencoder: input = target
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_split=self.config["validation_split"],
            callbacks=callbacks,
            verbose=1,
        )

        # Compute reconstruction error threshold on training data
        recon = self.model.predict(X_seq, verbose=0)
        errors = np.mean(np.square(X_seq - recon), axis=(1, 2))
        self.threshold = np.percentile(
            errors, self.config["reconstruction_threshold_percentile"]
        )

        print(f"[LSTM-AE] Training complete.")
        print(f"  Reconstruction error threshold (p{self.config['reconstruction_threshold_percentile']}): {self.threshold:.6f}")
        print(f"  Mean error (normal): {errors.mean():.6f}")
        print(f"  Max error (normal): {errors.max():.6f}")

        self._save()
        return self

    def score(self, X, entity_ids):
        """
        Compute anomaly scores for events via reconstruction error.
        Returns per-event scores normalized to 0–1.
        """
        if self.model is None:
            print("[LSTM-AE] WARNING: Model not trained. Returning zero scores.")
            return np.zeros(len(X))

        seq_length = self.config["sequence_length"]
        X_seq, seq_entities = _build_sequences(X, entity_ids, seq_length)

        if len(X_seq) == 0:
            return np.zeros(len(X))

        # Get reconstruction errors
        recon = self.model.predict(X_seq, verbose=0)
        seq_errors = np.mean(np.square(X_seq - recon), axis=(1, 2))

        # Map sequence errors back to per-event scores
        # Each sequence covers events [i, i+seq_length). Assign the last event
        # in each sequence the sequence's error. Later sequences overwrite earlier
        # ones, so each event gets the error from its most recent sequence.
        event_scores = np.zeros(len(X))
        entity_event_idx = {}
        for i in range(len(X)):
            eid = entity_ids[i]
            if eid not in entity_event_idx:
                entity_event_idx[eid] = []
            entity_event_idx[eid].append(i)

        seq_idx = 0
        for entity in np.unique(seq_entities):
            entity_mask = seq_entities == entity
            entity_errors = seq_errors[entity_mask]
            indices = entity_event_idx.get(entity, [])
            for j, err in enumerate(entity_errors):
                event_pos = min(j + seq_length - 1, len(indices) - 1)
                event_scores[indices[event_pos]] = err
            # Fill early events with first sequence error
            if len(entity_errors) > 0:
                for j in range(min(seq_length - 1, len(indices))):
                    if event_scores[indices[j]] == 0:
                        event_scores[indices[j]] = entity_errors[0]

        # Normalize to 0–1
        if event_scores.max() > 0:
            event_scores = event_scores / max(event_scores.max(), self.threshold * 2)

        return np.clip(event_scores, 0, 1)

    def predict(self, X, entity_ids):
        """Binary prediction: 1 = anomaly, 0 = normal."""
        scores = self.score(X, entity_ids)
        if self.threshold is not None:
            # Convert the raw training-time error threshold into the same 0-1 scale as scores
            normalized_threshold = self.threshold / max(self.threshold * 2, 1e-6)
            return (scores > normalized_threshold).astype(int)
        return (scores > 0.5).astype(int)

    def _save(self):
        """Save model and threshold."""
        if self.model is not None:
            self.model.save(MODELS_DIR / "lstm_autoencoder.keras")
            joblib.dump(self.threshold, MODELS_DIR / "lstm_threshold.pkl")
            print(f"[LSTM-AE] Model saved to {MODELS_DIR}")

    def load(self):
        """Load model and threshold from disk."""
        import tensorflow as tf
        model_path = MODELS_DIR / "lstm_autoencoder.keras"
        threshold_path = MODELS_DIR / "lstm_threshold.pkl"
        if model_path.exists() and threshold_path.exists():
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(optimizer="adam", loss="mse")
            self.threshold = joblib.load(threshold_path)
            print("[LSTM-AE] Model loaded from disk.")
        else:
            print("[LSTM-AE] No saved model found.")


class TransformerDetector:
    """
    Transformer-based sequence anomaly detector.
    Uses self-attention to capture temporal patterns in event sequences,
    then reconstructs the input — high reconstruction error = anomaly.
    """

    def __init__(self, config=None):
        self.config = config or TRANSFORMER_CONFIG
        self.model = None
        self.threshold = None
        self.history = None

    def _build_model(self, n_features):
        """Build a Transformer Encoder with dense reconstruction head."""
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        seq_length = self.config["sequence_length"]
        d_model = self.config["d_model"]
        num_heads = self.config["num_heads"]
        num_layers = self.config["num_layers"]
        d_ff = self.config["d_ff"]
        dropout = self.config["dropout"]

        inputs = keras.Input(shape=(seq_length, n_features))

        # Project raw features into d_model dimension
        x = layers.Dense(d_model)(inputs)

        # Fixed sinusoidal positional encoding (Vaswani et al. "Attention is All You Need")
        positions = np.arange(seq_length)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, 2 * (dims // 2) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_enc_matrix = tf.constant(angles, dtype=tf.float32)
        x = x + pos_enc_matrix

        # Stack of Transformer encoder blocks
        for _ in range(num_layers):
            # Multi-head self-attention
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads
            )(x, x)
            attn_output = layers.Dropout(dropout)(attn_output)
            x = layers.LayerNormalization()(x + attn_output)

            # Feed-forward network
            ff = layers.Dense(d_ff, activation="relu")(x)
            ff = layers.Dense(d_model)(ff)
            ff = layers.Dropout(dropout)(ff)
            x = layers.LayerNormalization()(x + ff)

        # Reconstruct original features from the encoded representation
        outputs = layers.Dense(n_features)(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss="mse",
        )

        print(f"[Transformer] Model built: input_shape=({seq_length}, {n_features}), "
              f"d_model={d_model}, heads={num_heads}, layers={num_layers}")
        model.summary()
        return model

    def fit(self, X_train, entity_ids_train, labels_train):
        """
        Train the Transformer on normal event sequences.
        Same interface as LSTMAutoencoder for easy comparison.
        """
        import tensorflow as tf

        normal_enc = max(np.unique(labels_train))
        normal_mask = labels_train == normal_enc
        X_normal = X_train[normal_mask]
        entities_normal = entity_ids_train[normal_mask]

        n_features = X_normal.shape[1]
        seq_length = self.config["sequence_length"]

        print(f"[Transformer] Building sequences (window={seq_length})...")
        X_seq, seq_entities = _build_sequences(X_normal, entities_normal, seq_length)
        print(f"[Transformer] Created {len(X_seq)} sequences from {len(np.unique(entities_normal))} entities")

        if len(X_seq) == 0:
            print("[Transformer] WARNING: No sequences created. Skipping training.")
            return self

        self.model = self._build_model(n_features)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
            ),
        ]

        print(f"[Transformer] Training for up to {self.config['epochs']} epochs...")
        self.history = self.model.fit(
            X_seq, X_seq,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            validation_split=self.config["validation_split"],
            callbacks=callbacks,
            verbose=1,
        )

        # Compute reconstruction error threshold on training data
        recon = self.model.predict(X_seq, verbose=0)
        errors = np.mean(np.square(X_seq - recon), axis=(1, 2))
        self.threshold = np.percentile(
            errors, self.config["reconstruction_threshold_percentile"]
        )

        print(f"[Transformer] Training complete.")
        print(f"  Reconstruction error threshold (p{self.config['reconstruction_threshold_percentile']}): {self.threshold:.6f}")
        print(f"  Mean error (normal): {errors.mean():.6f}")
        print(f"  Max error (normal): {errors.max():.6f}")

        self._save()
        return self

    def score(self, X, entity_ids):
        """Compute per-event anomaly scores via reconstruction error (0-1 normalized)."""
        if self.model is None:
            print("[Transformer] WARNING: Model not trained. Returning zero scores.")
            return np.zeros(len(X))

        seq_length = self.config["sequence_length"]
        X_seq, seq_entities = _build_sequences(X, entity_ids, seq_length)

        if len(X_seq) == 0:
            return np.zeros(len(X))

        recon = self.model.predict(X_seq, verbose=0)
        seq_errors = np.mean(np.square(X_seq - recon), axis=(1, 2))

        # Map sequence errors back to per-event scores (same logic as LSTM)
        event_scores = np.zeros(len(X))
        entity_event_idx = {}
        for i in range(len(X)):
            eid = entity_ids[i]
            if eid not in entity_event_idx:
                entity_event_idx[eid] = []
            entity_event_idx[eid].append(i)

        for entity in np.unique(seq_entities):
            entity_mask = seq_entities == entity
            entity_errors = seq_errors[entity_mask]
            indices = entity_event_idx.get(entity, [])
            for j, err in enumerate(entity_errors):
                event_pos = min(j + seq_length - 1, len(indices) - 1)
                event_scores[indices[event_pos]] = err
            if len(entity_errors) > 0:
                for j in range(min(seq_length - 1, len(indices))):
                    if event_scores[indices[j]] == 0:
                        event_scores[indices[j]] = entity_errors[0]

        if event_scores.max() > 0:
            event_scores = event_scores / max(event_scores.max(), self.threshold * 2)

        return np.clip(event_scores, 0, 1)

    def predict(self, X, entity_ids):
        """Binary prediction: 1 = anomaly, 0 = normal."""
        scores = self.score(X, entity_ids)
        if self.threshold is not None:
            normalized_threshold = self.threshold / max(self.threshold * 2, 1e-6)
            return (scores > normalized_threshold).astype(int)
        return (scores > 0.5).astype(int)

    def _save(self):
        """Save model and threshold."""
        if self.model is not None:
            self.model.save(MODELS_DIR / "transformer_detector.keras")
            joblib.dump(self.threshold, MODELS_DIR / "transformer_threshold.pkl")
            print(f"[Transformer] Model saved to {MODELS_DIR}")

    def load(self):
        """Load model and threshold from disk."""
        import tensorflow as tf
        model_path = MODELS_DIR / "transformer_detector.keras"
        threshold_path = MODELS_DIR / "transformer_threshold.pkl"
        if model_path.exists() and threshold_path.exists():
            self.model = tf.keras.models.load_model(model_path, compile=False)
            self.model.compile(optimizer="adam", loss="mse")
            self.threshold = joblib.load(threshold_path)
            print("[Transformer] Model loaded from disk.")
        else:
            print("[Transformer] No saved model found.")
