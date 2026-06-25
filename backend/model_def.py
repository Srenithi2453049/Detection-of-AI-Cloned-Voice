from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from . import config


def build_cnn_lstm_model(time_steps: int, n_mfcc: int) -> tf.keras.Model:
    """
    Hybrid CNN-LSTM model operating on MFCC features.
    Input shape: (time_steps, n_mfcc)

    This variant is intentionally lighter so it trains
    faster on CPU while preserving good performance.
    """
    input_layer = layers.Input(shape=(time_steps, n_mfcc), name="mfcc_input")

    # Add a channel dimension: (batch, time, n_mfcc, 1)
    x = layers.Reshape((time_steps, n_mfcc, 1))(input_layer)

    # CNN block (smaller and faster)
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Prepare for LSTM: collapse the frequency/channel dimensions, keep time
    shape = x.shape
    time_dim = shape[1]
    feature_dim = int(shape[2] * shape[3])
    x = layers.Reshape((time_dim, feature_dim))(x)

    # LSTM block (reduced units)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.2)
    )(x)

    # Dense classification head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid", name="deepfake_score")(x)

    model = models.Model(inputs=input_layer, outputs=output, name="cnn_lstm_mfcc")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def save_model(model: tf.keras.Model, path: Path | None = None) -> None:
    if path is None:
        path = config.MODEL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path: Path | None = None) -> tf.keras.Model:
    if path is None:
        path = config.MODEL_FILE
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}")
    return tf.keras.models.load_model(path)


def predict_proba(model: tf.keras.Model, mfcc_batch: np.ndarray) -> np.ndarray:
    """Run inference on a batch of MFCC tensors and return fake probabilities."""
    preds = model.predict(mfcc_batch, verbose=0)
    return preds.squeeze(axis=-1)

