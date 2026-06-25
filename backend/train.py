from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from . import audio_utils, config, model_def


def train() -> None:
    config.ensure_dataset_structure()

    X_train, X_test, y_train, y_test = audio_utils.load_dataset()

    time_steps = X_train.shape[1]
    n_mfcc = X_train.shape[2]

    model = model_def.build_cnn_lstm_model(time_steps=time_steps, n_mfcc=n_mfcc)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_FILE),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=config.VALIDATION_SPLIT,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        class_weight=_compute_class_weights(y_train),
        verbose=2,
    )

    # Save final model and training history
    model_def.save_model(model, config.MODEL_FILE)
    _save_history(history.history, config.MODEL_DIR / "training_history.json")

    # Evaluation on held-out test set
    y_proba = model_def.predict_proba(model, X_test)
    y_pred = (y_proba >= 0.5).astype("int32")

    report = classification_report(
        y_test,
        y_pred,
        target_names=["real", "fake"],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics_path = config.MODEL_DIR / "test_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"classification_report": report, "confusion_matrix": cm}, f, indent=2)

    print(f"Saved model to: {config.MODEL_FILE}")
    print(f"Saved test metrics to: {metrics_path}")


def _compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute simple inverse-frequency class weights for imbalanced datasets."""
    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[0]
    weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    return weights


def _save_history(history: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    train()

