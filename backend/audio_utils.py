from __future__ import annotations
import random
from pathlib import Path
from typing import List, Tuple
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from . import config
def load_audio_mono(path: Path) -> np.ndarray:
    """Load an audio file as mono at the configured sample rate."""
    y, _ = librosa.load(
        str(path),
        sr=config.SAMPLE_RATE,
        mono=True,
        res_type="kaiser_fast",
    )
    return y
def pad_or_trim(y: np.ndarray) -> np.ndarray:
    """Pad or trim the audio signal to a fixed length."""
    target_length = int(config.SAMPLE_RATE * config.MAX_DURATION_SECONDS)
    if len(y) < target_length:
        pad_width = target_length - len(y)
        y = np.pad(y, (0, pad_width), mode="constant")
    elif len(y) > target_length:
        y = y[:target_length]
    return y
def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """
    Extract MFCC features from an audio signal.
    Returns an array of shape (time_steps, n_mfcc).
    """
    y = pad_or_trim(y)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    # Transpose to (time, n_mfcc)
    mfcc = mfcc.T
    return mfcc.astype("float32")
def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the labeled dataset from disk and return train/test splits.

    Expected structure (already matching your project):
        database/
            real/
                *.wav
            fake/
                *.wav
    """
    config.ensure_dataset_structure()

    real_files = sorted(
        [p for p in config.get_real_dir().glob("**/*") if p.is_file()]
    )
    fake_files = sorted(
        [p for p in config.get_fake_dir().glob("**/*") if p.is_file()]
    )

    if not real_files or not fake_files:
        raise RuntimeError(
            f"No audio files found in dataset directories: "
            f"{config.get_real_dir()} and {config.get_fake_dir()}"
        )

    paths: List[Path] = []
    labels: List[int] = []

    for p in real_files:
        paths.append(p)
        labels.append(0)  # 0 = real
    for p in fake_files:
        paths.append(p)
        labels.append(1)  # 1 = fake

    # Shuffle deterministically
    rng = random.Random(config.RANDOM_SEED)
    indices = list(range(len(paths)))
    rng.shuffle(indices)
    paths = [paths[i] for i in indices]
    labels = [labels[i] for i in indices]

    features: List[np.ndarray] = []
    for p in paths:
        y = load_audio_mono(p)
        mfcc = extract_mfcc(y)
        features.append(mfcc)

    # All MFCC tensors should have the same shape after padding
    X = np.stack(features, axis=0)  # (num_samples, time_steps, n_mfcc)
    y_arr = np.array(labels, dtype="int32")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_arr,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=y_arr,
    )
    return X_train, X_test, y_train, y_test


def extract_mfcc_from_file(path: Path) -> np.ndarray:
    """Convenience helper to load a single audio file and compute MFCC features."""
    y = load_audio_mono(path)
    mfcc = extract_mfcc(y)
    # Add batch dimension
    return np.expand_dims(mfcc, axis=0)

