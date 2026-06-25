import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Dataset configuration
# By default we use a folder named "database" (containing "real" and "fake")
# because that matches your current project structure.
DATASET_DIR = Path(os.getenv("DATASET_DIR", BASE_DIR.parent / "database"))
REAL_SUBDIR_NAME = os.getenv("REAL_SUBDIR_NAME", "real")
FAKE_SUBDIR_NAME = os.getenv("FAKE_SUBDIR_NAME", "fake")

# Audio / feature configuration (MFCC-based)
SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 4.0  # audio will be trimmed / padded to this duration
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

# Model configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "artifacts"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = MODEL_DIR / "cnn_lstm_mfcc.h5"

# Training configuration
# Tuned for faster training on CPU while still keeping good accuracy.
BATCH_SIZE = 64
EPOCHS = 8
VALIDATION_SPLIT = 0.15
RANDOM_SEED = 42


def get_real_dir() -> Path:
    return DATASET_DIR / REAL_SUBDIR_NAME


def get_fake_dir() -> Path:
    return DATASET_DIR / FAKE_SUBDIR_NAME


def ensure_dataset_structure() -> None:
    """
    Create the expected dataset directory structure if it does not already exist.
    """
    get_real_dir().mkdir(parents=True, exist_ok=True)
    get_fake_dir().mkdir(parents=True, exist_ok=True)

