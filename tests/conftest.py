"""
Test fixtures for sk-align.

Provides:
- Model download (from HuggingFace, cached)
- Test audio files
- Reference alignment data
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

# Root of the sk-align package
PACKAGE_ROOT = Path(__file__).parent.parent
# Root of the workspace (sk-align repo)  
WORKSPACE_ROOT = PACKAGE_ROOT.parent
TEST_FILES_DIR = PACKAGE_ROOT / "test_files"
REFERENCE_DIR = PACKAGE_ROOT / "tests" / "reference_data"


def _download_model() -> Path:
    """Download the nnet3 alignment model from HuggingFace (cached)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        pytest.skip("huggingface_hub not installed — run: pip install huggingface_hub")

    model_id = os.getenv(
        "ALIGNMENT_MODEL_ID", "eist-edinburgh/nnet3_alignment_model"
    )
    cache_dir = PACKAGE_ROOT / ".model_cache"
    try:
        path = snapshot_download(
            model_id,
            cache_dir=str(cache_dir),
            token=os.getenv("HF_TOKEN"),
        )
        return Path(path)
    except Exception as e:
        pytest.skip(f"Failed to download model: {e}")


@pytest.fixture(scope="session")
def model_dir() -> Path:
    """Path to the Kaldi nnet3 alignment model directory."""
    # Check environment variable first
    env_path = os.getenv("ALIGNMENT_MODEL_DIR")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    # Try downloading from HuggingFace
    return _download_model()


@pytest.fixture(scope="session")
def test_audio_path() -> Path:
    """Path to a test WAV file."""
    path = TEST_FILES_DIR / "m001rmb0.wav"
    if not path.exists():
        # Also check workspace test_files
        path = WORKSPACE_ROOT / "test_files" / "m001rmb0.wav"
    if not path.exists():
        pytest.skip(f"Test audio not found at {path}")
    return path


@pytest.fixture
def test_audio(test_audio_path) -> np.ndarray:
    """Load test audio as float32 array."""
    import wave
    with wave.open(str(test_audio_path), "r") as wf:
        raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


@pytest.fixture
def reference_data() -> dict | None:
    """Load pre-computed reference alignment data (if available)."""
    ref_dir = REFERENCE_DIR / "m001rmb0"
    if not ref_dir.exists():
        return None
    data = {}
    if (ref_dir / "mfcc_features.npy").exists():
        data["mfcc"] = np.load(ref_dir / "mfcc_features.npy")
    if (ref_dir / "alignment.npy").exists():
        data["alignment"] = np.load(ref_dir / "alignment.npy")
    if (ref_dir / "expected_output.json").exists():
        with open(ref_dir / "expected_output.json") as f:
            data["expected_output"] = json.load(f)
    if (ref_dir / "metadata.json").exists():
        with open(ref_dir / "metadata.json") as f:
            data["metadata"] = json.load(f)
    return data if data else None
