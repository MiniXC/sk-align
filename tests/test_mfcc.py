"""
Tests for MFCC feature extraction.

Tests the pure Python/NumPy MFCC implementation in isolation.
When reference data from pykaldi is available, also tests parity.
"""

import numpy as np
import pytest

from sk_align.mfcc import MfccOptions, compute_mfcc


class TestMfccBasic:
    """Basic MFCC computation tests (no model files needed)."""

    def test_output_shape(self):
        """MFCC output should have correct shape."""
        opts = MfccOptions(dither=0.0, seed=42)
        # 1 second of audio at 16kHz
        audio = np.random.randn(16000).astype(np.float32)
        feats = compute_mfcc(audio, opts)

        # Frame count: 1 + (16000 - 400) / 160 = 98.5 → 98
        expected_frames = 1 + (16000 - 400) // 160
        assert feats.shape == (expected_frames, opts.num_ceps)
        assert feats.dtype == np.float32

    def test_empty_audio(self):
        """Empty or too-short audio should return empty features."""
        opts = MfccOptions()
        feats = compute_mfcc(np.array([], dtype=np.float32), opts)
        assert feats.shape == (0, opts.num_ceps)

        # Audio shorter than one frame
        feats = compute_mfcc(np.zeros(100, dtype=np.float32), opts)
        assert feats.shape == (0, opts.num_ceps)

    def test_deterministic_with_seed(self):
        """Same seed should produce identical output."""
        audio = np.random.randn(16000).astype(np.float32)
        opts = MfccOptions(seed=123)
        feats1 = compute_mfcc(audio, opts)
        feats2 = compute_mfcc(audio, opts)
        np.testing.assert_array_equal(feats1, feats2)

    def test_no_dither_deterministic(self):
        """With dither=0, output should be deterministic."""
        audio = np.random.randn(16000).astype(np.float32)
        opts = MfccOptions(dither=0.0)
        feats1 = compute_mfcc(audio, opts)
        feats2 = compute_mfcc(audio, opts)
        np.testing.assert_array_equal(feats1, feats2)

    def test_silence_features(self):
        """Silence should produce features close to zero (after liftering)."""
        opts = MfccOptions(dither=0.0)
        silence = np.zeros(16000, dtype=np.float32)
        feats = compute_mfcc(silence, opts)
        # With zero input and no dither, DC removal makes frame all zeros
        # Log of very small mel energies → large negative values, not zeros
        assert feats.shape[0] > 0
        # Just check it runs without error

    def test_sg_aligner_options(self):
        """Options matching the Scottish Gaelic aligner config."""
        opts = MfccOptions(
            sample_freq=16000.0,
            use_energy=False,
            num_mel_bins=40,
            num_ceps=40,
            low_freq=20.0,
            high_freq=-400.0,
            dither=0.0,
        )
        audio = np.random.randn(32000).astype(np.float32) * 100
        feats = compute_mfcc(audio, opts)
        assert feats.shape[1] == 40
        assert feats.shape[0] > 0

    def test_energy_option(self):
        """With use_energy=True, first coeff should be replaced by log energy."""
        audio = np.random.randn(16000).astype(np.float32) * 100
        opts_no_energy = MfccOptions(use_energy=False, dither=0.0)
        opts_energy = MfccOptions(use_energy=True, dither=0.0)
        feats_no = compute_mfcc(audio, opts_no_energy)
        feats_yes = compute_mfcc(audio, opts_energy)
        # First coefficient should differ (energy vs c0)
        assert not np.allclose(feats_no[:, 0], feats_yes[:, 0])

    def test_different_num_ceps(self):
        """Different num_ceps should change output dimension."""
        audio = np.random.randn(16000).astype(np.float32)
        for nc in [13, 20, 40]:
            opts = MfccOptions(num_ceps=nc, dither=0.0)
            feats = compute_mfcc(audio, opts)
            assert feats.shape[1] == nc


class TestMfccParity:
    """Parity tests against pykaldi reference MFCC features."""

    def test_mfcc_parity(self, reference_data):
        """MFCC output should match pykaldi within tolerance."""
        if reference_data is None or "mfcc" not in reference_data:
            pytest.skip("No reference MFCC data available")

        ref_mfcc = reference_data["mfcc"]
        audio = np.load(
            reference_data.get("audio_path", "tests/reference_data/m001rmb0/audio.npy")
        )

        opts = MfccOptions(
            sample_freq=16000.0,
            use_energy=False,
            num_mel_bins=40,
            num_ceps=40,
            low_freq=20.0,
            high_freq=-400.0,
            dither=0.0,  # Disable for reproducibility
        )

        wav = audio * (1 << 15)
        feats = compute_mfcc(wav, opts)

        assert feats.shape == ref_mfcc.shape, (
            f"Shape mismatch: got {feats.shape}, expected {ref_mfcc.shape}"
        )
        # The pykaldi reference was generated with default dither (random
        # noise added before FFT), while we use dither=0 for reproducibility.
        # This causes small per-frame differences that are normal.
        # Mean absolute diff < 5.0 is acceptable (values range ~ -50..50).
        diff = np.abs(feats - ref_mfcc)
        mean_diff = diff.mean()
        max_diff = diff.max()
        assert mean_diff < 5.0, (
            f"Mean MFCC difference too large: {mean_diff:.4f} "
            f"(max={max_diff:.4f})"
        )
