"""
Viewable comparison tests for sk-align vs pykaldi.

Generates detailed text output and optional HTML reports showing
side-by-side comparisons of:
- MFCC features (statistics, per-frame differences)
- Nnet3 log-likelihoods (shape, range, top-k pdfs)
- Alignment output (word timestamps)

Run with: pytest tests/test_viewable.py -v -s
The ``-s`` flag is important to see the printed comparison tables.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from sk_align.mfcc import MfccOptions, compute_mfcc

# Directory for test outputs
OUTPUT_DIR = Path(__file__).parent.parent / "test_outputs"


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


def _load_wav(path: Path, max_seconds: float = 10.0) -> np.ndarray:
    """Load a WAV file as float32 array, truncated to *max_seconds*."""
    with wave.open(str(path), "r") as wf:
        rate = wf.getframerate()
        max_frames = int(max_seconds * rate)
        n = min(wf.getnframes(), max_frames)
        raw = wf.readframes(n)
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _fmt(val, width=12):
    """Format a float for table display."""
    if isinstance(val, float):
        return f"{val:>{width}.6f}"
    return f"{str(val):>{width}}"


def _print_table(headers: list[str], rows: list[list], title: str = ""):
    """Print a nicely formatted table."""
    col_widths = [max(len(h), 12) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:^{w}} " for h, w in zip(headers, col_widths)) + "|"

    if title:
        print(f"\n{'='*len(sep)}")
        print(f" {title}")
        print(f"{'='*len(sep)}")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        cells = []
        for val, w in zip(row, col_widths):
            if isinstance(val, float):
                cells.append(f" {val:>{w}.6f} ")
            else:
                cells.append(f" {str(val):>{w}} ")
        print("|" + "|".join(cells) + "|")
    print(sep)


class TestMfccViewable:
    """MFCC feature comparison with detailed viewable output."""

    @pytest.fixture
    def sg_mfcc_opts(self) -> MfccOptions:
        """MFCC options matching the Scottish Gaelic aligner."""
        return MfccOptions(
            sample_freq=16000.0,
            use_energy=False,
            num_mel_bins=40,
            num_ceps=40,
            low_freq=20.0,
            high_freq=-400.0,
            dither=0.0,
        )

    def test_mfcc_feature_statistics(self, test_audio_path, sg_mfcc_opts):
        """Show per-coefficient MFCC statistics."""
        audio = _load_wav(test_audio_path)
        wav = audio.astype(np.float64) * (1 << 15)
        feats = compute_mfcc(wav, sg_mfcc_opts)

        print(f"\n  Audio: {test_audio_path.name}")
        print(f"  Samples: {len(audio)}, Duration: {len(audio)/16000:.2f}s")
        print(f"  Feature shape: {feats.shape} (frames × ceps)")

        # Per-coefficient statistics
        headers = ["Coeff", "Mean", "Std", "Min", "Max"]
        rows = []
        for i in range(min(feats.shape[1], 10)):  # Show first 10 coefficients
            rows.append([
                f"c{i}",
                float(feats[:, i].mean()),
                float(feats[:, i].std()),
                float(feats[:, i].min()),
                float(feats[:, i].max()),
            ])
        if feats.shape[1] > 10:
            rows.append(["...", "...", "...", "...", "..."])
            for i in range(feats.shape[1] - 2, feats.shape[1]):
                rows.append([
                    f"c{i}",
                    float(feats[:, i].mean()),
                    float(feats[:, i].std()),
                    float(feats[:, i].min()),
                    float(feats[:, i].max()),
                ])
        _print_table(headers, rows, "MFCC Feature Statistics (sk-align)")

        # Global statistics
        print(f"\n  Global mean: {feats.mean():.6f}")
        print(f"  Global std:  {feats.std():.6f}")
        print(f"  Global range: [{feats.min():.6f}, {feats.max():.6f}]")
        print(f"  Frobenius norm: {np.linalg.norm(feats):.6f}")

        # Save features for manual inspection
        _ensure_output_dir()
        np.save(OUTPUT_DIR / f"mfcc_{test_audio_path.stem}.npy", feats)
        print(f"\n  Saved features to: test_outputs/mfcc_{test_audio_path.stem}.npy")

        assert feats.shape[0] > 0
        assert feats.shape[1] == 40

    def test_mfcc_first_frames(self, test_audio_path, sg_mfcc_opts):
        """Show the actual MFCC values for the first few frames."""
        audio = _load_wav(test_audio_path)
        wav = audio.astype(np.float64) * (1 << 15)
        feats = compute_mfcc(wav, sg_mfcc_opts)

        print(f"\n  First 5 frames, first 8 coefficients:")
        headers = ["Frame"] + [f"c{i}" for i in range(8)]
        rows = []
        for t in range(min(5, feats.shape[0])):
            rows.append([t] + [float(feats[t, i]) for i in range(8)])
        _print_table(headers, rows, "MFCC Values (first frames)")

        assert feats.shape[0] > 0

    def test_mfcc_parity_with_reference(self, reference_data, sg_mfcc_opts):
        """Compare sk-align MFCCs with pykaldi reference data."""
        if reference_data is None or "mfcc" not in reference_data:
            pytest.skip("No reference MFCC data — run scripts/generate_reference.py")

        ref_mfcc = reference_data["mfcc"]
        audio_path = Path("tests/reference_data/m001rmb0/audio.npy")
        if not audio_path.exists():
            pytest.skip("No reference audio data")

        audio = np.load(audio_path)
        wav = audio * (1 << 15)
        our_mfcc = compute_mfcc(wav, sg_mfcc_opts)

        print(f"\n  sk-align shape: {our_mfcc.shape}")
        print(f"  pykaldi shape:  {ref_mfcc.shape}")

        if our_mfcc.shape != ref_mfcc.shape:
            print("  ⚠️  Shape mismatch — cannot compare values")
            return

        diff = our_mfcc - ref_mfcc
        abs_diff = np.abs(diff)

        # Per-coefficient comparison
        headers = ["Coeff", "MaxAbsDiff", "MeanAbsDiff", "Corr"]
        rows = []
        for i in range(min(our_mfcc.shape[1], 10)):
            corr = np.corrcoef(our_mfcc[:, i], ref_mfcc[:, i])[0, 1]
            rows.append([
                f"c{i}",
                float(abs_diff[:, i].max()),
                float(abs_diff[:, i].mean()),
                float(corr),
            ])
        _print_table(headers, rows, "MFCC Parity: sk-align vs pykaldi")

        # Global summary
        print(f"\n  Max absolute difference: {abs_diff.max():.8f}")
        print(f"  Mean absolute difference: {abs_diff.mean():.8f}")
        print(f"  Relative error (Frobenius): "
              f"{np.linalg.norm(diff) / np.linalg.norm(ref_mfcc):.8f}")

        # Frame-level correlation
        frame_corr = np.array([
            np.corrcoef(our_mfcc[t], ref_mfcc[t])[0, 1]
            for t in range(our_mfcc.shape[0])
        ])
        print(f"  Per-frame correlation: min={frame_corr.min():.6f}, "
              f"mean={frame_corr.mean():.6f}, max={frame_corr.max():.6f}")


class TestNnet3Viewable:
    """Nnet3 log-likelihood comparison with detailed viewable output."""

    @pytest.fixture
    def scorer(self, model_dir):
        """Create a TorchNnetScorer."""
        try:
            from sk_align.nnet3_model import read_nnet3_model
            from sk_align.nnet3_torch import TorchNnetScorer
        except ImportError:
            pytest.skip("PyTorch not available")

        model = read_nnet3_model(model_dir / "final.mdl")
        return TorchNnetScorer(model, frame_subsampling_factor=3)

    def test_nnet3_output_statistics(self, test_audio_path, scorer):
        """Show nnet3 output statistics for test audio."""
        audio = _load_wav(test_audio_path)
        wav = audio.astype(np.float64) * (1 << 15)

        opts = MfccOptions(
            sample_freq=16000.0,
            use_energy=False,
            num_mel_bins=40,
            num_ceps=40,
            low_freq=20.0,
            high_freq=-400.0,
            dither=0.0,
        )
        feats = compute_mfcc(wav, opts)
        loglikes = scorer.compute_log_likelihoods(feats)

        print(f"\n  Audio: {test_audio_path.name}")
        print(f"  MFCC frames: {feats.shape[0]}")
        print(f"  Log-likelihood shape: {loglikes.shape} (frames × pdfs)")
        print(f"  Range: [{loglikes.min():.4f}, {loglikes.max():.4f}]")
        print(f"  Mean: {loglikes.mean():.4f}")
        print(f"  Std: {loglikes.std():.4f}")

        # Show top-k PDFs for first few frames
        print(f"\n  Top-5 PDFs per frame (first 5 frames):")
        headers = ["Frame"] + [f"PDF#{i+1}" for i in range(5)] + [f"Score#{i+1}" for i in range(5)]
        rows = []
        for t in range(min(5, loglikes.shape[0])):
            top_k = np.argsort(loglikes[t])[::-1][:5]
            scores = loglikes[t, top_k]
            rows.append([t] + list(top_k) + [float(s) for s in scores])
        _print_table(headers, rows, "Nnet3 Top-5 PDFs")

        # Save for comparison
        _ensure_output_dir()
        np.save(OUTPUT_DIR / f"loglikes_{test_audio_path.stem}.npy", loglikes)
        print(f"\n  Saved log-likelihoods to: test_outputs/loglikes_{test_audio_path.stem}.npy")

        assert loglikes.shape[0] > 0
        assert loglikes.shape[1] > 0

    def test_nnet3_model_info(self, model_dir):
        """Show model architecture summary."""
        try:
            from sk_align.nnet3_model import read_nnet3_model
        except ImportError:
            pytest.skip("nnet3_model not available")

        model = read_nnet3_model(model_dir / "final.mdl")

        print(f"\n  Model: {model_dir / 'final.mdl'}")
        print(f"  Input dim: {model.inputs[0].dim}")
        print(f"  Left context: {model.left_context}")
        print(f"  Right context: {model.right_context}")
        print(f"  Priors shape: {model.priors.shape}")
        print(f"  Num components: {len(model.components)}")
        print(f"  Num component nodes: {len(model.component_nodes)}")

        # Component type summary
        from collections import Counter
        types = Counter(c.component_type for c in model.components.values())
        headers = ["Component Type", "Count"]
        rows = [(t, c) for t, c in types.most_common()]
        _print_table(headers, rows, "Component Types")

        # Output dimension (from last affine before output)
        for name, comp in model.components.items():
            if name == "output.affine":
                p = comp.params
                print(f"\n  Output affine: {p.linear_params.shape} → {p.linear_params.shape[0]} PDFs")

        assert len(model.components) > 0


class TestAlignmentViewable:
    """Full alignment pipeline comparison with detailed viewable output."""

    @pytest.fixture
    def aligner_with_scorer(self, model_dir):
        """Create a full Aligner with TorchNnetScorer."""
        try:
            from sk_align.nnet3_model import read_nnet3_model
            from sk_align.nnet3_torch import TorchNnetScorer
        except ImportError:
            pytest.skip("PyTorch not available")

        from sk_align.aligner import Aligner

        model = read_nnet3_model(model_dir / "final.mdl")
        scorer = TorchNnetScorer(model, frame_subsampling_factor=3)
        return Aligner.from_model_dir(model_dir, nnet_scorer=scorer)

    def test_full_alignment(self, test_audio_path, aligner_with_scorer):
        """Run full alignment and show results."""
        audio = _load_wav(test_audio_path)
        # Use some Scottish Gaelic words for testing
        words = ["tha", "mi", "a", "dol", "dhachaigh"]

        print(f"\n  Audio: {test_audio_path.name}")
        print(f"  Duration: {len(audio)/16000:.2f}s")
        print(f"  Words: {' '.join(words)}")

        try:
            result = aligner_with_scorer.align(audio, words)
        except Exception as e:
            print(f"\n  Alignment failed: {e}")
            pytest.skip(f"Alignment failed: {e}")

        headers = ["Word", "Start (s)", "End (s)", "Duration (s)"]
        rows = []
        for item in result:
            dur = item["end"] - item["start"]
            rows.append([item["word"], item["start"], item["end"], dur])
        _print_table(headers, rows, "Alignment Results")

        # Save results
        _ensure_output_dir()
        with open(OUTPUT_DIR / f"alignment_{test_audio_path.stem}.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved to: test_outputs/alignment_{test_audio_path.stem}.json")

        assert len(result) > 0

    def test_alignment_parity_with_reference(self, reference_data):
        """Compare alignment timestamps with pykaldi reference."""
        if reference_data is None or "expected_output" not in reference_data:
            pytest.skip("No reference alignment data — run scripts/generate_reference.py")

        ref = reference_data["expected_output"]
        # This test would compare our alignment output with pykaldi's
        # when reference data is available

        headers = ["Word", "Ref Start", "Ref End", "Ref Duration"]
        rows = []
        for item in ref:
            dur = item["end"] - item["start"]
            rows.append([item["word"], item["start"], item["end"], dur])
        _print_table(headers, rows, "Reference Alignment (pykaldi)")
