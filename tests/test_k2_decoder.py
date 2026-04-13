"""
Tests for the k2-based Viterbi decoder.

Includes:
- Correctness tests on synthetic graphs
- Speed comparison vs the pure-Python Viterbi decoder
- End-to-end speed comparison vs the old pykaldi-based aligner
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from sk_align.graph import HmmGraph, HmmGraphArc
from sk_align.k2_decoder import viterbi_decode_k2

REFERENCE_DIR = Path(__file__).parent / "reference_data"
PYKALDI_ALIGNER = (
    Path(__file__).parent.parent.parent / "scottish-gaelic-subtitling" / "aligner.py"
)
PYKALDI_PYTHON = Path.home() / "miniforge3" / "envs" / "pykaldi" / "bin" / "python"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_graph() -> tuple[HmmGraph, list[int]]:
    """3-state HMM with self-loops.  Returns (graph, tid_to_pdf)."""
    graph = HmmGraph()
    s0 = graph.add_state()
    s1 = graph.add_state()
    s2 = graph.add_state()
    s_final = graph.add_state(is_final=True)
    graph.start = s0

    graph.add_arc(s0, HmmGraphArc(tid=1, word_id=1, weight=0.5, nextstate=s1))
    graph.add_arc(s0, HmmGraphArc(tid=2, word_id=0, weight=0.3, nextstate=s0))
    graph.add_arc(s1, HmmGraphArc(tid=3, word_id=2, weight=0.5, nextstate=s2))
    graph.add_arc(s1, HmmGraphArc(tid=4, word_id=0, weight=0.3, nextstate=s1))
    graph.add_arc(s2, HmmGraphArc(tid=5, word_id=0, weight=0.5, nextstate=s_final))
    graph.add_arc(s2, HmmGraphArc(tid=6, word_id=0, weight=0.3, nextstate=s2))

    tid_to_pdf = [0, 0, 0, 1, 1, 2, 2]
    return graph, tid_to_pdf


def _make_chain_graph(n_phones: int, n_hmm_states: int = 3) -> tuple[HmmGraph, list[int]]:
    """Build a long chain of phones, each with *n_hmm_states* HMM states.

    Similar in structure to what ``compile_training_graph`` produces.
    Returns (graph, tid_to_pdf).
    """
    graph = HmmGraph()
    n_pdfs = n_phones * n_hmm_states
    tid_to_pdf = [0]  # index 0 unused

    prev_state = graph.add_state()
    graph.start = prev_state
    tid = 1

    for phone_idx in range(n_phones):
        for hmm_idx in range(n_hmm_states):
            cur_state = prev_state
            next_state = graph.add_state()
            pdf = phone_idx * n_hmm_states + hmm_idx

            # Self-loop
            tid_to_pdf.append(pdf)
            graph.add_arc(
                cur_state,
                HmmGraphArc(tid=tid, word_id=0, weight=0.3, nextstate=cur_state),
            )
            tid += 1

            # Forward
            tid_to_pdf.append(pdf)
            word_id = phone_idx + 1 if hmm_idx == 0 else 0
            graph.add_arc(
                cur_state,
                HmmGraphArc(tid=tid, word_id=word_id, weight=0.5, nextstate=next_state),
            )
            tid += 1
            prev_state = next_state

    # Mark last state as final
    graph.states[prev_state].is_final = True
    return graph, tid_to_pdf


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestK2DecoderCorrectness:
    """Verify the k2 decoder produces correct alignments."""

    def test_basic_alignment(self):
        """k2 should align all frames through a simple graph."""
        graph, tid_to_pdf = _make_simple_graph()
        num_frames = 6
        loglikes = np.zeros((num_frames, 3), dtype=np.float32)
        loglikes[:2, 0] = 1.0
        loglikes[2:4, 1] = 1.0
        loglikes[4:6, 2] = 1.0

        alignment, wids, cost, ok = viterbi_decode_k2(
            graph, loglikes, tid_to_pdf, acoustic_scale=1.0
        )
        assert ok
        assert len(alignment) == num_frames

    def test_word_ids_propagated(self):
        """Word-ids should be recovered from the best path."""
        graph, tid_to_pdf = _make_simple_graph()
        num_frames = 6
        loglikes = np.zeros((num_frames, 3), dtype=np.float32)
        loglikes[:2, 0] = 1.0
        loglikes[2:4, 1] = 1.0
        loglikes[4:6, 2] = 1.0

        alignment, wids, cost, ok = viterbi_decode_k2(
            graph, loglikes, tid_to_pdf, acoustic_scale=1.0
        )
        assert ok
        assert len(wids) > 0, "Should recover at least one word-id"
        assert 1 in wids or 2 in wids

    def test_epsilon_arcs_handled(self):
        """Graphs with epsilon arcs should work correctly."""
        graph = HmmGraph()
        s0 = graph.add_state()
        s1 = graph.add_state()
        s2 = graph.add_state()
        s_final = graph.add_state(is_final=True)
        graph.start = s0

        # s0 --eps--> s1 --tid=1--> s2 --tid=2--> final
        graph.add_arc(s0, HmmGraphArc(tid=0, word_id=0, weight=0.0, nextstate=s1))
        graph.add_arc(s1, HmmGraphArc(tid=1, word_id=1, weight=0.5, nextstate=s1))
        graph.add_arc(s1, HmmGraphArc(tid=2, word_id=0, weight=0.5, nextstate=s2))
        graph.add_arc(s2, HmmGraphArc(tid=3, word_id=0, weight=0.5, nextstate=s2))
        graph.add_arc(s2, HmmGraphArc(tid=4, word_id=0, weight=0.5, nextstate=s_final))

        tid_to_pdf = [0, 0, 0, 1, 1]
        loglikes = np.zeros((4, 2), dtype=np.float32)
        loglikes[:2, 0] = 1.0
        loglikes[2:, 1] = 1.0

        alignment, wids, cost, ok = viterbi_decode_k2(
            graph, loglikes, tid_to_pdf, acoustic_scale=1.0
        )
        assert ok
        assert len(alignment) == 4

    def test_chain_graph(self):
        """Chain of 20 phones × 3 HMM states should align correctly."""
        graph, tid_to_pdf = _make_chain_graph(20, 3)
        n_pdfs = 60
        num_frames = 120  # 2x the 60 states — enough for self-loops
        loglikes = np.random.randn(num_frames, n_pdfs).astype(np.float32)
        # Make the diagonal preferred so the alignment proceeds monotonically
        for i in range(num_frames):
            pdf_idx = min(int(i * n_pdfs / num_frames), n_pdfs - 1)
            loglikes[i, pdf_idx] += 5.0

        alignment, wids, cost, ok = viterbi_decode_k2(
            graph, loglikes, tid_to_pdf, acoustic_scale=1.0
        )
        assert ok
        assert len(alignment) == num_frames


# ---------------------------------------------------------------------------
# Speed comparison
# ---------------------------------------------------------------------------

class TestK2DecoderSpeed:
    """Benchmark k2 vs pure-Python Viterbi on various graph sizes."""

    @staticmethod
    def _bench(graph, loglikes, tid_to_pdf, acoustic_scale=1.0, n_warmup=1, n_iter=3):
        """Run k2 decoder, return (k2_time, k2_ok)."""
        for _ in range(n_warmup):
            viterbi_decode_k2(graph, loglikes, tid_to_pdf, acoustic_scale=acoustic_scale)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _, _, _, ok_k2 = viterbi_decode_k2(
                graph, loglikes, tid_to_pdf, acoustic_scale=acoustic_scale
            )
        k2_time = (time.perf_counter() - t0) / n_iter

        return k2_time, ok_k2

    def test_speed_small_graph(self):
        """Speed on a small graph (3 states, 6 frames)."""
        graph, tid_to_pdf = _make_simple_graph()
        loglikes = np.zeros((6, 3), dtype=np.float32)
        loglikes[:2, 0] = 1.0
        loglikes[2:4, 1] = 1.0
        loglikes[4:6, 2] = 1.0

        k2_t, ok_k2 = self._bench(graph, loglikes, tid_to_pdf)
        print(f"\n  Small graph (3 states, 6 frames):")
        print(f"    k2:     {k2_t*1000:8.2f} ms")
        assert ok_k2

    def test_speed_medium_graph(self):
        """Speed on a medium graph (~150 states, 300 frames)."""
        graph, tid_to_pdf = _make_chain_graph(50, 3)
        n_pdfs = 150
        num_frames = 300  # must be >= n_states for path to reach final
        loglikes = np.random.randn(num_frames, n_pdfs).astype(np.float32)
        for i in range(num_frames):
            pdf_idx = min(int(i * n_pdfs / num_frames), n_pdfs - 1)
            loglikes[i, pdf_idx] += 5.0

        k2_t, ok_k2 = self._bench(graph, loglikes, tid_to_pdf)
        print(f"\n  Medium graph ({graph.num_states} states, {num_frames} frames):")
        print(f"    k2:     {k2_t*1000:8.2f} ms")
        assert ok_k2

    def test_speed_large_graph(self):
        """Speed on a large graph (~750 states, 1500 frames).

        This approximates a long real alignment workload.
        The pure-Python decoder takes many seconds or hangs on this.
        """
        graph, tid_to_pdf = _make_chain_graph(250, 3)
        n_pdfs = 750
        num_frames = 1500  # must be >= n_states for path to reach final
        loglikes = np.random.randn(num_frames, n_pdfs).astype(np.float32)
        for i in range(num_frames):
            pdf_idx = min(int(i * n_pdfs / num_frames), n_pdfs - 1)
            loglikes[i, pdf_idx] += 5.0

        # Only run k2 (Python would take too long)
        t0 = time.perf_counter()
        alignment, _, cost, ok = viterbi_decode_k2(
            graph, loglikes, tid_to_pdf, acoustic_scale=1.0
        )
        k2_time = time.perf_counter() - t0

        print(f"\n  Large graph ({graph.num_states} states, {num_frames} frames):")
        print(f"    k2:     {k2_time*1000:8.2f} ms")
        print(f"    python: skipped (would take minutes)")
        assert ok
        assert len(alignment) == num_frames
        assert k2_time < 5.0, f"k2 should finish under 5s, took {k2_time:.2f}s"

    def test_speed_real_model(self, model_dir):
        """End-to-end speed: sk-align (k2) vs pykaldi subprocess.

        Measures the full alignment pipeline for both implementations:
        - sk-align:  MFCC → nnet3 (PyTorch) → graph compile → k2 Viterbi → word-align
        - pykaldi:   MFCC → nnet3 (Kaldi C++) → NnetAligner → word-align
        """
        ref_dir = REFERENCE_DIR / "m001rmb0"
        if not ref_dir.exists():
            pytest.skip("No reference data")

        metadata_path = ref_dir / "metadata.json"
        audio_path = ref_dir / "audio.npy"
        if not metadata_path.exists() or not audio_path.exists():
            pytest.skip("Incomplete reference data")

        with open(metadata_path) as f:
            meta = json.load(f)

        audio = np.load(audio_path)
        words = meta["words"]

        # ---- sk-align (k2) end-to-end ----
        from sk_align.aligner import Aligner
        from sk_align.nnet3_torch import TorchNnetScorer

        nnet_scorer = TorchNnetScorer.from_model_file(model_dir / "final.mdl")
        aligner = Aligner.from_model_dir(model_dir, nnet_scorer=nnet_scorer)

        # Warmup
        aligner.align(audio, words)

        t0 = time.perf_counter()
        n_iter = 3
        for _ in range(n_iter):
            sk_result = aligner.align(audio, words)
        sk_time = (time.perf_counter() - t0) / n_iter

        # ---- pykaldi subprocess ----
        pykaldi_time = None
        pykaldi_result = None
        pykaldi_skip_reason = None

        if not PYKALDI_PYTHON.exists():
            pykaldi_skip_reason = f"pykaldi python not found at {PYKALDI_PYTHON}"
        elif not PYKALDI_ALIGNER.exists():
            pykaldi_skip_reason = f"pykaldi aligner script not found at {PYKALDI_ALIGNER}"
        else:
            pykaldi_time, pykaldi_result = _run_pykaldi_timed(
                model_dir, audio, words, n_iter=n_iter,
            )

        # ---- Report ----
        print(f"\n  End-to-end alignment ({len(words)} words):")
        print(f"    sk-align (k2): {sk_time*1000:8.1f} ms")
        if pykaldi_time is not None:
            print(f"    pykaldi:       {pykaldi_time*1000:8.1f} ms")
            print(f"    ratio:         {pykaldi_time/sk_time:.1f}x (pykaldi / sk-align)")
        else:
            print(f"    pykaldi:       skipped ({pykaldi_skip_reason})")

        # Verify sk-align result
        assert len(sk_result) == len(words)

        # If pykaldi ran, compare word counts
        if pykaldi_result is not None:
            assert len(pykaldi_result) == len(words), (
                f"pykaldi returned {len(pykaldi_result)} words, expected {len(words)}"
            )

            # Print side-by-side timestamps
            print(f"\n  {'Word':20s}  {'sk-align':>10s}  {'pykaldi':>10s}  {'Δstart':>8s}")
            print(f"  {'─' * 20}  {'─' * 10}  {'─' * 10}  {'─' * 8}")
            for sk_w, pk_w in zip(sk_result, pykaldi_result):
                ds = sk_w["start"] - pk_w["start"]
                print(
                    f"  {sk_w['word']:20s}"
                    f"  {sk_w['start']:10.3f}"
                    f"  {pk_w['start']:10.3f}"
                    f"  {ds:+8.3f}"
                )


def _run_pykaldi_timed(
    model_dir: Path,
    audio: np.ndarray,
    words: list[str],
    n_iter: int = 3,
) -> tuple[float | None, list[dict] | None]:
    """Launch the pykaldi subprocess aligner and measure alignment time.

    Returns ``(avg_seconds_per_call, result_list)`` or ``(None, None)`` on
    failure.
    """
    # Write audio to a temp raw file (float32 PCM)
    tmpdir = tempfile.mkdtemp(prefix="sk_align_bench_")
    audio_file = os.path.join(tmpdir, "audio.raw")
    audio.astype(np.float32).tofile(audio_file)

    try:
        proc = subprocess.Popen(
            [str(PYKALDI_PYTHON), str(PYKALDI_ALIGNER), str(model_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for "ready" signal
        ready_line = proc.stdout.readline()
        ready = json.loads(ready_line)
        if not ready.get("ok"):
            proc.kill()
            return None, None

        # Warmup
        req = json.dumps({
            "audio_path": audio_file,
            "words": words,
            "offset": 0.0,
        })
        proc.stdin.write(req + "\n")
        proc.stdin.flush()
        warmup_resp = json.loads(proc.stdout.readline())
        if not warmup_resp.get("ok"):
            proc.kill()
            return None, None

        # Timed runs
        t0 = time.perf_counter()
        last_result = None
        for _ in range(n_iter):
            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            if not resp.get("ok"):
                proc.kill()
                return None, None
            last_result = resp["words"]
        elapsed = (time.perf_counter() - t0) / n_iter

        # Shut down
        proc.stdin.close()
        proc.wait(timeout=5)

        return elapsed, last_result
    except Exception as e:
        print(f"  [pykaldi error: {e}]")
        return None, None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
