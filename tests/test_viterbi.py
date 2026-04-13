"""
Tests for the Viterbi decoder and graph compilation.

Uses synthetic data (no model files needed) to verify the core algorithms.
"""

import numpy as np
import pytest

from sk_align.graph import HmmGraph, HmmGraphArc
from sk_align.viterbi import MatrixAcousticScorer, viterbi_decode


class TestViterbiSynthetic:
    """Viterbi decoding tests with hand-crafted graphs."""

    def _make_simple_graph(self) -> HmmGraph:
        """Create a minimal 3-state HMM (self-loop + forward per state).

        Graph: s0 --(tid=1)--> s1 --(tid=3)--> s2 --(tid=5)--> final
                |                |                |
                └──(tid=2)──┘    └──(tid=4)──┘    └──(tid=6)──┘
                  (self-loop)      (self-loop)      (self-loop)
        """
        graph = HmmGraph()
        s0 = graph.add_state()
        s1 = graph.add_state()
        s2 = graph.add_state()
        s_final = graph.add_state(is_final=True)
        graph.start = s0

        # State 0: forward (tid=1), self-loop (tid=2)
        graph.add_arc(s0, HmmGraphArc(tid=1, word_id=1, weight=0.5, nextstate=s1))
        graph.add_arc(s0, HmmGraphArc(tid=2, word_id=0, weight=0.3, nextstate=s0))

        # State 1: forward (tid=3), self-loop (tid=4)
        graph.add_arc(s1, HmmGraphArc(tid=3, word_id=0, weight=0.5, nextstate=s2))
        graph.add_arc(s1, HmmGraphArc(tid=4, word_id=0, weight=0.3, nextstate=s1))

        # State 2: forward (tid=5), self-loop (tid=6)
        graph.add_arc(s2, HmmGraphArc(tid=5, word_id=0, weight=0.5, nextstate=s_final))
        graph.add_arc(s2, HmmGraphArc(tid=6, word_id=0, weight=0.3, nextstate=s2))

        return graph

    def test_basic_alignment(self):
        """Viterbi should find a valid path through a simple graph."""
        graph = self._make_simple_graph()

        # 6 frames, 3 pdfs (each tid maps to: tid 1,2→pdf0, 3,4→pdf1, 5,6→pdf2)
        num_frames = 6
        num_pdfs = 3
        loglikes = np.zeros((num_frames, num_pdfs), dtype=np.float32)
        # Make each pdf preferred at different times
        loglikes[:2, 0] = 1.0  # frames 0-1: prefer pdf 0
        loglikes[2:4, 1] = 1.0  # frames 2-3: prefer pdf 1
        loglikes[4:6, 2] = 1.0  # frames 4-5: prefer pdf 2

        tid_to_pdf = [0, 0, 0, 1, 1, 2, 2]  # index 0 unused
        scorer = MatrixAcousticScorer(loglikes, tid_to_pdf, acoustic_scale=1.0)

        result = viterbi_decode(graph, scorer, beam=100.0)
        assert result.succeeded
        assert len(result.alignment) == num_frames

    def test_all_self_loops(self):
        """With only one state, should produce all self-loops then forward."""
        graph = HmmGraph()
        s0 = graph.add_state()
        s_final = graph.add_state(is_final=True)
        graph.start = s0

        graph.add_arc(s0, HmmGraphArc(tid=1, word_id=0, weight=0.0, nextstate=s_final))
        graph.add_arc(s0, HmmGraphArc(tid=2, word_id=0, weight=0.0, nextstate=s0))

        num_frames = 5
        loglikes = np.ones((num_frames, 1), dtype=np.float32)
        tid_to_pdf = [0, 0, 0]
        scorer = MatrixAcousticScorer(loglikes, tid_to_pdf, acoustic_scale=1.0)

        result = viterbi_decode(graph, scorer, beam=100.0)
        assert result.succeeded
        assert len(result.alignment) == num_frames

    def test_empty_frames(self):
        """Zero frames should return empty alignment."""
        graph = self._make_simple_graph()
        loglikes = np.zeros((0, 3), dtype=np.float32)
        tid_to_pdf = [0, 0, 0, 1, 1, 2, 2]
        scorer = MatrixAcousticScorer(loglikes, tid_to_pdf, acoustic_scale=1.0)

        result = viterbi_decode(graph, scorer, beam=100.0)
        assert len(result.alignment) == 0

    def test_word_ids_in_result(self):
        """Word IDs from graph arcs should appear in result."""
        graph = self._make_simple_graph()

        num_frames = 6
        loglikes = np.zeros((num_frames, 3), dtype=np.float32)
        loglikes[:2, 0] = 1.0
        loglikes[2:4, 1] = 1.0
        loglikes[4:6, 2] = 1.0

        tid_to_pdf = [0, 0, 0, 1, 1, 2, 2]
        scorer = MatrixAcousticScorer(loglikes, tid_to_pdf, acoustic_scale=1.0)

        result = viterbi_decode(graph, scorer, beam=100.0)
        assert result.succeeded
        # The graph has word_id=1 on the first forward arc
        assert 1 in result.word_ids


class TestGraphCompilation:
    """Tests for graph compilation (requires model files)."""

    def test_compile_single_word(self, model_dir):
        """Compile a graph for a single word."""
        from sk_align.fst import StdVectorFst
        from sk_align.graph import compile_training_graph
        from sk_align.kaldi_io import read_disambig_symbols, read_symbol_table
        from sk_align.transition_model import TransitionModel
        from sk_align.tree import ContextDependency

        tm = TransitionModel.from_file(model_dir / "final.mdl")
        tree = ContextDependency.from_file(model_dir / "tree")
        lexicon = StdVectorFst.from_file(model_dir / "L.fst")
        words = read_symbol_table(model_dir / "words.txt")
        disambig = read_disambig_symbols(model_dir / "disambig.int")

        # Find a valid word
        test_word = None
        for w, wid in words.items():
            if w not in ("<eps>", "#0", "<unk>", "<s>", "</s>") and wid > 0:
                test_word = w
                break

        if test_word is None:
            pytest.skip("No suitable test word")

        wid = words[test_word]
        graph = compile_training_graph([wid], lexicon, tree, tm, disambig)

        assert graph.num_states > 0
        # Should have at least one final state
        has_final = any(s.is_final for s in graph.states)
        assert has_final, "Graph should have a final state"
