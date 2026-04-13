"""
Tests for Kaldi I/O readers: TransitionModel, ContextDependency, FST.

These tests download the actual model files from HuggingFace.
"""

import pytest
import numpy as np

from sk_align.kaldi_io import (
    read_disambig_symbols,
    read_symbol_table,
    read_symbol_table_reverse,
    read_word_boundary,
)
from sk_align.transition_model import TransitionModel
from sk_align.tree import ContextDependency
from sk_align.fst import StdVectorFst


class TestTextFileReaders:
    """Tests for simple text-format Kaldi files."""

    def test_read_symbol_table(self, model_dir):
        """Read words.txt and verify basic structure."""
        table = read_symbol_table(model_dir / "words.txt")
        assert len(table) > 0
        assert "<eps>" in table
        assert table["<eps>"] == 0
        # Should have an <unk> symbol
        assert "<unk>" in table or "#0" in table

    def test_read_symbol_table_reverse(self, model_dir):
        """Reverse symbol table should map id → symbol."""
        table = read_symbol_table_reverse(model_dir / "words.txt")
        assert 0 in table
        assert table[0] == "<eps>"

    def test_read_disambig_symbols(self, model_dir):
        """Read disambig.int and verify it contains integers."""
        syms = read_disambig_symbols(model_dir / "disambig.int")
        assert len(syms) > 0
        assert all(isinstance(s, int) for s in syms)

    def test_read_word_boundary(self, model_dir):
        """Read word_boundary.int and verify valid types."""
        wb = read_word_boundary(model_dir / "word_boundary.int")
        assert len(wb) > 0
        valid_types = {"nonword", "begin", "end", "singleton", "internal"}
        for phone_id, btype in wb.items():
            assert btype in valid_types, f"Invalid type {btype!r} for phone {phone_id}"


class TestTransitionModel:
    """Tests for TransitionModel loading."""

    def test_load_transition_model(self, model_dir):
        """Load TransitionModel from final.mdl."""
        tm = TransitionModel.from_file(model_dir / "final.mdl")
        assert tm.num_transition_ids > 0
        assert tm.num_pdfs > 0
        assert len(tm.tuples) > 0
        assert len(tm.log_probs) > 0

    def test_transition_id_mappings(self, model_dir):
        """Verify transition-id → pdf and phone mappings are valid."""
        tm = TransitionModel.from_file(model_dir / "final.mdl")
        for tid in range(1, min(100, tm.num_transition_ids + 1)):
            pdf = tm.transition_id_to_pdf(tid)
            phone = tm.transition_id_to_phone(tid)
            assert 0 <= pdf < tm.num_pdfs
            assert phone > 0  # phones are 1-based
            # is_self_loop should be bool
            assert isinstance(tm.is_self_loop(tid), bool)

    def test_topology_structure(self, model_dir):
        """HMM topology should have at least one entry."""
        tm = TransitionModel.from_file(model_dir / "final.mdl")
        assert len(tm.topology.entries) > 0
        assert len(tm.topology.phones) > 0
        # Each entry should have HMM states with transitions
        for entry in tm.topology.entries:
            assert len(entry.states) > 0
            for state in entry.states:
                if state.forward_pdf_class >= 0:
                    assert len(state.transitions) > 0


class TestContextDependency:
    """Tests for decision tree loading."""

    def test_load_tree(self, model_dir):
        """Load ContextDependency from tree file."""
        tree = ContextDependency.from_file(model_dir / "tree")
        assert tree.N > 0  # context width
        assert 0 <= tree.P < tree.N  # central position
        assert tree.to_pdf is not None

    def test_tree_lookup(self, model_dir):
        """Tree should return valid pdf-ids for known phones."""
        tree = ContextDependency.from_file(model_dir / "tree")
        tm = TransitionModel.from_file(model_dir / "final.mdl")

        # Pick a few phones from the topology
        phones = tm.topology.phones[:5]
        for phone in phones:
            context = [0] * tree.P + [phone] + [0] * (tree.N - tree.P - 1)
            for pdf_class in range(3):  # typical 3-state HMM
                result = tree.compute_pdf_id(context, pdf_class)
                # Result can be None for some contexts, but should mostly work
                if result is not None:
                    assert 0 <= result < tm.num_pdfs


class TestFstReader:
    """Tests for OpenFst binary reader."""

    def test_read_lexicon_fst(self, model_dir):
        """Read L.fst and verify basic structure."""
        fst = StdVectorFst.from_file(model_dir / "L.fst")
        assert fst.num_states > 0
        assert fst.start >= 0
        # Should have arcs
        total_arcs = sum(len(s.arcs) for s in fst.states)
        assert total_arcs > 0

    def test_lexicon_has_words(self, model_dir):
        """Lexicon FST should have output labels matching words.txt."""
        fst = StdVectorFst.from_file(model_dir / "L.fst")
        words = read_symbol_table(model_dir / "words.txt")
        # Collect all output labels from the FST
        olabels = set()
        for state in fst.states:
            for arc in state.arcs:
                if arc.olabel != 0:
                    olabels.add(arc.olabel)
        # Should overlap with word IDs
        word_ids = set(words.values())
        assert len(olabels & word_ids) > 0

    def test_compose_linear_simple(self, model_dir):
        """Compose L.fst with a single-word sequence."""
        fst = StdVectorFst.from_file(model_dir / "L.fst")
        words = read_symbol_table(model_dir / "words.txt")

        # Find a word that's in the vocabulary
        test_word = None
        for w, wid in words.items():
            if w not in ("<eps>", "#0", "<unk>", "<s>", "</s>") and wid > 0:
                test_word = w
                break

        if test_word is None:
            pytest.skip("No suitable test word found")

        wid = words[test_word]
        result = fst.compose_linear([wid])
        # Should have at least one state
        assert result.num_states > 0
        # Should have at least one final state with phone arcs
        has_final = any(
            s.final_weight != float("inf") for s in result.states
        )
        assert has_final, f"No final state in composed FST for word '{test_word}'"
