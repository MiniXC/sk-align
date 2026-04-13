"""
Reader for Kaldi's ``TransitionModel`` and ``HmmTopology`` binary formats.

These are needed to:
- Map transition-ids to pdf-ids (for acoustic scoring)
- Map transition-ids to phones (for word boundary alignment)
- Know self-loop vs forward transitions (for HMM topology)

Reference: kaldi/src/hmm/transition-model.cc, hmm-topology.cc
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

from sk_align.kaldi_io import (
    expect_token,
    read_bool,
    read_float32,
    read_float_vector,
    read_int32,
    read_integer_vector,
    read_token,
)


# ---------------------------------------------------------------------------
# HMM Topology
# ---------------------------------------------------------------------------

@dataclass
class HmmState:
    """One state in a phone's HMM."""

    forward_pdf_class: int
    self_loop_pdf_class: int
    transitions: list[tuple[int, float]]  # (dest_state, probability)


@dataclass
class TopologyEntry:
    """HMM topology for a group of phones."""

    states: list[HmmState]


@dataclass
class HmmTopology:
    """Kaldi ``HmmTopology`` — phone-level HMM structures."""

    phones: list[int]
    phone2idx: list[int]
    entries: list[TopologyEntry]

    def topology_for_phone(self, phone: int) -> TopologyEntry:
        """Return the topology entry for *phone*."""
        idx = self.phone2idx[phone]
        return self.entries[idx]

    @staticmethod
    def read(f: BinaryIO) -> HmmTopology:
        expect_token(f, "<Topology>")
        phones = read_integer_vector(f)
        phone2idx = read_integer_vector(f)

        # Peek for non-HMM flag (int32 value -1)
        pos = f.tell()
        prefix = f.read(1)
        is_hmm = True
        if prefix == b"\x04":  # looks like an int32 prefix byte
            val_bytes = f.read(4)
            val = int.from_bytes(val_bytes, "little", signed=True)
            if val == -1:
                is_hmm = False
            else:
                # Not the -1 sentinel; rewind
                f.seek(pos)
        else:
            f.seek(pos)

        num_entries = read_int32(f)
        entries = []
        for _ in range(num_entries):
            num_states = read_int32(f)
            states = []
            for _ in range(num_states):
                forward_pdf_class = read_int32(f)
                if is_hmm:
                    self_loop_pdf_class = forward_pdf_class
                else:
                    self_loop_pdf_class = read_int32(f)
                num_transitions = read_int32(f)
                transitions = []
                for _ in range(num_transitions):
                    dst = read_int32(f)
                    prob = read_float32(f)
                    transitions.append((dst, prob))
                states.append(HmmState(forward_pdf_class, self_loop_pdf_class, transitions))
            entries.append(TopologyEntry(states))

        expect_token(f, "</Topology>")
        return HmmTopology(phones, phone2idx, entries)


# ---------------------------------------------------------------------------
# Transition Model
# ---------------------------------------------------------------------------

@dataclass
class TransitionTuple:
    """One entry in the TransitionModel's tuple array."""

    phone: int
    hmm_state: int
    forward_pdf: int
    self_loop_pdf: int


@dataclass
class TransitionModel:
    """Kaldi ``TransitionModel`` — maps transition-ids to phones and pdf-ids.

    Transition IDs are 1-based.  Index 0 is unused.

    Key lookup arrays (1-indexed, matching Kaldi):
    - ``id2pdf[tid]`` → pdf-id  (0-based)
    - ``id2phone[tid]`` → phone  (1-based)
    - ``id2is_self_loop[tid]`` → bool
    - ``log_probs[tid]`` → log transition probability
    """

    topology: HmmTopology
    tuples: list[TransitionTuple]
    log_probs: np.ndarray  # shape (num_transition_ids + 1,), index 0 unused

    # Derived lookup tables (built by _build_lookup_tables)
    _state2id: list[int] = field(default_factory=list, repr=False)
    _id2state: list[int] = field(default_factory=list, repr=False)
    id2pdf: list[int] = field(default_factory=list, repr=False)
    id2phone: list[int] = field(default_factory=list, repr=False)
    id2is_self_loop: list[bool] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._build_lookup_tables()

    @property
    def num_transition_ids(self) -> int:
        return len(self.id2pdf) - 1  # index 0 is unused

    @property
    def num_pdfs(self) -> int:
        return max(self.id2pdf) + 1 if self.id2pdf else 0

    def transition_id_to_pdf(self, tid: int) -> int:
        return self.id2pdf[tid]

    def transition_id_to_phone(self, tid: int) -> int:
        return self.id2phone[tid]

    def is_self_loop(self, tid: int) -> bool:
        return self.id2is_self_loop[tid]

    def get_transition_log_prob(self, tid: int) -> float:
        return float(self.log_probs[tid])

    def _build_lookup_tables(self) -> None:
        """Build id2pdf, id2phone, id2is_self_loop from tuples and topology."""
        # state2id: for each transition-state (1-based), first transition-id
        # Transition-states are 1-indexed; state s has tuples[s-1]
        state2id = [0]  # index 0 unused
        tid = 1
        for tup in self.tuples:
            state2id.append(tid)
            # Number of transitions for this state = topology transitions
            entry = self.topology.topology_for_phone(tup.phone)
            hmm_state = entry.states[tup.hmm_state]
            tid += len(hmm_state.transitions)
        state2id.append(tid)  # sentinel
        self._state2id = state2id

        total_tids = tid  # one past last

        # id2state: for each transition-id, which transition-state?
        id2state = [0] * total_tids
        for s in range(1, len(self.tuples) + 1):
            for t in range(self._state2id[s], self._state2id[s + 1]):
                id2state[t] = s
        self._id2state = id2state

        # id2pdf, id2phone, id2is_self_loop
        id2pdf = [0] * total_tids
        id2phone = [0] * total_tids
        id2is_self_loop = [False] * total_tids

        for s in range(1, len(self.tuples) + 1):
            tup = self.tuples[s - 1]
            entry = self.topology.topology_for_phone(tup.phone)
            hmm_state = entry.states[tup.hmm_state]

            first_tid = self._state2id[s]
            for trans_idx, (dst, _prob) in enumerate(hmm_state.transitions):
                current_tid = first_tid + trans_idx
                if current_tid >= total_tids:
                    break

                # Self-loop = transition to same HMM state
                is_self = (dst == tup.hmm_state)
                id2is_self_loop[current_tid] = is_self

                if is_self:
                    id2pdf[current_tid] = tup.self_loop_pdf
                else:
                    id2pdf[current_tid] = tup.forward_pdf

                id2phone[current_tid] = tup.phone

        self.id2pdf = id2pdf
        self.id2phone = id2phone
        self.id2is_self_loop = id2is_self_loop

    # -----------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------

    @staticmethod
    def read(f: BinaryIO) -> TransitionModel:
        """Read a ``TransitionModel`` from a Kaldi binary stream.

        The stream should be positioned just after the ``\\0B`` header
        (i.e. the next token is ``<TransitionModel>``).
        """
        expect_token(f, "<TransitionModel>")
        topo = HmmTopology.read(f)

        tag = read_token(f)
        use_tuples = tag == "<Tuples>"
        if tag not in ("<Triples>", "<Tuples>"):
            raise ValueError(f"Expected <Triples> or <Tuples>, got {tag!r}")

        num = read_int32(f)
        tuples = []
        for _ in range(num):
            phone = read_int32(f)
            hmm_state = read_int32(f)
            forward_pdf = read_int32(f)
            if use_tuples:
                self_loop_pdf = read_int32(f)
            else:
                self_loop_pdf = forward_pdf
            tuples.append(TransitionTuple(phone, hmm_state, forward_pdf, self_loop_pdf))

        end_tag = "</Tuples>" if use_tuples else "</Triples>"
        expect_token(f, end_tag)

        expect_token(f, "<LogProbs>")
        log_probs = read_float_vector(f)
        expect_token(f, "</LogProbs>")
        expect_token(f, "</TransitionModel>")

        return TransitionModel(topo, tuples, log_probs)

    @classmethod
    def from_file(cls, path: str | Path) -> TransitionModel:
        """Read a TransitionModel from a Kaldi model file (``final.mdl``).

        Only reads the TransitionModel portion (skips the acoustic model).
        """
        with open(path, "rb") as f:
            from sk_align.kaldi_io import read_binary_header
            read_binary_header(f)
            return cls.read(f)
