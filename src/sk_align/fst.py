"""
Minimal OpenFst binary reader and lightweight FST representation.

Only supports ``StdVectorFst`` (tropical semiring, single-float weight)
which is what Kaldi uses for ``L.fst`` and decoding graphs.

Reference: OpenFst header format, kaldi/src/fstext/kaldi-fst-io.cc
"""

from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import numpy as np

# Tropical-semiring "zero" weight (= +infinity, meaning impossible arc)
WEIGHT_ZERO = float("inf")
# Tropical-semiring "one" weight (= 0.0, meaning free arc)
WEIGHT_ONE = 0.0


@dataclass
class Arc:
    """A single FST arc (transition)."""

    ilabel: int
    olabel: int
    weight: float
    nextstate: int


@dataclass
class State:
    """A single FST state."""

    final_weight: float  # WEIGHT_ZERO if non-final
    arcs: list[Arc] = field(default_factory=list)


@dataclass
class StdVectorFst:
    """A minimal in-memory FST (StdVectorFst equivalent).

    States are numbered 0..len(states)-1.
    """

    start: int
    states: list[State] = field(default_factory=list)

    @property
    def num_states(self) -> int:
        return len(self.states)

    def add_state(self) -> int:
        """Add a new state and return its id."""
        sid = len(self.states)
        self.states.append(State(final_weight=WEIGHT_ZERO))
        return sid

    def add_arc(self, src: int, arc: Arc) -> None:
        self.states[src].arcs.append(arc)

    def set_final(self, state: int, weight: float = WEIGHT_ONE) -> None:
        self.states[state].final_weight = weight

    def set_start(self, state: int) -> None:
        self.start = state

    def arcs_sorted_by_ilabel(self, state: int) -> list[Arc]:
        """Return arcs from *state* sorted by input label."""
        return sorted(self.states[state].arcs, key=lambda a: a.ilabel)

    # -----------------------------------------------------------------
    # I/O — Read OpenFst binary format
    # -----------------------------------------------------------------

    @classmethod
    def read_binary(cls, f: BinaryIO) -> StdVectorFst:
        """Read a standard OpenFst binary ``VectorFst<StdArc>``.

        This handles the OpenFst file header (magic number, fst_type, etc.).
        """
        # --- Header ---
        magic = struct.unpack("<i", f.read(4))[0]
        if magic != 2125659606:  # OpenFst magic
            raise ValueError(f"Not an OpenFst binary file (magic={magic:#x})")

        fst_type = _read_openfst_string(f)
        arc_type = _read_openfst_string(f)

        if fst_type != "vector":
            raise ValueError(f"Unsupported FST type: {fst_type!r}")
        if arc_type != "standard":
            raise ValueError(f"Unsupported arc type: {arc_type!r}")

        version = struct.unpack("<i", f.read(4))[0]
        flags = struct.unpack("<i", f.read(4))[0]
        properties = struct.unpack("<Q", f.read(8))[0]
        start = struct.unpack("<q", f.read(8))[0]
        numstates = struct.unpack("<q", f.read(8))[0]

        # OpenFst version >= 2 includes total arc count in header
        if version >= 2:
            _numarcs = struct.unpack("<q", f.read(8))[0]

        # The header may have symbol tables if flags indicate (bits 0, 1)
        has_isymbols = bool(flags & 1)
        has_osymbols = bool(flags & 2)
        if has_isymbols:
            _skip_symbol_table(f)
        if has_osymbols:
            _skip_symbol_table(f)

        # --- States and arcs ---
        fst = cls(start=start)
        for _ in range(numstates):
            # TropicalWeight is a single float; inf = non-final
            final_weight = struct.unpack("<f", f.read(4))[0]
            num_arcs = struct.unpack("<q", f.read(8))[0]

            arcs = []
            for _ in range(num_arcs):
                ilabel = struct.unpack("<i", f.read(4))[0]
                olabel = struct.unpack("<i", f.read(4))[0]
                weight = struct.unpack("<f", f.read(4))[0]
                nextstate = struct.unpack("<i", f.read(4))[0]
                arcs.append(Arc(ilabel, olabel, weight, nextstate))

            fst.states.append(State(final_weight=final_weight, arcs=arcs))

        return fst

    @classmethod
    def from_file(cls, path: str | Path) -> StdVectorFst:
        """Read from an OpenFst binary file (e.g. ``L.fst``)."""
        with open(path, "rb") as f:
            return cls.read_binary(f)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def compose_linear(self, word_ids: list[int]) -> StdVectorFst:
        """Compose this FST with a linear acceptor of *word_ids*.

        This is a simplified composition for forced alignment: the
        right-hand side is a simple chain of word labels.  Returns the
        composed FST (subset of paths matching the word sequence).

        The input FST is typically ``L.fst`` (phones → words) and the
        result maps phone sequences to the given word sequence.
        """
        # Build a minimal composed FST by state-pair expansion
        # (self_state, word_idx) → new_state
        result = StdVectorFst(start=-1)
        pair_to_state: dict[tuple[int, int], int] = {}
        queue: deque[tuple[int, int]] = deque()

        def get_state(fst_s: int, w_idx: int) -> int:
            key = (fst_s, w_idx)
            if key in pair_to_state:
                return pair_to_state[key]
            sid = result.add_state()
            pair_to_state[key] = sid
            queue.append(key)
            return sid

        start_state = get_state(self.start, 0)
        result.set_start(start_state)

        n_words = len(word_ids)

        while queue:
            fst_s, w_idx = queue.popleft()
            res_s = pair_to_state[(fst_s, w_idx)]

            # Check final
            if w_idx == n_words and self.states[fst_s].final_weight != WEIGHT_ZERO:
                result.set_final(res_s, self.states[fst_s].final_weight)

            for arc in self.states[fst_s].arcs:
                if arc.olabel == 0:
                    # Epsilon on output: advance FST state, keep word index
                    new_s = get_state(arc.nextstate, w_idx)
                    result.add_arc(
                        res_s,
                        Arc(arc.ilabel, 0, arc.weight, new_s),
                    )
                elif w_idx < n_words and arc.olabel == word_ids[w_idx]:
                    # Matching word label: advance both
                    new_s = get_state(arc.nextstate, w_idx + 1)
                    result.add_arc(
                        res_s,
                        Arc(arc.ilabel, arc.olabel, arc.weight, new_s),
                    )
                # else: non-matching output label → skip

        return result


def _read_openfst_string(f: BinaryIO) -> str:
    """Read a length-prefixed string (OpenFst format)."""
    length = struct.unpack("<i", f.read(4))[0]
    return f.read(length).decode("ascii")


def _skip_symbol_table(f: BinaryIO) -> None:
    """Skip over an embedded OpenFst symbol table."""
    # Symbol table binary: magic + name + ... + entries
    # Magic number for symbol table
    magic = struct.unpack("<q", f.read(8))[0]
    _name = _read_openfst_string(f)
    _available_key = struct.unpack("<q", f.read(8))[0]
    num_symbols = struct.unpack("<q", f.read(8))[0]
    for _ in range(num_symbols):
        _symbol = _read_openfst_string(f)
        _key = struct.unpack("<q", f.read(8))[0]


# ---------------------------------------------------------------------------
# Simple FST algorithms
# ---------------------------------------------------------------------------

def epsilon_closure(fst: StdVectorFst, state: int) -> list[tuple[int, float]]:
    """Compute epsilon closure of *state*.

    Returns list of ``(reachable_state, total_weight)`` pairs where weight
    is the sum of arc weights along epsilon paths (tropical semiring = min
    over paths, so we track the best weight).
    """
    best: dict[int, float] = {state: 0.0}
    queue = [state]
    while queue:
        s = queue.pop(0)
        for arc in fst.states[s].arcs:
            if arc.ilabel == 0:
                new_w = best[s] + arc.weight
                if arc.nextstate not in best or new_w < best[arc.nextstate]:
                    best[arc.nextstate] = new_w
                    queue.append(arc.nextstate)
    return list(best.items())
