"""
Reader for Kaldi's ``ContextDependency`` (phonetic decision tree).

The tree maps (phone-in-context, pdf-class) → pdf-id.  It is used during
graph compilation to determine which neural network output (pdf) corresponds
to a given phone in a given context.

Reference: kaldi/src/tree/context-dep.cc, event-map.cc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from sk_align.kaldi_io import (
    expect_token,
    peek_char,
    read_binary_header,
    read_int32,
    read_integer_vector,
    read_token,
)


# ---------------------------------------------------------------------------
# Event Map nodes (recursive tree structure)
# ---------------------------------------------------------------------------

# An "event" is a dict {key: value} — e.g. {0: left_phone, 1: center_phone,
# 2: right_phone, -1: pdf_class}.  The tree maps events to pdf-ids.

EventType = dict[int, int]


class EventMap:
    """Base for all event-map node types."""

    def map(self, event: EventType) -> int | None:
        """Map an event to a pdf-id (or None if not found)."""
        raise NotImplementedError


@dataclass
class ConstantEventMap(EventMap):
    """Leaf node — always returns a constant pdf-id."""

    answer: int

    def map(self, event: EventType) -> int:
        return self.answer


@dataclass
class TableEventMap(EventMap):
    """Lookup-table node — selects child based on event[key]."""

    key: int
    table: list[EventMap | None]

    def map(self, event: EventType) -> int | None:
        val = event.get(self.key)
        if val is None or val < 0 or val >= len(self.table):
            return None
        child = self.table[val]
        if child is None:
            return None
        return child.map(event)


@dataclass
class SplitEventMap(EventMap):
    """Binary-decision node — tests if event[key] is in yes_set."""

    key: int
    yes_set: set[int]
    yes_child: EventMap
    no_child: EventMap

    def map(self, event: EventType) -> int | None:
        val = event.get(self.key)
        if val is not None and val in self.yes_set:
            return self.yes_child.map(event)
        else:
            return self.no_child.map(event)


def _read_event_map(f: BinaryIO) -> EventMap | None:
    """Recursively read an EventMap from a Kaldi binary stream."""
    c = peek_char(f)

    if c == "N":
        expect_token(f, "NULL")
        return None
    elif c == "C":
        expect_token(f, "CE")
        answer = read_int32(f)
        return ConstantEventMap(answer)
    elif c == "T":
        expect_token(f, "TE")
        key = read_int32(f)
        size = read_int32(f)
        # size is written as uint32 via WriteBasicType but with negative sign byte
        # Actually in the CLIF, it's uint32.  Our read_int32 handles both.
        expect_token(f, "(")
        table: list[EventMap | None] = []
        for _ in range(size):
            child = _read_event_map(f)
            table.append(child)
        expect_token(f, ")")
        return TableEventMap(key, table)
    elif c == "S":
        expect_token(f, "SE")
        key = read_int32(f)
        yes_list = read_integer_vector(f)
        yes_set = set(yes_list)
        expect_token(f, "{")
        yes_child = _read_event_map(f)
        no_child = _read_event_map(f)
        expect_token(f, "}")
        return SplitEventMap(key, yes_set, yes_child, no_child)
    else:
        raise ValueError(f"Unknown EventMap type character: {c!r}")


# ---------------------------------------------------------------------------
# Context Dependency
# ---------------------------------------------------------------------------

@dataclass
class ContextDependency:
    """Kaldi ``ContextDependency`` — phonetic decision tree.

    Parameters
    ----------
    N : int
        Context width (e.g. 3 for triphone).
    P : int
        Central phone position in context window (e.g. 1).
    to_pdf : EventMap
        Root of the decision tree.
    """

    N: int
    P: int
    to_pdf: EventMap

    def compute_pdf_id(
        self,
        phone_context: list[int],
        pdf_class: int,
    ) -> int | None:
        """Look up the pdf-id for a phone in context.

        Parameters
        ----------
        phone_context : list[int]
            Context window of phone IDs, length *N*.
            E.g. for triphone: [left_phone, center_phone, right_phone].
        pdf_class : int
            The HMM state's pdf-class (typically 0, 1, or 2).

        Returns
        -------
        int or None
            The pdf-id, or None if the tree has no mapping.
        """
        if len(phone_context) != self.N:
            raise ValueError(
                f"Context window length {len(phone_context)} != N={self.N}"
            )
        # Build the event: keys 0..N-1 are context phones, key -1 is pdf_class
        event: EventType = {i: phone_context[i] for i in range(self.N)}
        event[-1] = pdf_class
        return self.to_pdf.map(event)

    # -----------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------

    @staticmethod
    def read(f: BinaryIO) -> ContextDependency:
        """Read from a Kaldi binary stream (after ``\\0B`` header)."""
        expect_token(f, "ContextDependency")
        N = read_int32(f)
        P = read_int32(f)
        expect_token(f, "ToPdf")
        to_pdf = _read_event_map(f)
        if to_pdf is None:
            raise ValueError("ContextDependency tree root is NULL")
        expect_token(f, "EndContextDependency")
        return ContextDependency(N, P, to_pdf)

    @classmethod
    def from_file(cls, path: str | Path) -> ContextDependency:
        """Read from a Kaldi ``tree`` file."""
        with open(path, "rb") as f:
            read_binary_header(f)
            return cls.read(f)
