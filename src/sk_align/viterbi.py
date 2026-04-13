"""
Viterbi decoding over the HMM decoding graph.

Reimplements Kaldi's ``FasterDecoder`` for forced alignment:
token-passing Viterbi search with beam pruning.

For forced alignment, the graph is heavily constrained (one transcript),
so the beam can be very wide and pruning is minimal.

Reference: kaldi/src/decoder/faster-decoder.cc
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sk_align.graph import HmmGraph, HmmGraphArc


class AcousticScorer(Protocol):
    """Interface for acoustic scoring (log-likelihood computation).

    Matches the ``DecodableInterface`` concept from Kaldi.
    """

    def log_likelihood(self, frame: int, tid: int) -> float:
        """Return acoustic log-likelihood for *frame* and transition-id *tid*."""
        ...

    @property
    def num_frames(self) -> int:
        ...


class MatrixAcousticScorer:
    """Score from a pre-computed log-likelihood matrix.

    Parameters
    ----------
    loglikes : np.ndarray
        Shape ``(num_frames, num_pdfs)`` — log-likelihood for each frame/pdf.
    tid_to_pdf : list[int]
        Mapping from transition-id → pdf-id (from TransitionModel.id2pdf).
    acoustic_scale : float
        Applied to log-likelihoods before use.
    """

    def __init__(
        self,
        loglikes: np.ndarray,
        tid_to_pdf: list[int],
        acoustic_scale: float = 0.1,
    ):
        self._loglikes = loglikes
        self._tid_to_pdf = tid_to_pdf
        self._acoustic_scale = acoustic_scale

    def log_likelihood(self, frame: int, tid: int) -> float:
        pdf = self._tid_to_pdf[tid]
        return float(self._acoustic_scale * self._loglikes[frame, pdf])

    @property
    def num_frames(self) -> int:
        return self._loglikes.shape[0]


@dataclass
class Token:
    """Viterbi token for traceback."""

    cost: float
    arc: HmmGraphArc | None
    prev: Token | None
    frame: int  # frame at which this token was created (-1 for initial)


@dataclass
class AlignmentResult:
    """Result of Viterbi decoding."""

    alignment: list[int]         # transition-id per frame
    best_cost: float             # total path cost
    word_ids: list[int]          # word-ids encountered along the path
    succeeded: bool


def viterbi_decode(
    graph: HmmGraph,
    scorer: AcousticScorer,
    beam: float = 200.0,
) -> AlignmentResult:
    """Run Viterbi beam search over *graph* using *scorer*.

    Parameters
    ----------
    graph : HmmGraph
        Per-utterance decoding graph (from ``compile_training_graph``).
    scorer : AcousticScorer
        Acoustic log-likelihoods.
    beam : float
        Beam width for pruning (larger = wider search).

    Returns
    -------
    AlignmentResult
        Frame-level transition-id alignment and total cost.
    """
    num_frames = scorer.num_frames
    if num_frames == 0:
        return AlignmentResult([], 0.0, [], False)

    # --- Initialization ---
    # Active tokens: state → Token
    cur_toks: dict[int, Token] = {}

    # Start token
    start_token = Token(cost=0.0, arc=None, prev=None, frame=-1)
    cur_toks[graph.start] = start_token

    # Process initial epsilon (non-emitting) arcs
    cur_toks = _process_nonemitting(graph, cur_toks, beam)

    # --- Frame-by-frame decoding ---
    for frame in range(num_frames):
        next_toks: dict[int, Token] = {}
        best_cost = float("inf")

        for state, token in cur_toks.items():
            for arc in graph.states[state].arcs:
                if arc.tid == 0:
                    continue  # skip epsilon arcs in emitting pass

                # Acoustic cost (negated log-likelihood)
                ac_cost = -scorer.log_likelihood(frame, arc.tid)
                new_cost = token.cost + arc.weight + ac_cost

                if new_cost < best_cost + beam:
                    new_token = Token(
                        cost=new_cost, arc=arc, prev=token, frame=frame
                    )
                    if (
                        arc.nextstate not in next_toks
                        or new_cost < next_toks[arc.nextstate].cost
                    ):
                        next_toks[arc.nextstate] = new_token
                        if new_cost < best_cost:
                            best_cost = new_cost

        if not next_toks:
            # No surviving tokens — alignment failed
            return AlignmentResult([], float("inf"), [], False)

        # Prune
        cutoff = best_cost + beam
        cur_toks = {s: t for s, t in next_toks.items() if t.cost < cutoff}

        # Process non-emitting arcs
        cur_toks = _process_nonemitting(graph, cur_toks, beam)

    # --- Find best final token ---
    best_token = None
    best_cost = float("inf")

    for state, token in cur_toks.items():
        if graph.states[state].is_final and token.cost < best_cost:
            best_cost = token.cost
            best_token = token

    # If no final state reached, take the best token regardless
    if best_token is None:
        for token in cur_toks.values():
            if token.cost < best_cost:
                best_cost = token.cost
                best_token = token

    if best_token is None:
        return AlignmentResult([], float("inf"), [], False)

    # --- Traceback ---
    alignment: list[int] = []
    word_ids: list[int] = []
    token = best_token
    while token is not None and token.arc is not None:
        if token.arc.tid != 0:  # skip epsilon arcs
            alignment.append(token.arc.tid)
            if token.arc.word_id != 0:
                word_ids.append(token.arc.word_id)
        token = token.prev

    alignment.reverse()
    word_ids.reverse()

    return AlignmentResult(
        alignment=alignment,
        best_cost=best_cost,
        word_ids=word_ids,
        succeeded=True,
    )


def _process_nonemitting(
    graph: HmmGraph,
    toks: dict[int, Token],
    beam: float,
) -> dict[int, Token]:
    """Expand epsilon (non-emitting) arcs."""
    changed = True
    while changed:
        changed = False
        best_cost = min(t.cost for t in toks.values()) if toks else float("inf")
        cutoff = best_cost + beam

        new_toks = dict(toks)
        for state, token in list(toks.items()):
            if token.cost > cutoff:
                continue
            for arc in graph.states[state].arcs:
                if arc.tid != 0:
                    continue  # only epsilon arcs
                new_cost = token.cost + arc.weight
                if new_cost < cutoff:
                    new_token = Token(
                        cost=new_cost, arc=arc, prev=token, frame=token.frame
                    )
                    if (
                        arc.nextstate not in new_toks
                        or new_cost < new_toks[arc.nextstate].cost
                    ):
                        new_toks[arc.nextstate] = new_token
                        changed = True
        toks = new_toks

    return toks
