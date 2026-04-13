"""
Per-utterance decoding graph compilation for forced alignment.

Given a word sequence, builds a constrained HCLG-like graph that the Viterbi
decoder searches.  For forced alignment the graph is small (one sentence)
compared to general ASR decoding.

The pipeline mirrors Kaldi's ``TrainingGraphCompiler::CompileGraphFromText``:

1. Linear word acceptor (G)
2. Compose with lexicon FST (L ∘ G)  → phone sequences
3. Expand context dependency (C)     → triphone → pdf-id via decision tree
4. Add HMM structure (H)            → transition-id sequences
5. Add self-loops and transition probs

For forced alignment, we simplify by directly traversing L∘G and building
the HMM state graph, avoiding full WFST determinization/minimization.

Reference: kaldi/src/decoder/training-graph-compiler.cc
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sk_align.fst import Arc, StdVectorFst, WEIGHT_ONE, WEIGHT_ZERO
from sk_align.transition_model import TransitionModel
from sk_align.tree import ContextDependency


@dataclass
class HmmGraphArc:
    """Arc in the decoding graph.

    ``tid`` is a Kaldi transition-id (1-based); 0 means epsilon (non-emitting).
    ``word_id`` is non-zero only at word boundaries (for traceback).
    """

    tid: int       # transition-id (input label); 0 = epsilon
    word_id: int   # output label (word id at boundary); 0 = no word
    weight: float  # graph weight (transition log-prob, negated for tropical)
    nextstate: int


@dataclass
class HmmGraphState:
    arcs: list[HmmGraphArc] = field(default_factory=list)
    is_final: bool = False


@dataclass
class HmmGraph:
    """Flat decoding graph: states with arcs labelled by transition-ids.

    This is the analogue of Kaldi's HCLG for a single utterance.
    """

    start: int = 0
    states: list[HmmGraphState] = field(default_factory=list)

    def add_state(self, is_final: bool = False) -> int:
        sid = len(self.states)
        self.states.append(HmmGraphState(is_final=is_final))
        return sid

    def add_arc(self, src: int, arc: HmmGraphArc) -> None:
        self.states[src].arcs.append(arc)

    @property
    def num_states(self) -> int:
        return len(self.states)


def compile_training_graph(
    word_ids: list[int],
    lexicon_fst: StdVectorFst,
    tree: ContextDependency,
    trans_model: TransitionModel,
    disambig_syms: list[int] | None = None,
) -> HmmGraph:
    """Compile a per-utterance HCLG-like decoding graph for forced alignment.

    Parameters
    ----------
    word_ids : list[int]
        Word-ID sequence for the transcript (from ``words.txt``).
    lexicon_fst : StdVectorFst
        The lexicon FST (``L.fst``).
    tree : ContextDependency
        The phonetic decision tree.
    trans_model : TransitionModel
        The transition model (from ``final.mdl``).
    disambig_syms : list[int] or None
        Disambiguation phone symbols to ignore.

    Returns
    -------
    HmmGraph
        A flat decoding graph suitable for Viterbi decoding.
    """
    disambig_set = set(disambig_syms) if disambig_syms else set()

    # Step 1-2: Compose L with linear word acceptor
    # This gives us all phone-sequence paths that produce the word sequence
    lg_fst = lexicon_fst.compose_linear(word_ids)

    # Step 3-4: Enumerate paths through L∘G and expand into HMM states
    # We do a breadth-first expansion that resolves context dependency
    # and builds HMM state sequences.
    #
    # For simplicity, we enumerate phone sequences from L∘G, expand each
    # into context-dependent HMM states, and combine into the final graph.

    phone_paths = _enumerate_phone_paths(lg_fst, disambig_set)
    if not phone_paths:
        raise ValueError("No valid phone paths found for the given transcript")

    # When there are many paths (e.g. <unk> with many pronunciations), the
    # graph explodes.  Limit to a manageable number of best-weight paths.
    MAX_PATHS = 10
    if len(phone_paths) > MAX_PATHS:
        phone_paths.sort(key=lambda p: p[2])  # sort by weight (lower = better)
        phone_paths = phone_paths[:MAX_PATHS]

    graph = HmmGraph()

    if len(phone_paths) == 1:
        # Single path — most common for forced alignment with unambiguous lexicon
        phones, word_boundaries, path_weight = phone_paths[0]
        _build_hmm_chain(graph, phones, word_boundaries, path_weight, tree, trans_model)
    else:
        # Multiple paths — build with a shared start and end
        graph_start = graph.add_state()
        graph.start = graph_start
        graph_end = graph.add_state(is_final=True)

        for phones, word_boundaries, path_weight in phone_paths:
            # Each path gets its own HMM chain
            path_start, path_end = _build_hmm_chain_segment(
                graph, phones, word_boundaries, tree, trans_model
            )
            # Connect start → path_start (epsilon, carry path weight)
            graph.add_arc(
                graph_start,
                HmmGraphArc(tid=0, word_id=0, weight=path_weight, nextstate=path_start),
            )
            # Connect path_end → graph_end (epsilon)
            graph.add_arc(
                path_end,
                HmmGraphArc(tid=0, word_id=0, weight=0.0, nextstate=graph_end),
            )

    # Post-process: merge SIL self-loops into following phone states
    # (mimics Kaldi's determinized graph where SIL and word-initial phones
    #  share the same state, allowing the decoder to stay in SIL before
    #  transitioning to the next word).
    _merge_sil_into_successors(graph, trans_model)

    return graph


def _merge_sil_into_successors(
    graph: HmmGraph,
    trans_model: TransitionModel,
    sil_phone: int = 1,
) -> None:
    """Insert merged SIL/phone states after SIL forward transitions.

    In Kaldi's determinized HCLG, the SIL phone's forward transition
    leads to a *merged* state that has both a SIL self-loop and the
    next phone's **forward** arc — but **not** the next phone's
    self-loop.  This prevents the decoder from starting to consume
    word-initial phone frames while still in silence.

    Our explicit-chain graph has SIL_FW → phone_state, where
    phone_state has *both* phone_SL (self-loop) and phone_FW
    (forward).  That lets the decoder take phone_SL during silence,
    pulling the word boundary earlier than it should be.

    This function fixes the graph by, for each SIL_FW arc:
      1. Creating a **new intermediate state** with SIL self-loop and
         only the forward (non-self-loop) arcs from the original target.
      2. Adding an additional arc from the SIL source state to this
         intermediate state (keeping the original SIL_FW → phone_state
         arc intact as an alternative path).

    The Viterbi decoder will choose the optimal path: the intermediate
    path for long silences (SIL_SL is cheaper than phone_SL during
    silence) and the direct path for short/no silence boundaries.
    """
    # Find the SIL self-loop tid and its weight from any SIL state
    sil_sl_tid: int = 0
    sil_sl_weight: float = 0.0
    for sid in range(graph.num_states):
        for arc in graph.states[sid].arcs:
            if arc.tid > 0 and trans_model.id2phone[arc.tid] == sil_phone:
                if trans_model.id2is_self_loop[arc.tid]:
                    sil_sl_tid = arc.tid
                    sil_sl_weight = arc.weight
                    break
        if sil_sl_tid > 0:
            break

    if sil_sl_tid == 0:
        return  # no SIL phone in graph

    # Collect all (source_state, arc_index) pairs for SIL forward arcs.
    sil_fw_arcs: list[tuple[int, int]] = []
    for sid in range(graph.num_states):
        for ai, arc in enumerate(graph.states[sid].arcs):
            if arc.tid > 0 and trans_model.id2phone[arc.tid] == sil_phone:
                if not trans_model.id2is_self_loop[arc.tid]:
                    sil_fw_arcs.append((sid, ai))

    # For each SIL forward arc, create an intermediate state and add
    # it as an ALTERNATIVE path (the original arc is kept as-is).
    for src_sid, arc_idx in sil_fw_arcs:
        arc = graph.states[src_sid].arcs[arc_idx]
        orig_target = arc.nextstate

        # Collect forward (non-self-loop) arcs from the original target
        fwd_arcs = [
            a for a in graph.states[orig_target].arcs
            if a.tid > 0 and not trans_model.id2is_self_loop[a.tid]
        ]
        # Also keep epsilon arcs
        eps_arcs = [
            a for a in graph.states[orig_target].arcs
            if a.tid == 0
        ]

        if not fwd_arcs and not eps_arcs:
            continue  # nothing to merge

        # Create new intermediate state
        new_sid = graph.add_state()

        # Add SIL self-loop on the new state
        graph.add_arc(
            new_sid,
            HmmGraphArc(
                tid=sil_sl_tid,
                word_id=0,
                weight=sil_sl_weight,
                nextstate=new_sid,
            ),
        )

        # Copy forward arcs (preserving their destinations and weights)
        for fa in fwd_arcs:
            graph.add_arc(
                new_sid,
                HmmGraphArc(
                    tid=fa.tid,
                    word_id=fa.word_id,
                    weight=fa.weight,
                    nextstate=fa.nextstate,
                ),
            )

        # Copy epsilon arcs
        for ea in eps_arcs:
            graph.add_arc(
                new_sid,
                HmmGraphArc(
                    tid=ea.tid,
                    word_id=ea.word_id,
                    weight=ea.weight,
                    nextstate=ea.nextstate,
                ),
            )

        # Add a NEW arc from src to intermediate (same tid/weight as
        # the original SIL_FW, but targeting the intermediate state).
        # The original SIL_FW → orig_target arc is kept intact.
        graph.add_arc(
            src_sid,
            HmmGraphArc(
                tid=arc.tid,
                word_id=arc.word_id,
                weight=arc.weight,
                nextstate=new_sid,
            ),
        )


def _enumerate_phone_paths(
    lg_fst: StdVectorFst,
    disambig_set: set[int],
) -> list[tuple[list[int], list[int], float]]:
    """Extract all phone-sequence paths from L∘G.

    Returns list of ``(phone_ids, word_boundaries, total_weight)`` tuples.

    ``word_boundaries[i]`` is the word-id that starts at phone index ``i``,
    or 0 if that phone is not at a word boundary.
    """
    if lg_fst.start < 0 or lg_fst.start >= lg_fst.num_states:
        return []

    results: list[tuple[list[int], list[int], float]] = []

    # DFS with (state, phones_so_far, word_boundaries_so_far, weight)
    # Use tuples for immutable accumulation (avoids list copies)
    stack: list[tuple[int, tuple[int, ...], tuple[int, ...], float]] = [
        (lg_fst.start, (), (), 0.0)
    ]

    max_paths = 1000  # safety limit

    while stack and len(results) < max_paths:
        state, phones, word_bounds, weight = stack.pop()

        st = lg_fst.states[state]
        if st.final_weight != WEIGHT_ZERO:
            results.append((list(phones), list(word_bounds), weight + st.final_weight))

        for arc in st.arcs:
            if arc.ilabel != 0 and arc.ilabel not in disambig_set:
                # Real phone arc
                stack.append(
                    (arc.nextstate, phones + (arc.ilabel,),
                     word_bounds + (arc.olabel,), weight + arc.weight)
                )
            else:
                # Epsilon or disambiguation symbol — skip phone, keep word label
                if arc.olabel != 0:
                    stack.append(
                        (arc.nextstate, phones, word_bounds, weight + arc.weight)
                    )
                else:
                    stack.append(
                        (arc.nextstate, phones, word_bounds, weight + arc.weight)
                    )

    return results


def _build_hmm_chain(
    graph: HmmGraph,
    phones: list[int],
    word_boundaries: list[int],
    path_weight: float,
    tree: ContextDependency,
    trans_model: TransitionModel,
) -> None:
    """Build a single-path HMM chain in the graph.

    Sets ``graph.start`` and marks the final state.
    """
    N = tree.N
    P = tree.P

    # Pad phones with silence/zero context at boundaries
    # Kaldi uses phone 0 for out-of-context positions
    padded = [0] * P + phones + [0] * (N - P - 1)

    all_states: list[int] = []  # flat list of HMM states
    all_tids: list[list[tuple[int, int, float]]] = []  # per-HMM-state: (tid_self, tid_fwd, log_prob)
    phone_word_ids: list[int] = []  # word_id for each HMM-state group

    for i, phone in enumerate(phones):
        context = padded[i : i + N]
        topo_entry = trans_model.topology.topology_for_phone(phone)

        # Find the transition tuples for this phone in this context
        for hmm_state_idx, hmm_state in enumerate(topo_entry.states):
            if hmm_state.forward_pdf_class < 0:
                continue  # non-emitting final state

            # Get pdf-id from decision tree
            pdf_class_fwd = hmm_state.forward_pdf_class
            pdf_class_sl = hmm_state.self_loop_pdf_class

            pdf_id_fwd = tree.compute_pdf_id(context, pdf_class_fwd)
            pdf_id_sl = tree.compute_pdf_id(context, pdf_class_sl)

            if pdf_id_fwd is None or pdf_id_sl is None:
                raise ValueError(
                    f"Tree lookup failed for phone={phone}, context={context}, "
                    f"pdf_class_fwd={pdf_class_fwd}, pdf_class_sl={pdf_class_sl}"
                )

            # Find transition-ids for this (phone, hmm_state, pdf_fwd, pdf_sl) tuple
            tid_self, tid_fwd = _find_transition_ids(
                trans_model, phone, hmm_state_idx, pdf_id_fwd, pdf_id_sl
            )

            sid = graph.add_state()
            all_states.append(sid)
            all_tids.append([(tid_self, tid_fwd)])

            # Word ID: attach to first HMM state of each phone
            if hmm_state_idx == 0:
                phone_word_ids.append(word_boundaries[i] if i < len(word_boundaries) else 0)
            else:
                phone_word_ids.append(0)

    if not all_states:
        return

    # Set start and end
    graph.start = all_states[0]
    final_state = graph.add_state(is_final=True)

    # Build arcs: self-loop + forward for each state
    for idx, sid in enumerate(all_states):
        tid_self, tid_fwd = all_tids[idx][0]
        word_id = phone_word_ids[idx] if idx == 0 or phone_word_ids[idx] != 0 else 0

        # Self-loop (emitting)
        if tid_self > 0:
            log_prob = trans_model.get_transition_log_prob(tid_self)
            graph.add_arc(
                sid,
                HmmGraphArc(
                    tid=tid_self,
                    word_id=0,
                    weight=-log_prob,  # negate for tropical (lower = better)
                    nextstate=sid,
                ),
            )

        # Forward transition (emitting)
        if tid_fwd > 0:
            log_prob = trans_model.get_transition_log_prob(tid_fwd)
            next_sid = all_states[idx + 1] if idx + 1 < len(all_states) else final_state
            graph.add_arc(
                sid,
                HmmGraphArc(
                    tid=tid_fwd,
                    word_id=word_id if next_sid != sid else 0,
                    weight=-log_prob,
                    nextstate=next_sid,
                ),
            )

    # Add initial weight from path
    if path_weight != 0.0:
        # Wrap with an extra start state
        real_start = graph.add_state()
        graph.add_arc(
            real_start,
            HmmGraphArc(tid=0, word_id=0, weight=path_weight, nextstate=all_states[0]),
        )
        graph.start = real_start


def _build_hmm_chain_segment(
    graph: HmmGraph,
    phones: list[int],
    word_boundaries: list[int],
    tree: ContextDependency,
    trans_model: TransitionModel,
) -> tuple[int, int]:
    """Like _build_hmm_chain but returns (first_state, last_state) without
    setting graph.start or final flags.  Used when building multi-path graphs.
    """
    N = tree.N
    P = tree.P
    padded = [0] * P + phones + [0] * (N - P - 1)

    all_states: list[int] = []
    all_tids: list[tuple[int, int]] = []  # (tid_self, tid_fwd) per HMM state
    phone_word_ids: list[int] = []  # word_id per HMM state

    for i, phone in enumerate(phones):
        context = padded[i : i + N]
        topo_entry = trans_model.topology.topology_for_phone(phone)

        for hmm_state_idx, hmm_state in enumerate(topo_entry.states):
            if hmm_state.forward_pdf_class < 0:
                continue

            pdf_id_fwd = tree.compute_pdf_id(context, hmm_state.forward_pdf_class)
            pdf_id_sl = tree.compute_pdf_id(context, hmm_state.self_loop_pdf_class)

            if pdf_id_fwd is None or pdf_id_sl is None:
                raise ValueError(f"Tree lookup failed")

            tid_self, tid_fwd = _find_transition_ids(
                trans_model, phone, hmm_state_idx, pdf_id_fwd, pdf_id_sl
            )

            sid = graph.add_state()
            all_states.append(sid)
            all_tids.append((tid_self, tid_fwd))

            if hmm_state_idx == 0:
                phone_word_ids.append(word_boundaries[i] if i < len(word_boundaries) else 0)
            else:
                phone_word_ids.append(0)

    if not all_states:
        dummy = graph.add_state()
        return dummy, dummy

    last_sid = graph.add_state()  # non-emitting end of segment

    for idx, sid in enumerate(all_states):
        tid_self, tid_fwd = all_tids[idx]
        word_id = phone_word_ids[idx] if idx == 0 or phone_word_ids[idx] != 0 else 0

        # Self-loop
        if tid_self > 0:
            log_prob = trans_model.get_transition_log_prob(tid_self)
            graph.add_arc(
                sid,
                HmmGraphArc(tid=tid_self, word_id=0, weight=-log_prob, nextstate=sid),
            )

        # Forward transition
        if tid_fwd > 0:
            log_prob = trans_model.get_transition_log_prob(tid_fwd)
            next_sid = all_states[idx + 1] if idx + 1 < len(all_states) else last_sid
            graph.add_arc(
                sid,
                HmmGraphArc(
                    tid=tid_fwd,
                    word_id=word_id if next_sid != sid else 0,
                    weight=-log_prob,
                    nextstate=next_sid,
                ),
            )

    return all_states[0], last_sid


def _find_transition_ids(
    trans_model: TransitionModel,
    phone: int,
    hmm_state: int,
    forward_pdf: int,
    self_loop_pdf: int,
) -> tuple[int, int]:
    """Find the self-loop and forward transition-ids for given (phone, hmm_state, pdfs).

    Returns ``(tid_self_loop, tid_forward)``.
    """
    cache = _get_transition_cache(trans_model)
    key = (phone, hmm_state, forward_pdf, self_loop_pdf)
    result = cache.get(key)
    if result is not None:
        return result

    # Fallback: try with just (phone, hmm_state)
    fallback_key = (phone, hmm_state)
    result = cache.get(("_fallback", fallback_key))
    if result is not None:
        return result

    raise ValueError(
        f"No transition tuple found for phone={phone}, hmm_state={hmm_state}, "
        f"forward_pdf={forward_pdf}, self_loop_pdf={self_loop_pdf}"
    )


def _get_transition_cache(trans_model: TransitionModel) -> dict:
    """Build and cache a lookup dict: (phone, hmm_state, fwd_pdf, sl_pdf) → (tid_self, tid_fwd)."""
    cache_attr = "_transition_cache"
    if hasattr(trans_model, cache_attr):
        return getattr(trans_model, cache_attr)

    cache: dict = {}
    for s_idx, tup in enumerate(trans_model.tuples):
        trans_state = s_idx + 1  # 1-based
        first_tid = trans_model._state2id[trans_state]
        last_tid = trans_model._state2id[trans_state + 1]

        tid_self = 0
        tid_fwd = 0
        for tid in range(first_tid, last_tid):
            if trans_model.is_self_loop(tid):
                tid_self = tid
            else:
                tid_fwd = tid

        key = (tup.phone, tup.hmm_state, tup.forward_pdf, tup.self_loop_pdf)
        cache[key] = (tid_self, tid_fwd)
        # Also store fallback by (phone, hmm_state) — last one wins
        cache[("_fallback", (tup.phone, tup.hmm_state))] = (tid_self, tid_fwd)

    setattr(trans_model, cache_attr, cache)
    return cache
