"""
k2-based Viterbi decoder for forced alignment.

Converts the ``HmmGraph`` (from ``graph.py``) into a k2 FSA and uses
``intersect_dense`` + ``shortest_path`` for efficient Viterbi decoding.

Requires: ``pip install k2`` (matched to your PyTorch version).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import k2
import numpy as np
import torch

if TYPE_CHECKING:
    from sk_align.graph import HmmGraph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def viterbi_decode_k2(
    graph: "HmmGraph",
    loglikes: np.ndarray,
    tid_to_pdf: list[int],
    acoustic_scale: float = 0.1,
    output_beam: float = 10.0,
) -> tuple[list[int], list[int], float, bool]:
    """Viterbi alignment using k2.

    Parameters
    ----------
    graph : HmmGraph
        Per-utterance decoding graph (from ``compile_training_graph``).
    loglikes : np.ndarray
        Shape ``(num_frames, num_pdfs)`` — nnet3 log-likelihoods.
    tid_to_pdf : list[int]
        Mapping from transition-id → pdf-id.
    acoustic_scale : float
        Scaling factor for acoustic scores.
    output_beam : float
        Beam for ``intersect_dense`` lattice pruning.

    Returns
    -------
    alignment : list[int]
        Transition-id per frame.
    word_ids : list[int]
        Word-ids encountered along the best path.
    best_cost : float
        Total path cost (tropical semiring — lower is better).
    succeeded : bool
        True if a valid alignment was found.
    """
    if k2 is None:
        raise ImportError("k2 is required for k2_decoder. pip install k2")

    num_frames, num_pdfs = loglikes.shape

    # ---- 1. Convert HmmGraph → epsilon-free k2 Fsa ----
    fsa, arc_tids, arc_wids = _hmm_graph_to_k2(graph, tid_to_pdf)
    if fsa is None:
        return [], [], float("inf"), False

    # ---- 2. Prepare DenseFsaVec from log-likelihoods ----
    #  Shape (1, T, num_pdfs).  Labels in the FSA are 0‥num_pdfs-1,
    #  which index directly into columns of this matrix.
    loglikes_t = (
        torch.from_numpy(loglikes.astype(np.float32)) * acoustic_scale
    ).unsqueeze(0)                       # (1, T, num_pdfs)
    supervision = torch.tensor([[0, 0, num_frames]], dtype=torch.int32)
    dense = k2.DenseFsaVec(loglikes_t, supervision)

    # ---- 3.  intersect_dense + shortest_path ----
    fsa_vec = k2.create_fsa_vec([fsa])
    try:
        lattice = k2.intersect_dense(fsa_vec, dense, output_beam=output_beam)
        best = k2.shortest_path(lattice, use_double_scores=True)
    except Exception:
        return [], [], float("inf"), False

    if best[0].num_arcs == 0:
        return [], [], float("inf"), False

    # ---- 4. Traceback via arc-map chain ----
    total_score = best[0].scores.sum().item()
    total_cost = -total_score           # negate: k2 log-probs → tropical cost

    alignment, word_ids = _traceback(best, lattice, arc_tids, arc_wids)

    return alignment, word_ids, total_cost, len(alignment) > 0


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _epsilon_closure(graph: "HmmGraph") -> dict[int, dict[int, float]]:
    """Compute epsilon closure for every state.

    Returns ``{state: {reachable_state: accumulated_weight, …}, …}``.
    Weight semantics: tropical (lower = better), same as ``HmmGraphArc.weight``.

    Optimisation: pre-compute which states have epsilon *outgoing* arcs,
    and skip BFS for all-emitting states (closure = {s: 0.0}).
    """
    # Pre-scan: which states have outgoing epsilon arcs?
    has_eps: set[int] = set()
    for s in range(graph.num_states):
        for arc in graph.states[s].arcs:
            if arc.tid == 0:
                has_eps.add(s)
                break

    # Also need BFS for states reachable through epsilon from others;
    # but we only need closures for states that START the BFS.  A state's
    # closure is non-trivial only if it can reach an epsilon-arc state.
    closures: dict[int, dict[int, float]] = {}
    for s in range(graph.num_states):
        if s not in has_eps:
            # Trivial closure — no epsilon arcs from this state.
            # But we still need to check eps-reachable: since no outgoing
            # epsilon from `s`, closure is just {s: 0.0}.
            closures[s] = {s: 0.0}
            continue
        # BFS
        reached: dict[int, float] = {s: 0.0}
        q: deque[int] = deque([s])
        while q:
            cur = q.popleft()
            cur_w = reached[cur]
            for arc in graph.states[cur].arcs:
                if arc.tid == 0:                             # epsilon
                    nw = cur_w + arc.weight
                    if arc.nextstate not in reached or nw < reached[arc.nextstate]:
                        reached[arc.nextstate] = nw
                        q.append(arc.nextstate)
        closures[s] = reached
    return closures


def _hmm_graph_to_k2(
    graph: "HmmGraph",
    tid_to_pdf: list[int],
) -> tuple["k2.Fsa | None", list[int], list[int]]:
    """Convert an ``HmmGraph`` into an epsilon-free k2 Fsa.

    Arc labels are pdf-ids (0-based), suitable for ``intersect_dense``
    where they index into the columns of the log-likelihood matrix.

    Returns ``(fsa, arc_tids, arc_wids)`` where *arc_tids* / *arc_wids*
    give the transition-id and word-id for each arc **in the k2 fsa** (by
    arc index, excluding the final arcs).
    """
    closures = _epsilon_closure(graph)

    # -- collect emitting arcs with epsilon removal --------------------------
    raw: list[tuple[int, int, int, float, int, int]] = []
    #          src   dst   pdf   gw     tid   wid

    for s in range(graph.num_states):
        for s_eps, eps_w in closures[s].items():
            for arc in graph.states[s_eps].arcs:
                if arc.tid == 0:
                    continue                                 # skip eps
                pdf = tid_to_pdf[arc.tid]
                gw = eps_w + arc.weight     # accumulated graph weight (tropical)
                raw.append((s, arc.nextstate, pdf, gw, arc.tid, arc.word_id))

    if not raw:
        return None, [], []

    # -- determine final states (reach a final via epsilon) ------------------
    final_set: set[int] = set()
    for s in range(graph.num_states):
        for s_eps in closures[s]:
            if graph.states[s_eps].is_final:
                final_set.add(s)

    # -- forward reachability from start (BFS on emitting arcs) ---------------
    # Build adjacency list from raw arcs for O(V+E) BFS
    adj: dict[int, list[int]] = {}
    for src, dst, *_ in raw:
        adj.setdefault(src, []).append(dst)

    reachable: set[int] = set()
    bfs_q: deque[int] = deque()

    def _add_reachable(s: int) -> None:
        if s not in reachable:
            reachable.add(s)
            bfs_q.append(s)
            for s_eps in closures.get(s, {s: 0.0}):
                if s_eps not in reachable:
                    reachable.add(s_eps)
                    bfs_q.append(s_eps)

    _add_reachable(graph.start)

    while bfs_q:
        cur = bfs_q.popleft()
        for dst in adj.get(cur, []):
            _add_reachable(dst)

    # -- state remapping (start → 0, super-final → last) --------------------
    active = sorted(reachable)
    if graph.start in active:
        active.remove(graph.start)
    active = [graph.start] + active
    s2k = {s: i for i, s in enumerate(active)}
    super_final = len(active)

    # -- build k2 arc list ---------------------------------------------------
    k2_arcs: list[tuple[int, int, int, float]] = []
    arc_tids: list[int] = []
    arc_wids: list[int] = []

    for src, dst, pdf, gw, tid, wid in raw:
        if src not in s2k or dst not in s2k:
            continue
        score = -gw                       # tropical → log-prob
        k2_arcs.append((s2k[src], s2k[dst], pdf, score))
        arc_tids.append(tid)
        arc_wids.append(wid)

    for s in final_set & set(active):
        k2_arcs.append((s2k[s], super_final, -1, 0.0))
        # no tid/wid for final arcs — append placeholders to keep indices aligned
        arc_tids.append(0)
        arc_wids.append(0)

    if not k2_arcs:
        return None, [], []

    # sort by (src, dst, label) — k2 requires arcs grouped by source state
    order = sorted(range(len(k2_arcs)),
                   key=lambda i: (k2_arcs[i][0], k2_arcs[i][1], k2_arcs[i][2]))
    k2_arcs = [k2_arcs[i] for i in order]
    arc_tids = [arc_tids[i] for i in order]
    arc_wids = [arc_wids[i] for i in order]

    # -- build k2 Fsa -------------------------------------------------------
    lines: list[str] = []
    for src, dst, label, score in k2_arcs:
        lines.append(f"{src} {dst} {label} {score:.6f}")
    lines.append(str(super_final))
    fsa = k2.Fsa.from_str("\n".join(lines))

    # store metadata tensors on the fsa (k2 propagates these via arc_map)
    fsa.tid = torch.tensor(arc_tids, dtype=torch.int32)
    fsa.wid = torch.tensor(arc_wids, dtype=torch.int32)

    return fsa, arc_tids, arc_wids


def _traceback(
    best: "k2.Fsa",
    lattice: "k2.Fsa",
    arc_tids: list[int],
    arc_wids: list[int],
) -> tuple[list[int], list[int]]:
    """Map k2 best-path arcs back to transition-ids and word-ids.

    Tries three strategies in order:
    1. Propagated tensor attributes (``best[0].tid``, ``best[0].wid``).
    2. Arc-map chain (best → lattice → original fsa).
    3. Direct label lookup via the ``tid`` / ``wid`` lists (least reliable).
    """
    bfsa = best[0]

    # --- Strategy 1: propagated attributes ---
    try:
        tids = bfsa.tid.tolist()
        wids = bfsa.wid.tolist()
        alignment = [t for t in tids if t > 0]
        word_ids = [w for w in wids if w > 0]
        if alignment:
            return alignment, word_ids
    except Exception:
        pass

    # --- Strategy 2: arc-map chain ---
    try:
        # best_path arc → lattice arc → original fsa arc
        bp_map = bfsa.arc_map.tolist() if hasattr(bfsa, "arc_map") else None
        lat_map = lattice.arc_map if hasattr(lattice, "arc_map") else None

        if bp_map is not None and lat_map is not None:
            lat_map_list = lat_map.tolist()
            alignment: list[int] = []
            word_ids: list[int] = []
            for bp_idx in bp_map:
                if 0 <= bp_idx < len(lat_map_list):
                    orig = lat_map_list[bp_idx]
                    if 0 <= orig < len(arc_tids):
                        if arc_tids[orig] > 0:
                            alignment.append(arc_tids[orig])
                        if arc_wids[orig] > 0:
                            word_ids.append(arc_wids[orig])
            if alignment:
                return alignment, word_ids
    except Exception:
        pass

    # --- Strategy 3: labels fallback ---
    labels = bfsa.labels.tolist()
    alignment = []
    for label in labels:
        if label >= 0 and label != -1:
            alignment.append(label)            # pdf-id, not tid — imperfect
    return alignment, []
