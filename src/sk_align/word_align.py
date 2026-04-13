"""
Word-level alignment extraction from frame-level transition-id sequences.

Given a frame-by-frame alignment (list of transition-ids) and word boundary
information, groups frames into words with start/end timestamps.

Reference: kaldi/src/lat/word-align-lattice.cc, lattice-functions.cc
"""

from __future__ import annotations

from dataclasses import dataclass

from sk_align.kaldi_io import read_word_boundary
from sk_align.transition_model import TransitionModel

# Phone boundary types (from word_boundary.int)
NONWORD = "nonword"
BEGIN = "begin"
END = "end"
SINGLETON = "singleton"
INTERNAL = "internal"


@dataclass
class WordSegment:
    """A single word with frame-level timing."""

    word_id: int
    start_frame: int
    duration_frames: int


def extract_word_alignment(
    alignment: list[int],
    trans_model: TransitionModel,
    word_boundary: dict[int, str],
    word_ids_from_graph: list[int] | None = None,
) -> list[WordSegment]:
    """Extract word-level alignment from a frame-level transition-id sequence.

    Uses phone boundary types from ``word_boundary.int`` to determine where
    words start and end.  This mirrors Kaldi's ``CompactLatticeToWordAlignment``
    after ``WordAlignLattice``.

    Parameters
    ----------
    alignment : list[int]
        Transition-id per frame (from Viterbi decoding).
    trans_model : TransitionModel
        For mapping transition-ids to phones.
    word_boundary : dict[int, str]
        Phone-id → boundary type (from ``word_boundary.int``).
    word_ids_from_graph : list[int] or None
        Word IDs from the decoding graph traceback (alternative to inferring
        from phone boundaries).

    Returns
    -------
    list[WordSegment]
        Word segments with frame-level timing.
    """
    if not alignment:
        return []

    # Convert alignment to phone sequence with frame ranges
    phone_segments = _alignment_to_phones(alignment, trans_model)

    # Group phones into words using boundary types
    words: list[WordSegment] = []
    current_start = -1
    current_word_id = 0
    word_id_idx = 0

    for phone, start_frame, num_frames in phone_segments:
        btype = word_boundary.get(phone, NONWORD)

        if btype == SINGLETON:
            # Single-phone word
            if word_ids_from_graph and word_id_idx < len(word_ids_from_graph):
                wid = word_ids_from_graph[word_id_idx]
                word_id_idx += 1
            else:
                wid = 0
            words.append(WordSegment(wid, start_frame, num_frames))

        elif btype == BEGIN:
            current_start = start_frame
            if word_ids_from_graph and word_id_idx < len(word_ids_from_graph):
                current_word_id = word_ids_from_graph[word_id_idx]
                word_id_idx += 1
            else:
                current_word_id = 0

        elif btype == END:
            if current_start >= 0:
                total_dur = start_frame + num_frames - current_start
                words.append(WordSegment(current_word_id, current_start, total_dur))
            current_start = -1

        elif btype == INTERNAL:
            pass  # mid-word phone, accumulate

        elif btype == NONWORD:
            # Silence / noise — emit as word-id 0 (epsilon)
            words.append(WordSegment(0, start_frame, num_frames))

    return words


def _alignment_to_phones(
    alignment: list[int],
    trans_model: TransitionModel,
) -> list[tuple[int, int, int]]:
    """Convert frame-level alignment to phone segments.

    Returns list of ``(phone_id, start_frame, num_frames)`` tuples.
    Uses the transition model to detect phone boundaries (a new phone starts
    when the phone-id changes or when we see a non-self-loop transition
    following a self-loop to a different transition-state).
    """
    if not alignment:
        return []

    segments: list[tuple[int, int, int]] = []
    current_phone = trans_model.transition_id_to_phone(alignment[0])
    current_start = 0

    # Track by transition-state changes (more reliable than phone changes alone)
    current_trans_state = trans_model._id2state[alignment[0]]

    for frame in range(1, len(alignment)):
        tid = alignment[frame]
        phone = trans_model.transition_id_to_phone(tid)
        trans_state = trans_model._id2state[tid]

        # Phone boundary: phone changed, or transition-state changed and
        # the previous frame was not a self-loop to the same state
        if phone != current_phone or (
            trans_state != current_trans_state
            and not trans_model.is_self_loop(alignment[frame])
        ):
            segments.append((current_phone, current_start, frame - current_start))
            current_phone = phone
            current_start = frame
            current_trans_state = trans_state
        else:
            current_trans_state = trans_state

    # Final segment
    segments.append((current_phone, current_start, len(alignment) - current_start))

    return segments


def word_alignment_to_timestamps(
    word_segments: list[WordSegment],
    id_to_symbol: dict[int, str],
    frame_dur: float = 0.03,
    offset: float = 0.0,
) -> list[dict]:
    """Convert word segments to timestamped word list.

    Parameters
    ----------
    word_segments : list[WordSegment]
        From ``extract_word_alignment``.
    id_to_symbol : dict[int, str]
        Word-id → symbol string (reverse of ``words.txt``).
    frame_dur : float
        Duration of one frame in seconds (0.03 for frame_subsampling_factor=3).
    offset : float
        Time offset to add to all timestamps.

    Returns
    -------
    list[dict]
        ``[{"word": "hello", "start": 0.12, "end": 0.45}, ...]``
        Excludes silence/epsilon words (word_id == 0).
    """
    results = []
    for seg in word_segments:
        if seg.word_id == 0:
            continue  # skip silence
        symbol = id_to_symbol.get(seg.word_id, "<unk>")
        if symbol == "<eps>":
            continue
        results.append({
            "word": symbol,
            "start": round(seg.start_frame * frame_dur + offset, 3),
            "end": round((seg.start_frame + seg.duration_frames) * frame_dur + offset, 3),
        })
    return results
