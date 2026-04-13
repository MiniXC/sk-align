"""
High-level forced-alignment API.

Provides the ``Aligner`` class that matches the interface of the
pykaldi-based ``scottish-gaelic-subtitling/aligner.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from sk_align.fst import StdVectorFst
from sk_align.graph import compile_training_graph
from sk_align.k2_decoder import k2_available, viterbi_decode_k2
from sk_align.kaldi_io import (
    read_disambig_symbols,
    read_symbol_table,
    read_symbol_table_reverse,
    read_word_boundary,
)
from sk_align.mfcc import MfccOptions, compute_mfcc
from sk_align.transition_model import TransitionModel
from sk_align.tree import ContextDependency
from sk_align.viterbi import AlignmentResult, MatrixAcousticScorer, viterbi_decode
from sk_align.word_align import (
    extract_word_alignment,
    word_alignment_to_timestamps,
)


# Frame duration: 10ms base frame × frame_subsampling_factor=3 = 30ms
FRAME_DUR = 0.03


class NnetScorer(Protocol):
    """Interface for neural network acoustic scoring.

    Implementations can be:
    - Pre-computed log-likelihoods (for testing)
    - ONNX model
    - PyTorch model
    """

    def compute_log_likelihoods(self, features: np.ndarray) -> np.ndarray:
        """Compute log-likelihoods from MFCC features.

        Parameters
        ----------
        features : np.ndarray
            MFCC features, shape ``(num_frames, num_ceps)``.

        Returns
        -------
        np.ndarray
            Log-likelihoods, shape ``(num_output_frames, num_pdfs)``.
            With frame_subsampling_factor=3, ``num_output_frames ≈ num_frames / 3``.
        """
        ...


class PrecomputedScorer:
    """Scorer that uses pre-computed log-likelihood matrices.

    For testing: load a ``.npy`` file with the nnet3 output and use it
    for alignment without needing the actual neural network.
    """

    def __init__(self, loglikes: np.ndarray):
        self._loglikes = loglikes

    def compute_log_likelihoods(self, features: np.ndarray) -> np.ndarray:
        return self._loglikes

    @classmethod
    def from_file(cls, path: str | Path) -> PrecomputedScorer:
        return cls(np.load(path))


@dataclass
class Aligner:
    """Standalone forced aligner.

    Drop-in replacement for the pykaldi-based aligner in
    ``scottish-gaelic-subtitling/aligner.py``.
    """

    trans_model: TransitionModel
    tree: ContextDependency
    lexicon_fst: StdVectorFst
    word_to_id: dict[str, int]
    id_to_word: dict[int, str]
    disambig_syms: list[int]
    word_boundary: dict[int, str]
    vocabulary: set[str]
    mfcc_opts: MfccOptions
    nnet_scorer: NnetScorer | None = None
    acoustic_scale: float = 0.1
    frame_subsampling_factor: int = 3

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str | Path,
        nnet_scorer: NnetScorer | None = None,
        acoustic_scale: float = 0.1,
        frame_subsampling_factor: int = 3,
    ) -> Aligner:
        """Load all model files from a Kaldi nnet3 alignment model directory.

        Expected files:
        - ``final.mdl`` — TransitionModel (+ acoustic model, skipped here)
        - ``tree`` — ContextDependency
        - ``L.fst`` — Lexicon FST
        - ``words.txt`` — Word symbol table
        - ``disambig.int`` — Disambiguation symbols
        - ``word_boundary.int`` — Phone boundary types

        Parameters
        ----------
        model_dir : str or Path
            Path to the model directory.
        nnet_scorer : NnetScorer or None
            Neural network scorer.  If None, you must supply log-likelihoods
            directly via ``align_with_loglikes``.
        """
        d = Path(model_dir)

        trans_model = TransitionModel.from_file(d / "final.mdl")
        tree = ContextDependency.from_file(d / "tree")
        lexicon_fst = StdVectorFst.from_file(d / "L.fst")
        word_to_id = read_symbol_table(d / "words.txt")
        id_to_word = read_symbol_table_reverse(d / "words.txt")
        disambig_syms = read_disambig_symbols(d / "disambig.int")
        word_boundary = read_word_boundary(d / "word_boundary.int")
        vocabulary = set(word_to_id.keys())

        # MFCC options matching the SG aligner
        mfcc_opts = MfccOptions(
            sample_freq=16000.0,
            use_energy=False,
            num_mel_bins=40,
            num_ceps=40,
            low_freq=20.0,
            high_freq=-400.0,
        )

        return cls(
            trans_model=trans_model,
            tree=tree,
            lexicon_fst=lexicon_fst,
            word_to_id=word_to_id,
            id_to_word=id_to_word,
            disambig_syms=disambig_syms,
            word_boundary=word_boundary,
            vocabulary=vocabulary,
            mfcc_opts=mfcc_opts,
            nnet_scorer=nnet_scorer,
            acoustic_scale=acoustic_scale,
            frame_subsampling_factor=frame_subsampling_factor,
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "eist-edinburgh/nnet3_alignment_model",
        *,
        cache_dir: str | Path | None = None,
        token: str | None = None,
        acoustic_scale: float = 0.1,
        frame_subsampling_factor: int = 3,
        device: str = "cpu",
    ) -> Aligner:
        """Download a model from Hugging Face Hub and return a ready-to-use Aligner.

        This is the simplest way to get started::

            from sk_align import Aligner

            aligner = Aligner.from_pretrained()
            timestamps = aligner.align(audio, ["hello", "world"])

        Parameters
        ----------
        repo_id : str
            Hugging Face Hub repository ID (default:
            ``"eist-edinburgh/nnet3_alignment_model"``).
        cache_dir : str, Path, or None
            Where to cache the downloaded snapshot.  Defaults to the
            Hugging Face Hub cache (``~/.cache/huggingface/hub``).
        token : str or None
            Hugging Face access token for private repos.  Also read from
            the ``HF_TOKEN`` environment variable.
        acoustic_scale : float
            Acoustic model scale for decoding (default 0.1).
        frame_subsampling_factor : int
            Frame subsampling factor of the nnet3 model (default 3).
        device : str
            PyTorch device for nnet inference (``"cpu"`` or ``"cuda"``).

        Returns
        -------
        Aligner
            Fully initialised aligner with the PyTorch nnet scorer.

        Raises
        ------
        ImportError
            If ``huggingface_hub`` or ``torch`` is not installed.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for from_pretrained(). "
                "Install it with: pip install huggingface_hub"
            ) from None

        try:
            from sk_align.nnet3_torch import TorchNnetScorer
        except ImportError:
            raise ImportError(
                "torch is required for from_pretrained(). "
                "Install it with: pip install torch"
            ) from None

        resolved_token = token or os.environ.get("HF_TOKEN")
        kwargs: dict = {"token": resolved_token}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)

        model_dir = Path(snapshot_download(repo_id, **kwargs))

        nnet_scorer = TorchNnetScorer.from_model_file(
            model_dir / "final.mdl",
            frame_subsampling_factor=frame_subsampling_factor,
            device=device,
        )

        return cls.from_model_dir(
            model_dir,
            nnet_scorer=nnet_scorer,
            acoustic_scale=acoustic_scale,
            frame_subsampling_factor=frame_subsampling_factor,
        )

    def align(
        self,
        audio: np.ndarray,
        words: list[str],
        offset: float = 0.0,
    ) -> list[dict]:
        """Forced-align *words* to *audio* (float32 numpy array, 16 kHz).

        Matches the interface of the pykaldi-based aligner.

        Parameters
        ----------
        audio : np.ndarray
            Float32 audio samples at 16 kHz.
        words : list[str]
            Words to align.
        offset : float
            Time offset added to all timestamps.

        Returns
        -------
        list[dict]
            ``[{"word": "hello", "start": 0.12, "end": 0.45}, ...]``
        """
        if not words:
            return []

        if self.nnet_scorer is None:
            raise RuntimeError(
                "No NnetScorer configured.  Use align_with_loglikes() "
                "or provide an NnetScorer via from_model_dir()."
            )

        # 1. Compute MFCC features
        # Kaldi expects int16-range samples
        wav = audio.astype(np.float64) * (1 << 15)
        features = compute_mfcc(wav, self.mfcc_opts)

        # 2. Compute nnet log-likelihoods
        loglikes = self.nnet_scorer.compute_log_likelihoods(features)

        return self.align_with_loglikes(loglikes, words, offset)

    def align_with_loglikes(
        self,
        loglikes: np.ndarray,
        words: list[str],
        offset: float = 0.0,
    ) -> list[dict]:
        """Forced-align using pre-computed log-likelihoods.

        Parameters
        ----------
        loglikes : np.ndarray
            Shape ``(num_frames, num_pdfs)`` — raw nnet3 log-likelihoods.
        words : list[str]
            Words to align.
        offset : float
            Time offset.

        Returns
        -------
        list[dict]
            Word-level timestamps.
        """
        if not words:
            return []

        # Map OOV words to <unk>
        mapped = [w if w in self.vocabulary else "<unk>" for w in words]
        word_ids = [self.word_to_id[w] for w in mapped]

        # 3. Compile per-utterance decoding graph
        graph = compile_training_graph(
            word_ids,
            self.lexicon_fst,
            self.tree,
            self.trans_model,
            self.disambig_syms,
        )

        # 4. Create acoustic scorer
        scorer = MatrixAcousticScorer(
            loglikes,
            self.trans_model.id2pdf,
            self.acoustic_scale,
        )

        # 5. Viterbi decode (k2 when available, pure-Python fallback)
        if k2_available():
            alignment, word_id_list, cost, ok = viterbi_decode_k2(
                graph,
                loglikes,
                self.trans_model.id2pdf,
                acoustic_scale=self.acoustic_scale,
            )
            result = AlignmentResult(
                alignment=alignment,
                best_cost=cost,
                word_ids=word_id_list,
                succeeded=ok,
            )
        else:
            result = viterbi_decode(graph, scorer, beam=200.0)

        if not result.succeeded:
            # Fallback: return evenly-spaced words
            return self._fallback_alignment(words, loglikes.shape[0], offset)

        # 6. Extract word boundaries
        word_segments = extract_word_alignment(
            result.alignment,
            self.trans_model,
            self.word_boundary,
            result.word_ids,
        )

        # 7. Convert to timestamps
        timestamps = word_alignment_to_timestamps(
            word_segments,
            self.id_to_word,
            frame_dur=FRAME_DUR,
            offset=offset,
        )

        # Map back to original words (undo <unk> mapping)
        for i, ts in enumerate(timestamps):
            if i < len(words):
                ts["word"] = words[i]

        return timestamps

    def _fallback_alignment(
        self,
        words: list[str],
        num_frames: int,
        offset: float,
    ) -> list[dict]:
        """Even spacing fallback when alignment fails."""
        if not words or num_frames == 0:
            return []
        frames_per_word = num_frames / len(words)
        results = []
        for i, word in enumerate(words):
            start_frame = int(i * frames_per_word)
            end_frame = int((i + 1) * frames_per_word)
            results.append({
                "word": word,
                "start": round(start_frame * FRAME_DUR + offset, 3),
                "end": round(end_frame * FRAME_DUR + offset, 3),
            })
        return results
