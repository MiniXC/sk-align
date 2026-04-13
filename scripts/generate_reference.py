"""
Generate reference alignment data using the pykaldi-based aligner.

This script is run from the pykaldi conda environment to produce golden
test data that the sk-align parity tests compare against.

Usage (from the pykaldi conda env):
    python scripts/generate_reference.py <model_dir> <audio_path> <transcript_path> <output_dir> [--max-seconds N]

Example:
    python scripts/generate_reference.py \\
        .model_cache/.../dd87b.../  \\
        test_files/m001rmb0.wav \\
        test_files/m001rmb0.txt \\
        tests/reference_data/m001rmb0/ \\
        --max-seconds 10
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def parse_transcript(text):
    """Parse a transcript file into a flat list of words.

    Handles:
    - ``<Name>`` → keeps inner text as a word (names / English)
    - ``[7](seachd)`` → uses the parenthesised Gaelic form
    - ``<English sentence.>`` → keeps as individual words
    - Strips punctuation from words (.,;:!?) but keeps hyphens/apostrophes
    """
    # Strip BOM
    text = text.lstrip('\ufeff')
    # Replace [N](gaelic form) with the Gaelic form
    text = re.sub(r'\[.*?\]\(([^)]+)\)', r'\1', text)
    # Replace <word> with the word (strip angle brackets)
    text = re.sub(r'<([^>]+)>', r'\1', text)
    # Split into words
    tokens = text.split()
    # Strip leading/trailing punctuation but keep hyphens & apostrophes
    cleaned = []
    for tok in tokens:
        w = tok.strip('.,;:!?()[]"\'…')
        if w:
            cleaned.append(w.lower())
    return cleaned


def estimate_words_for_duration(words, max_seconds, speaking_rate=2.5):
    """Estimate how many words fit in *max_seconds* of audio.

    Uses a rough speaking rate (words/sec) to avoid passing a 30-minute
    transcript when we only use 10 seconds of audio.
    """
    n = int(max_seconds * speaking_rate)
    return words[:max(n, 5)]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate pykaldi reference data")
    parser.add_argument("model_dir", help="Path to nnet3 alignment model directory")
    parser.add_argument("audio_path", help="Path to WAV file")
    parser.add_argument("transcript_path", help="Path to .txt transcript")
    parser.add_argument("output_dir", help="Output directory for reference data")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="Only process first N seconds of audio (0 = all)")
    parser.add_argument("--words-json", type=str, default=None,
                        help="Override words as JSON array (instead of transcript)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load audio ---
    import wave
    with wave.open(args.audio_path, "r") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnchannels() == 1
        max_frames = int(args.max_seconds * 16000) if args.max_seconds > 0 else wf.getnframes()
        n = min(wf.getnframes(), max_frames)
        raw_bytes = wf.readframes(n)
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"Audio: {len(audio)} samples ({len(audio)/16000:.2f}s)")

    # --- Parse transcript ---
    if args.words_json:
        words = json.loads(args.words_json)
    else:
        transcript = Path(args.transcript_path).read_text(encoding="utf-8")
        all_words = parse_transcript(transcript)
        if args.max_seconds > 0:
            words = estimate_words_for_duration(all_words, args.max_seconds)
        else:
            words = all_words

    print(f"Words ({len(words)}): {' '.join(words[:20])}{'...' if len(words) > 20 else ''}")

    # --- pykaldi imports ---
    try:
        from kaldi.feat.mfcc import Mfcc, MfccOptions
        from kaldi.matrix import Vector
        from kaldi.alignment import NnetAligner
        from kaldi.fstext import SymbolTable
        from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
        from kaldi.nnet3 import NnetSimpleComputationOptions
    except ImportError:
        print("ERROR: pykaldi not available. Run from the pykaldi conda env.")
        sys.exit(1)

    # --- Load model ---
    model_path = args.model_dir
    print(f"Loading model from {model_path}...")
    decodable_opts = NnetSimpleComputationOptions()
    decodable_opts.frame_subsampling_factor = 3
    aligner = NnetAligner.from_files(
        f"{model_path}/final.mdl",
        f"{model_path}/tree",
        f"{model_path}/L.fst",
        f"{model_path}/words.txt",
        f"{model_path}/disambig.int",
        decodable_opts=decodable_opts,
    )
    wb_info = WordBoundaryInfo.from_file(
        WordBoundaryInfoNewOpts(), f"{model_path}/word_boundary.int"
    )

    mfcc_opts = MfccOptions()
    mfcc_opts.frame_opts.samp_freq = 16000
    mfcc_opts.use_energy = False
    mfcc_opts.mel_opts.num_bins = 40
    mfcc_opts.num_ceps = 40
    mfcc_opts.mel_opts.low_freq = 20
    mfcc_opts.mel_opts.high_freq = -400
    mfcc = Mfcc(mfcc_opts)

    vocabulary = set()
    with open(f"{model_path}/words.txt") as f:
        for line in f:
            vocabulary.add(line.strip().split()[0])

    # --- Compute MFCC ---
    wav = audio * (1 << 15)
    feats = mfcc.compute_features(Vector(wav), 16000, 1.0)
    feats_np = np.array(feats)
    print(f"MFCC features: {feats_np.shape}")

    # --- Save MFCC ---
    np.save(output_dir / "mfcc_features.npy", feats_np)

    # --- Align ---
    mapped = [w if w in vocabulary else "<unk>" for w in words]
    oov = [w for w in words if w not in vocabulary]
    if oov:
        print(f"OOV words ({len(oov)}): {oov[:20]}")
    text = " ".join(mapped)

    result = aligner.align(feats, text)
    word_alignment = aligner.to_word_alignment(result["best_path"], wb_info)

    # --- Save alignment ---
    alignment = list(result["alignment"])
    np.save(output_dir / "alignment.npy", np.array(alignment, dtype=np.int32))

    # --- Save word alignment ---
    FRAME_DUR = 0.03
    word_align_data = []
    word_idx = 0
    for word_label, start_frame, dur_frames in word_alignment:
        if word_label == "<eps>":
            word_align_data.append({
                "word": "<eps>",
                "start_frame": int(start_frame),
                "dur_frames": int(dur_frames),
            })
            continue
        if word_idx < len(words):
            word_align_data.append({
                "word": words[word_idx],
                "start_frame": int(start_frame),
                "dur_frames": int(dur_frames),
                "start": round(start_frame * FRAME_DUR, 3),
                "end": round((start_frame + dur_frames) * FRAME_DUR, 3),
            })
            word_idx += 1

    with open(output_dir / "word_alignment.json", "w") as f:
        json.dump(word_align_data, f, indent=2)

    # --- Save final output (matching aligner.py format) ---
    results = []
    word_idx = 0
    for word_label, start_frame, dur_frames in word_alignment:
        if word_label == "<eps>":
            continue
        if word_idx < len(words):
            results.append({
                "word": words[word_idx],
                "start": round(start_frame * FRAME_DUR, 3),
                "end": round((start_frame + dur_frames) * FRAME_DUR, 3),
            })
            word_idx += 1

    with open(output_dir / "expected_output.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReference data saved to {output_dir}/")
    print(f"  mfcc_features.npy: {feats_np.shape}")
    print(f"  alignment.npy: {len(alignment)} frames")
    print(f"  word_alignment.json: {len(word_align_data)} segments")
    print(f"  expected_output.json: {len(results)} words")
    print("\nWord alignment:")
    for r in results:
        print(f"  {r['word']:20s}  {r['start']:.3f} – {r['end']:.3f}s")

    # --- Save audio ---
    np.save(output_dir / "audio.npy", audio.astype(np.float32))

    # --- Save metadata ---
    meta = {
        "audio_path": str(args.audio_path),
        "transcript_path": str(args.transcript_path),
        "sample_rate": 16000,
        "num_samples": int(len(audio)),
        "duration_seconds": round(len(audio) / 16000, 3),
        "words": words,
        "model_path": str(model_path),
        "max_seconds": args.max_seconds,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
