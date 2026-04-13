# sk-align

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/sk-align.svg)](https://pypi.org/project/sk-align/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97-Model_on_Hub-yellow.svg)](https://huggingface.co/eist-edinburgh/nnet3_alignment_model)
[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen.svg)](#testing)

**Standalone forced alignment for Scottish Gaelic** — no Kaldi or PyKaldi dependency.

sk-align reimplements Kaldi's nnet3 forced-alignment pipeline entirely in
Python/NumPy/PyTorch, reading Kaldi model files directly. It produces
word-level timestamps at parity with PyKaldi while being easier to install
and deploy.

---

## Features

- **Zero Kaldi dependency** — pure Python reads Kaldi binary formats (`final.mdl`, `tree`, `L.fst`, etc.)
- **`from_pretrained()`** — one-line model download from Hugging Face Hub
- **MFCC extraction** — vectorised NumPy implementation matching Kaldi output
- **TDNN-F nnet3 inference** — full PyTorch reimplementation of the forward pass
- **k2 Viterbi decoder** — fast FSA-based decoding via `intersect_dense` + `shortest_path`
- **Word-level timestamps** — `[{"word": "hello", "start": 0.12, "end": 0.45}, ...]`
- **Parity-tested** — 55 tests verify numerical match against PyKaldi reference

## Installation

```bash
pip install sk-align              # core (numpy + scipy + torch + k2)
pip install sk-align[all]         # + huggingface_hub for from_pretrained()
```

Or install from source:

```bash
git clone https://github.com/your-org/sk-align.git
cd sk-align/sk-align
pip install -e ".[all]"           # editable with all extras
```

### Optional extras

| Extra     | Installs                         | Needed for                            |
| --------- | -------------------------------- | ------------------------------------- |
| `hub`     | `huggingface_hub>=0.20`          | `Aligner.from_pretrained()`           |
| `all`     | `huggingface_hub`                | Full end-to-end pipeline              |
| `test`    | `pytest` + `huggingface_hub`     | Running the test suite                |
| `dev`     | `test` extras + `ruff`           | Development                           |

## Quick start

```python
from sk_align import Aligner

# Download model from Hugging Face and load (cached after first call)
aligner = Aligner.from_pretrained()

# audio: float32 numpy array, 16 kHz, mono
timestamps = aligner.align(audio, ["cumaidh", "sinn", "a'", "dol"])
# [{"word": "cumaidh", "start": 0.33, "end": 0.72},
#  {"word": "sinn",    "start": 0.72, "end": 0.99},
#  ...]
```

### Loading a local model

```python
from sk_align import Aligner
from sk_align.nnet3_torch import TorchNnetScorer

scorer = TorchNnetScorer.from_model_file("/path/to/model/final.mdl")
aligner = Aligner.from_model_dir("/path/to/model", nnet_scorer=scorer)

timestamps = aligner.align(audio, words)
```

### Using pre-computed log-likelihoods

```python
import numpy as np
from sk_align import Aligner

aligner = Aligner.from_model_dir("/path/to/model")  # no scorer needed
loglikes = np.load("loglikes.npy")  # (num_frames, num_pdfs)

timestamps = aligner.align_with_loglikes(loglikes, words)
```

## Architecture

The alignment pipeline reimplements each stage of Kaldi's forced alignment
in pure Python:

```
Audio (float32, 16 kHz)
  │
  ▼
┌─────────────────────┐
│  MFCC Extraction    │  sk_align.mfcc        (NumPy, batch-vectorised)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Nnet3 Forward Pass │  sk_align.nnet3_torch  (PyTorch TDNN-F)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Graph Compilation  │  sk_align.graph        (L ∘ G, context expansion)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Viterbi Decoding   │  sk_align.k2_decoder   (k2 FSA intersection)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Word Alignment     │  sk_align.word_align   (boundary extraction)
└─────────────────────┘
          │
          ▼
  [{"word": "...", "start": 0.12, "end": 0.45}, ...]
```

### Modules

| Module                  | Description                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `sk_align.aligner`      | High-level `Aligner` class — main entry point                     |
| `sk_align.mfcc`         | MFCC feature extraction (batch NumPy, Kaldi-compatible)            |
| `sk_align.nnet3_model`  | Kaldi nnet3 binary parser                                          |
| `sk_align.nnet3_torch`  | PyTorch reimplementation of TDNN-F forward pass                    |
| `sk_align.fst`          | OpenFst binary format reader + FST representation                  |
| `sk_align.graph`        | Per-utterance decoding graph compiler (L ∘ G + context expansion)  |
| `sk_align.tree`         | Kaldi `ContextDependency` tree reader                              |
| `sk_align.transition_model` | Kaldi `TransitionModel` reader                                 |
| `sk_align.k2_decoder`   | k2-based Viterbi decoder                                          |
| `sk_align.word_align`   | Word boundary extraction + timestamp conversion                    |
| `sk_align.kaldi_io`     | Low-level Kaldi binary I/O helpers                                 |

## Model

The default model is hosted at
[`eist-edinburgh/nnet3_alignment_model`](https://huggingface.co/eist-edinburgh/nnet3_alignment_model)
on Hugging Face Hub. It is a TDNN-F nnet3 alignment model (3456 PDFs) trained
for Scottish Gaelic.

**Expected model files:**

```
final.mdl           TransitionModel + nnet3 weights
tree                ContextDependency tree
L.fst               Lexicon FST (OpenFst binary)
words.txt           Word symbol table
disambig.int        Disambiguation symbol IDs
word_boundary.int   Phone word-boundary types
```

## Testing

The test suite verifies numerical parity with PyKaldi at every stage.

```bash
pip install -e ".[test]"
pytest                   # 49 tests — MFCC, I/O, graph, decoder, end-to-end parity
```

Tests include:
- **MFCC parity** — feature output matches Kaldi within floating-point tolerance
- **I/O round-trip** — all Kaldi binary readers produce correct data structures
- **Graph compilation** — decoding graphs match expected state/arc counts
- **Decoder parity** — k2 decoder alignment matches reference Viterbi output
- **End-to-end parity** — word timestamps match PyKaldi within 30ms

## Performance

Benchmark on a 5-second Scottish Gaelic utterance (25 words), CPU:

| Stage         | Time     | % of total |
| ------------- | -------- | ---------- |
| MFCC          | 25 ms    | 4%         |
| Nnet3 forward | 434 ms   | 75%        |
| Graph compile | 46 ms    | 8%         |
| k2 decode     | 72 ms    | 13%        |
| Word align    | <1 ms    | <1%        |
| **Total**     | **578 ms** | —        |

End-to-end throughput is at parity with PyKaldi (~560 ms per utterance).

## License

MIT
