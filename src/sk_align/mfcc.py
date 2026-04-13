"""
MFCC feature extraction — pure Python/NumPy reimplementation of Kaldi's MFCC.

Matches the output of ``kaldi::Mfcc`` with the options used by the
Scottish Gaelic forced aligner:

    samp_freq       = 16000
    frame_length_ms = 25.0   (400 samples)
    frame_shift_ms  = 10.0   (160 samples)
    num_mel_bins    = 40
    num_ceps        = 40
    low_freq        = 20
    high_freq       = -400   (= Nyquist - 400 = 7600 Hz)
    use_energy      = False
    preemph_coeff   = 0.97
    dither          = 1.0
    window_type     = "povey"
    cepstral_lifter = 22.0
    snip_edges      = True
    raw_energy      = True
    round_to_power_of_two = True

Reference: kaldi/src/feat/feature-mfcc.cc, feature-window.cc, mel-computations.cc
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MfccOptions:
    """Configuration matching ``kaldi::MfccOptions``."""

    sample_freq: float = 16000.0
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    dither: float = 1.0
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    round_to_power_of_two: bool = True
    snip_edges: bool = True
    raw_energy: bool = True

    num_mel_bins: int = 40
    low_freq: float = 20.0
    high_freq: float = -400.0  # negative → Nyquist + value

    num_ceps: int = 40
    use_energy: bool = False
    energy_floor: float = 0.0
    cepstral_lifter: float = 22.0
    htk_compat: bool = False

    # Reproducibility: set to 0 to disable dithering randomness
    seed: int | None = None


# ---------------------------------------------------------------------------
# Mel-scale helpers
# ---------------------------------------------------------------------------

def _hz_to_mel(freq: float) -> float:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def _compute_mel_banks(
    num_bins: int,
    fft_len: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    """Build triangular mel filterbank matrix ``(num_bins, fft_len // 2 + 1)``.

    Matches ``kaldi::MelBanks``.
    """
    nyquist = sample_freq / 2.0
    if high_freq <= 0:
        high_freq = nyquist + high_freq
    assert low_freq >= 0 and high_freq > low_freq and high_freq <= nyquist

    num_fft_bins = fft_len // 2 + 1  # number of non-negative frequency bins

    mel_low = _hz_to_mel(low_freq)
    mel_high = _hz_to_mel(high_freq)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)

    # Center frequencies in mel and Hz
    center_mels = mel_low + mel_delta * np.arange(num_bins + 2)
    center_freqs = np.array([_mel_to_hz(m) for m in center_mels])

    # Map each FFT bin to its frequency
    fft_freqs = np.arange(num_fft_bins) * sample_freq / fft_len

    banks = np.zeros((num_bins, num_fft_bins), dtype=np.float32)
    for i in range(num_bins):
        left = center_freqs[i]
        center = center_freqs[i + 1]
        right = center_freqs[i + 2]
        # Vectorised over FFT bins
        mask_up = (fft_freqs > left) & (fft_freqs <= center)
        mask_down = (fft_freqs > center) & (fft_freqs < right)
        banks[i, mask_up] = (fft_freqs[mask_up] - left) / (center - left)
        banks[i, mask_down] = (right - fft_freqs[mask_down]) / (right - center)
    return banks


# ---------------------------------------------------------------------------
# Window functions
# ---------------------------------------------------------------------------

def _make_window(window_type: str, length: int) -> np.ndarray:
    """Create a window function matching Kaldi's ``FeatureWindowFunction``."""
    n = np.arange(length, dtype=np.float64)
    if window_type == "hamming":
        w = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (length - 1))
    elif window_type == "hanning":
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
    elif window_type == "povey":
        # Kaldi "povey" = Hann raised to 0.85
        w = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))) ** 0.85
    elif window_type == "rectangular":
        w = np.ones(length, dtype=np.float64)
    elif window_type == "blackman":
        w = (
            0.42
            - 0.5 * np.cos(2.0 * np.pi * n / (length - 1))
            + 0.08 * np.cos(4.0 * np.pi * n / (length - 1))
        )
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    return w.astype(np.float64)


# ---------------------------------------------------------------------------
# DCT matrix
# ---------------------------------------------------------------------------

def _make_dct_matrix(num_ceps: int, num_bins: int) -> np.ndarray:
    """Type-II DCT matrix matching Kaldi (``ComputeDctMatrix``).

    ``dct[i, j] = sqrt(2/N) * cos(pi/N * (j + 0.5) * i)``
    with the zeroth row scaled by ``sqrt(1/N)`` instead.
    """
    N = num_bins
    # Vectorised outer product: i-vector × j-vector
    i_vec = np.arange(num_ceps, dtype=np.float64)[:, None]   # (C, 1)
    j_vec = np.arange(N, dtype=np.float64)[None, :] + 0.5    # (1, N)
    dct = np.cos((np.pi / N) * j_vec * i_vec)                # (C, N)
    # Scale rows
    scale = np.full(num_ceps, np.sqrt(2.0 / N))
    scale[0] = np.sqrt(1.0 / N)
    dct *= scale[:, None]
    return dct


# ---------------------------------------------------------------------------
# Cepstral liftering
# ---------------------------------------------------------------------------

def _make_lifter_coeffs(num_ceps: int, cepstral_lifter: float) -> np.ndarray:
    """Compute cepstral liftering coefficients.

    ``lifter[i] = 1 + 0.5 * Q * sin(pi * i / Q)``  where Q = cepstral_lifter.
    """
    if cepstral_lifter == 0.0:
        return np.ones(num_ceps, dtype=np.float64)
    Q = cepstral_lifter
    return 1.0 + 0.5 * Q * np.sin(np.pi * np.arange(num_ceps) / Q)


# ---------------------------------------------------------------------------
# Main MFCC computation
# ---------------------------------------------------------------------------

def compute_mfcc(
    waveform: np.ndarray,
    opts: MfccOptions | None = None,
) -> np.ndarray:
    """Compute MFCC features from a waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Audio samples, float32 or float64. If coming from the aligner, these
        are already multiplied by 2**15 (Kaldi convention).
    opts : MfccOptions, optional
        Feature extraction options (defaults match the SG aligner config).

    Returns
    -------
    np.ndarray
        Feature matrix of shape ``(num_frames, num_ceps)`` in float32.
    """
    if opts is None:
        opts = MfccOptions()

    wav = np.asarray(waveform, dtype=np.float64)

    sr = opts.sample_freq
    frame_length = int(round(opts.frame_length_ms * sr / 1000.0))
    frame_shift = int(round(opts.frame_shift_ms * sr / 1000.0))

    # Number of frames (snip_edges=True)
    if opts.snip_edges:
        if len(wav) < frame_length:
            return np.zeros((0, opts.num_ceps), dtype=np.float32)
        num_frames = 1 + (len(wav) - frame_length) // frame_shift
    else:
        num_frames = max(1, (len(wav) + frame_shift // 2) // frame_shift)

    # Padded FFT length
    if opts.round_to_power_of_two:
        fft_len = 1
        while fft_len < frame_length:
            fft_len *= 2
    else:
        fft_len = frame_length

    # Pre-compute reusable matrices
    window = _make_window(opts.window_type, frame_length)
    mel_banks = _compute_mel_banks(
        opts.num_mel_bins, fft_len, sr, opts.low_freq, opts.high_freq
    ).astype(np.float64)
    dct_matrix = _make_dct_matrix(opts.num_ceps, opts.num_mel_bins)
    lifter = _make_lifter_coeffs(opts.num_ceps, opts.cepstral_lifter)

    rng = np.random.RandomState(opts.seed) if opts.seed is not None else np.random.RandomState()

    # ---- Batch frame extraction ----
    if opts.snip_edges:
        # Build all frames at once using stride tricks
        starts = np.arange(num_frames) * frame_shift
        indices = starts[:, None] + np.arange(frame_length)[None, :]  # (F, L)
        frames = wav[indices].copy()  # (F, L)
    else:
        half = frame_length // 2
        starts = np.arange(num_frames) * frame_shift - half
        indices = starts[:, None] + np.arange(frame_length)[None, :]
        indices = np.clip(indices, 0, len(wav) - 1)
        frames = wav[indices].copy()

    # ---- Dithering (batch) ----
    if opts.dither != 0.0:
        frames += rng.standard_normal(frames.shape) * opts.dither

    # ---- DC offset removal (batch) ----
    frames -= frames.mean(axis=1, keepdims=True)

    # ---- Raw log energy (batch, before pre-emphasis) ----
    if opts.raw_energy:
        energy = np.sum(frames ** 2, axis=1)
        if opts.energy_floor > 0.0:
            np.maximum(energy, opts.energy_floor, out=energy)
        log_energy = np.log(np.maximum(energy, np.finfo(np.float64).tiny))

    # ---- Pre-emphasis (batch) ----
    if opts.preemph_coeff != 0.0:
        frames[:, 1:] -= opts.preemph_coeff * frames[:, :-1].copy()
        frames[:, 0] *= (1.0 - opts.preemph_coeff)

    # ---- Windowing (batch) ----
    frames *= window[None, :]  # broadcast (F, L) * (1, L)

    # ---- Zero-pad + FFT (batch) ----
    if fft_len > frame_length:
        padded = np.zeros((num_frames, fft_len), dtype=np.float64)
        padded[:, :frame_length] = frames
    else:
        padded = frames

    spectra = np.fft.rfft(padded, axis=1)        # (F, fft_len//2+1)
    power_spectra = np.abs(spectra) ** 2          # (F, fft_len//2+1)

    # ---- Mel filterbank (batch matmul) ----
    mel_energies = power_spectra @ mel_banks.T    # (F, num_mel_bins)
    np.maximum(mel_energies, np.finfo(np.float64).tiny, out=mel_energies)
    np.log(mel_energies, out=mel_energies)

    # ---- DCT → cepstral coefficients (batch matmul) ----
    features = mel_energies @ dct_matrix.T        # (F, num_ceps)

    # ---- Cepstral liftering (batch) ----
    features *= lifter[None, :]

    # ---- Energy substitution ----
    if opts.use_energy:
        if opts.htk_compat:
            features = np.roll(features, 1, axis=1)
        features[:, 0] = log_energy

    return features.astype(np.float32)
