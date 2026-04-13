"""
Low-level readers for Kaldi binary / text file formats.

Handles the Kaldi binary marker (\0B), basic types, tokens,
integer vectors, float vectors, and matrices.

Reference: kaldi/src/base/io-funcs-inl.h, kaldi/src/matrix/kaldi-vector.cc
"""

from __future__ import annotations

import struct
from io import BufferedReader
from pathlib import Path
from typing import BinaryIO

import numpy as np


# ---------------------------------------------------------------------------
# Binary header
# ---------------------------------------------------------------------------

def read_binary_header(f: BinaryIO) -> bool:
    """Read the Kaldi binary marker.  Returns *True* if binary mode."""
    peek = f.read(2)
    if peek == b"\x00B":
        return True
    raise ValueError(f"Expected Kaldi binary header '\\0B', got {peek!r}")


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def read_token(f: BinaryIO) -> str:
    """Read a whitespace-terminated token (binary mode)."""
    chars: list[bytes] = []
    while True:
        c = f.read(1)
        if not c:
            break
        if c in (b" ", b"\t", b"\n", b"\r"):
            break
        chars.append(c)
    return b"".join(chars).decode("ascii")


def expect_token(f: BinaryIO, expected: str) -> None:
    """Read a token and assert it matches *expected*."""
    tok = read_token(f)
    if tok != expected:
        raise ValueError(f"Expected token {expected!r}, got {tok!r}")


def peek_char(f: BinaryIO) -> str:
    """Peek at the next non-whitespace character without consuming it."""
    while True:
        c = f.read(1)
        if not c:
            raise EOFError("Unexpected end of file")
        if c not in (b" ", b"\t", b"\n", b"\r"):
            f.seek(-1, 1)
            return c.decode("ascii")


# ---------------------------------------------------------------------------
# Basic types
# ---------------------------------------------------------------------------

def read_int32(f: BinaryIO) -> int:
    """Read a Kaldi binary int32 (1-byte signed length prefix + 4 bytes LE)."""
    length_byte = struct.unpack("b", f.read(1))[0]
    # Positive length → signed int, negative → unsigned int
    nbytes = abs(length_byte)
    if nbytes != 4:
        raise ValueError(f"Expected 4-byte int, got {nbytes}-byte (prefix={length_byte})")
    data = f.read(4)
    if length_byte > 0:
        return struct.unpack("<i", data)[0]
    else:
        return struct.unpack("<I", data)[0]


def read_int32_raw(f: BinaryIO) -> int:
    """Read a raw little-endian int32 (no Kaldi prefix)."""
    return struct.unpack("<i", f.read(4))[0]


def read_uint32_raw(f: BinaryIO) -> int:
    """Read a raw little-endian uint32 (no Kaldi prefix)."""
    return struct.unpack("<I", f.read(4))[0]


def read_uint64_raw(f: BinaryIO) -> int:
    """Read a raw little-endian uint64."""
    return struct.unpack("<Q", f.read(8))[0]


def read_int64_raw(f: BinaryIO) -> int:
    """Read a raw little-endian int64."""
    return struct.unpack("<q", f.read(8))[0]


def read_float32(f: BinaryIO) -> float:
    """Read a Kaldi binary float (1-byte size prefix + 4 bytes IEEE 754)."""
    nbytes = struct.unpack("b", f.read(1))[0]
    if nbytes == 4:
        return struct.unpack("<f", f.read(4))[0]
    elif nbytes == 8:
        return struct.unpack("<d", f.read(8))[0]
    else:
        raise ValueError(f"Unexpected float size: {nbytes}")


def read_bool(f: BinaryIO) -> bool:
    """Read a Kaldi bool ('T' or 'F')."""
    c = f.read(1)
    if c == b"T":
        return True
    elif c == b"F":
        return False
    raise ValueError(f"Expected 'T' or 'F', got {c!r}")


# ---------------------------------------------------------------------------
# Integer vectors
# ---------------------------------------------------------------------------

def read_integer_vector(f: BinaryIO) -> list[int]:
    """Read a Kaldi binary integer vector (int32 elements)."""
    elem_size = struct.unpack("b", f.read(1))[0]
    if elem_size != 4:
        raise ValueError(f"Expected elem_size 4, got {elem_size}")
    size = struct.unpack("<i", f.read(4))[0]
    data = f.read(4 * size)
    return list(struct.unpack(f"<{size}i", data))


# ---------------------------------------------------------------------------
# Float vectors (Kaldi Vector<float>)
# ---------------------------------------------------------------------------

def read_float_vector(f: BinaryIO) -> np.ndarray:
    """Read a Kaldi binary vector (``FV`` or ``DV`` token)."""
    tok = read_token(f)
    if tok == "FV":
        size = read_int32(f)
        data = f.read(4 * size)
        return np.frombuffer(data, dtype=np.float32).copy()
    elif tok == "DV":
        size = read_int32(f)
        data = f.read(8 * size)
        return np.frombuffer(data, dtype=np.float64).copy().astype(np.float32)
    else:
        raise ValueError(f"Expected 'FV' or 'DV', got {tok!r}")


# ---------------------------------------------------------------------------
# Float matrices (Kaldi Matrix<float>)
# ---------------------------------------------------------------------------

def read_float_matrix(f: BinaryIO) -> np.ndarray:
    """Read a Kaldi binary matrix (``FM`` or ``DM`` token)."""
    tok = read_token(f)
    if tok == "FM":
        rows = read_int32(f)
        cols = read_int32(f)
        data = f.read(4 * rows * cols)
        return np.frombuffer(data, dtype=np.float32).copy().reshape(rows, cols)
    elif tok == "DM":
        rows = read_int32(f)
        cols = read_int32(f)
        data = f.read(8 * rows * cols)
        return np.frombuffer(data, dtype=np.float64).copy().astype(np.float32).reshape(rows, cols)
    else:
        raise ValueError(f"Expected 'FM' or 'DM', got {tok!r}")


# ---------------------------------------------------------------------------
# Text file readers
# ---------------------------------------------------------------------------

def read_symbol_table(path: str | Path) -> dict[str, int]:
    """Read a Kaldi ``words.txt`` symbol table → {symbol: int_id}."""
    table: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                table[parts[0]] = int(parts[1])
    return table


def read_symbol_table_reverse(path: str | Path) -> dict[int, str]:
    """Read a Kaldi ``words.txt`` symbol table → {int_id: symbol}."""
    table: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                table[int(parts[1])] = parts[0]
    return table


def read_disambig_symbols(path: str | Path) -> list[int]:
    """Read disambiguation symbol IDs from ``disambig.int``."""
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def read_word_boundary(path: str | Path) -> dict[int, str]:
    """Read ``word_boundary.int`` → {phone_id: type_str}.

    Type strings: ``nonword``, ``begin``, ``end``, ``singleton``, ``internal``.
    """
    table: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                table[int(parts[0])] = parts[1]
    return table
