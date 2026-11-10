"""Biquad / parametric EQ maths (RBJ Audio EQ Cookbook).

These are the same filter definitions used by Equalizer APO, AutoEq-format
importers (SoundSource) and miniDSP, so a filter described here as
(type, Fc, gain, Q) sounds the same in all of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KINDS = ("PK", "LSC", "HSC")


@dataclass
class Filter:
    kind: str  # "PK" peaking, "LSC" low shelf, "HSC" high shelf
    fc: float  # Hz
    gain: float  # dB
    q: float

    def rounded(self) -> "Filter":
        fc = round(self.fc, 1) if self.fc < 1000 else float(round(self.fc))
        return Filter(self.kind, fc, round(self.gain, 1), round(self.q, 2))

    def to_dict(self) -> dict:
        return {"type": self.kind, "fc": self.fc, "gain": self.gain, "q": self.q}


def biquad(flt: Filter, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised (b, a) with a[0] == 1."""
    A = 10.0 ** (flt.gain / 40.0)
    w0 = 2.0 * np.pi * flt.fc / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * flt.q)
    if flt.kind == "PK":
        b0, b1, b2 = 1 + alpha * A, -2 * cw, 1 - alpha * A
        a0, a1, a2 = 1 + alpha / A, -2 * cw, 1 - alpha / A
    elif flt.kind == "LSC":
        sa = 2 * np.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * cw + sa)
        b1 = 2 * A * ((A - 1) - (A + 1) * cw)
        b2 = A * ((A + 1) - (A - 1) * cw - sa)
        a0 = (A + 1) + (A - 1) * cw + sa
        a1 = -2 * ((A - 1) + (A + 1) * cw)
        a2 = (A + 1) + (A - 1) * cw - sa
    elif flt.kind == "HSC":
        sa = 2 * np.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cw + sa)
        b1 = -2 * A * ((A - 1) + (A + 1) * cw)
        b2 = A * ((A + 1) + (A - 1) * cw - sa)
        a0 = (A + 1) - (A - 1) * cw + sa
        a1 = 2 * ((A - 1) - (A + 1) * cw)
        a2 = (A + 1) - (A - 1) * cw - sa
    else:
        raise ValueError(f"unknown filter type {flt.kind}")
    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def filter_db(flt: Filter, freqs: np.ndarray, fs: float) -> np.ndarray:
    b, a = biquad(flt, fs)
    z = np.exp(-1j * 2 * np.pi * np.asarray(freqs) / fs)
    h = (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))


def eq_db(filters: list[Filter], freqs: np.ndarray, fs: float) -> np.ndarray:
    out = np.zeros(len(freqs))
    for f in filters:
        out += filter_db(f, freqs, fs)
    return out
