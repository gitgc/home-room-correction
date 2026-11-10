"""Exponential sine sweep measurement (Farina method).

One stimulus plays a sweep on each speaker in turn (L, R, ... ) inside a single
continuous recording, so the relative arrival time of L and R is exact and can
be used for delay alignment. Deconvolving the recording with the sweep yields
one impulse response per sweep; harmonic distortion lands *before* each main
impulse and is windowed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter1d


@dataclass
class SweepParams:
    fs: int = 48000
    f1: float = 10.0
    f2: float = 22000.0
    duration: float = 5.5
    level_db: float = -12.0
    lead_s: float = 0.5
    gap_s: float = 1.0
    tail_s: float = 2.0

    @property
    def harmonic_lead_s(self) -> float:
        """How far before the main IR the 2nd-harmonic IR appears."""
        return self.duration * np.log(2) / np.log(self.f2 / self.f1)


def exp_sweep(p: SweepParams) -> np.ndarray:
    n = int(round(p.duration * p.fs))
    t = np.arange(n) / p.fs
    T = n / p.fs
    L = T / np.log(p.f2 / p.f1)
    x = np.sin(2 * np.pi * p.f1 * L * (np.exp(t / L) - 1))
    fi, fo = int(0.05 * p.fs), int(0.005 * p.fs)
    x[:fi] *= 0.5 * (1 - np.cos(np.pi * np.arange(fi) / fi))
    x[-fo:] *= 0.5 * (1 + np.cos(np.pi * np.arange(fo) / fo))
    return 10 ** (p.level_db / 20) * x


def build_stimulus(sweep: np.ndarray, p: SweepParams, sequence: list[str],
                   channel_map: dict[str, int]) -> tuple[np.ndarray, list[int]]:
    """Multichannel stimulus; returns (signal[n, ch], start sample of each sweep)."""
    lead, gap, tail = (int(x * p.fs) for x in (p.lead_s, p.gap_s, p.tail_s))
    step = len(sweep) + gap
    starts = [lead + k * step for k in range(len(sequence))]
    n = starts[-1] + len(sweep) + tail
    out = np.zeros((n, max(channel_map.values()) + 1), dtype=np.float32)
    for s, ch in zip(starts, sequence):
        out[s:s + len(sweep), channel_map[ch]] = sweep
    return out, starts


def deconvolve(rec: np.ndarray, sweep: np.ndarray, p: SweepParams) -> np.ndarray:
    """Regularised spectral division; result is the (circular) system response."""
    n = 1 << int(np.ceil(np.log2(len(rec) + len(sweep))))
    S = np.fft.rfft(sweep, n)
    R = np.fft.rfft(rec, n)
    f = np.fft.rfftfreq(n, 1 / p.fs)
    s2 = np.abs(S) ** 2
    lf = np.log2(np.maximum(f, 1e-3))
    lo, hi = np.log2(p.f1 * 1.25), np.log2(p.f2 * 0.95)
    band = np.clip(np.minimum((lf - (lo - 1 / 3)) * 3, ((hi + 1 / 3) - lf) * 3), 0, 1)
    eps = s2.max() * (1e-8 * band + 1e-1 * (1 - band))
    return np.fft.irfft(R * np.conj(S) / (s2 + eps), n)


@dataclass
class SweepIR:
    channel: str
    seg: np.ndarray  # IR excerpt: PRE_S before the peak to POST_S after
    peak: int  # index of the main peak inside seg
    toa: float  # peak position in the recording timeline (samples, sub-sample)
    pnr_db: float  # peak-to-noise ratio of the impulse response
    polarity: int  # sign of the main peak


@dataclass
class RecordingAnalysis:
    irs: list[SweepIR]
    rec_peak_dbfs: float
    starts: list[int]
    warnings: list[str] = field(default_factory=list)


PRE_S = 0.05
POST_S = 1.0


def analyse_recording(rec: np.ndarray, sweep: np.ndarray, p: SweepParams, starts: list[int],
                      sequence: list[str], max_latency_s: float = 3.0) -> RecordingAnalysis:
    fs = p.fs
    h = deconvolve(rec, sweep, p)
    env = np.abs(h)
    me = maximum_filter1d(env, size=int(0.005 * fs) | 1)
    o_max = max(1, min(int(max_latency_s * fs), len(rec) - starts[-1] - len(sweep) // 2))
    score = sum(me[s:s + o_max] for s in starts)
    o = int(np.argmax(score))

    pre, post, d = int(PRE_S * fs), int(POST_S * fs), int(0.005 * fs)
    n_lo = int(0.8 * p.harmonic_lead_s * fs)
    n_hi = int(0.06 * fs)
    irs = []
    for s, ch in zip(starts, sequence):
        c = o + s
        a = max(0, c - d)
        pk = a + int(np.argmax(env[a:c + d]))
        seg = np.take(h, np.arange(pk - pre, pk + post), mode="wrap").astype(np.float32)
        noise = np.take(h, np.arange(pk - n_lo, pk - n_hi), mode="wrap")
        pnr = 10 * np.log10(h[pk] ** 2 / max(np.mean(noise ** 2), 1e-30))
        # parabolic sub-sample peak
        y0, y1, y2 = env[pk - 1], env[pk], env[pk + 1]
        den = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        irs.append(SweepIR(ch, seg, pre, pk + float(np.clip(frac, -0.5, 0.5)), float(pnr),
                           1 if h[pk] >= 0 else -1))

    peak = float(np.max(np.abs(rec))) if len(rec) else 0.0
    res = RecordingAnalysis(irs, 20 * np.log10(max(peak, 1e-12)), starts)
    if res.rec_peak_dbfs > -1.0:
        res.warnings.append("mic input is clipping - turn the volume down")
    elif res.rec_peak_dbfs < -45:
        res.warnings.append("mic level is very low - turn the volume up")
    for ir in irs:
        if ir.pnr_db < 40:
            res.warnings.append(f"{ir.channel}: low signal-to-noise ({ir.pnr_db:.0f} dB) - "
                                "turn up or reduce background noise")
    return res


def timing_from_sequence(irs: list[SweepIR], starts: list[int], fs: int) -> dict:
    """Relative L/R arrival using an L,R,L sequence (second L cancels clock drift)."""
    idx = {ch: [i for i, ir in enumerate(irs) if ir.channel == ch] for ch in ("L", "R")}
    if not idx["L"] or not idx["R"]:
        return {}
    i0, i1 = idx["L"][0], idx["R"][0]
    eps = 0.0
    if len(idx["L"]) > 1:
        i2 = idx["L"][1]
        eps = (irs[i2].toa - irs[i0].toa - (starts[i2] - starts[i0])) / (starts[i2] - starts[i0])
        if abs(eps) > 500e-6:
            eps = 0.0
    tau = (irs[i1].toa - irs[i0].toa) - (starts[i1] - starts[i0]) * (1 + eps)
    return {
        "r_minus_l_ms": 1000 * tau / fs,
        "distance_diff_cm": 100 * tau / fs * 343.0,
        "drift_ppm": eps * 1e6,
        "polarity": {"L": irs[i0].polarity, "R": irs[i1].polarity},
    }
