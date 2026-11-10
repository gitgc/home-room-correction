"""Impulse responses -> spatially averaged, smoothed frequency response."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import MicCalibration

PPO = 48  # analysis grid points per octave


def log_grid(f_lo: float = 10.0, f_hi: float = 22000.0, ppo: int = PPO) -> np.ndarray:
    n = int(np.floor(np.log2(f_hi / f_lo) * ppo)) + 1
    return f_lo * 2.0 ** (np.arange(n) / ppo)


def variable_width(grid: np.ndarray, lf: float = 1 / 24, hf: float = 1 / 3,
                   f_a: float = 100.0, f_b: float = 1000.0) -> np.ndarray:
    """Psychoacoustic-style variable smoothing: 1/24 oct in the bass (room modes),
    widening to 1/3 oct from 1 kHz up (what the ear resolves tonally)."""
    t = np.clip(np.log2(grid / f_a) / np.log2(f_b / f_a), 0, 1)
    return lf * (hf / lf) ** t


def ir_power_spectrum(seg: np.ndarray, peak: int, fs: int, window_ms: float,
                      pre_ms: float = 5.0, nfft: int = 1 << 18) -> tuple[np.ndarray, np.ndarray]:
    pre = int(pre_ms * fs / 1000)
    post = int(window_ms * fs / 1000)
    a, b = max(0, peak - pre), min(len(seg), peak + post)
    x = np.asarray(seg[a:b], dtype=float).copy()
    w = np.ones(len(x))
    rise = peak - a
    if rise > 0:
        w[:rise] = 0.5 * (1 - np.cos(np.pi * np.arange(rise) / rise))
    fall = max(1, int(0.2 * (b - peak)))
    w[-fall:] *= 0.5 * (1 + np.cos(np.pi * np.arange(fall) / fall))
    nfft = max(nfft, 1 << int(np.ceil(np.log2(len(x)))))
    X = np.fft.rfft(x * w, nfft)
    return np.fft.rfftfreq(nfft, 1 / fs), np.abs(X) ** 2


def smooth_power(freqs: np.ndarray, power: np.ndarray, grid: np.ndarray,
                 width_oct: np.ndarray | float) -> np.ndarray:
    """Fractional-octave power average onto a log grid (exact box integral)."""
    df = freqs[1] - freqs[0]
    edges = np.concatenate([[freqs[0] - df / 2], freqs + df / 2])
    c = np.concatenate([[0.0], np.cumsum(power) * df])
    half = 2.0 ** (np.asarray(width_oct) / 2)
    lo, hi = grid / half, grid * half
    return (np.interp(hi, edges, c) - np.interp(lo, edges, c)) / (hi - lo)


@dataclass
class ChannelResponse:
    grid: np.ndarray
    fine_db: np.ndarray  # spatial average, variable smoothing - used for cuts
    coarse_db: np.ndarray  # spatial average, 1/3 oct - used to limit boosts
    positions_db: list[np.ndarray]  # each position, variable smoothing (for plots)
    freqs: np.ndarray  # raw averaged power spectrum (for combining channels)
    power: np.ndarray


def _db(p):
    return 10 * np.log10(np.maximum(p, 1e-30))


def channel_response(irs: list[tuple[np.ndarray, int, float, int]], fs: int,
                     cal: MicCalibration, grid: np.ndarray, window_ms: float) -> ChannelResponse:
    """irs: list of (segment, peak_index, weight, position_index)."""
    p_sum, w_sum = None, 0.0
    per_pos: dict[int, list] = {}
    freqs = None
    for seg, peak, w, pos in irs:
        freqs, P = ir_power_spectrum(seg, peak, fs, window_ms)
        p_sum = w * P if p_sum is None else p_sum + w * P
        w_sum += w
        per_pos.setdefault(pos, []).append(P)
    power = p_sum / w_sum
    return combine_response(freqs, power, grid, cal,
                            [np.mean(v, axis=0) for _, v in sorted(per_pos.items())])


def combine_response(freqs, power, grid, cal, position_powers=()) -> ChannelResponse:
    width = variable_width(grid)
    c = cal.correction_db(grid)
    return ChannelResponse(
        grid=grid,
        fine_db=_db(smooth_power(freqs, power, grid, width)) - c,
        coarse_db=_db(smooth_power(freqs, power, grid, 1 / 3)) - c,
        positions_db=[_db(smooth_power(freqs, P, grid, width)) - c for P in position_powers],
        freqs=freqs,
        power=power,
    )


def band_level(grid: np.ndarray, db: np.ndarray, f_lo: float = 300, f_hi: float = 3000) -> float:
    m = (grid >= f_lo) & (grid <= f_hi)
    return float(np.mean(db[m]))
