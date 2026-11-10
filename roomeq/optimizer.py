"""Fit N parametric filters to bring a measured response onto a target.

Room-correction rules (the same principles Dirac Live / REW auto-EQ follow):

* Peaks are cut fully; dips are only partly filled. Boost is limited by a
  1/3-octave-smoothed response (so narrow nulls - which EQ can't fix and
  which move with head position - are ignored) and by a hard maximum.
* Never boost below the speaker's own low-frequency extension.
* Full precision below the Schroeder frequency (room modes), broad gentle
  tonal correction above it (low Q, lower weight).
* The composite curve, not just each filter, is boost-limited.

Fitting is greedy (pick the band that reduces error most, local fit) followed
by a joint non-linear least-squares refinement of all filters after each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .analysis import PPO
from .peq import Filter, eq_db, filter_db


@dataclass
class EQConfig:
    n_filters: int = 10
    min_freq: float | None = None  # None -> detect speaker LF extension
    max_freq: float = 16000.0
    schroeder: float = 300.0
    max_boost: float = 4.0
    max_cut: float = 15.0
    max_cut_hf: float = 6.0  # deepest cut above ~3x Schroeder
    max_boost_hf: float = 3.0  # largest boost above ~3x Schroeder
    min_q: float = 0.4
    max_q_lf: float = 10.0
    max_q_hf: float = 2.0
    shelves: bool = True
    fs: float = 48000.0


@dataclass
class EQResult:
    filters: list[Filter]
    f_lo: float
    f_hi: float
    desired: np.ndarray
    weights: np.ndarray
    eq: np.ndarray
    notes: list[str] = field(default_factory=list)


def _oct_smooth(x: np.ndarray, octaves: float) -> np.ndarray:
    k = max(1, int(PPO * octaves))
    pad = np.pad(x, k, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="same")[k:-k]


def detect_low_limit(grid, coarse, target=None, drop_db=6.0, start=250.0, floor=20.0,
                     dead_fraction=0.7) -> float:
    """Speaker LF extension: the highest frequency f whose octave below
    (f/2..f) is mostly (>= dead_fraction) more than `drop_db` under the
    speaker's own midrange level. A real roll-off kills the whole octave;
    a room null only takes a slice of it, so nulls don't fool it."""
    mid = (grid >= 200) & (grid <= 2000)
    dead = (coarse - float(np.median(coarse[mid]))) < -drop_db
    for i in range(int(np.searchsorted(grid, start)), 0, -1):
        f = grid[i]
        if f < 2 * 12:
            break
        band = (grid >= f / 2) & (grid <= f)
        if np.mean(dead[band]) >= dead_fraction:
            return float(max(floor, f))
    return floor


def detect_high_limit(grid, coarse, target, drop_db=6.0, start=4000.0) -> float:
    """Speaker HF extension: scan up from `start` until `drop_db` below target."""
    rel = _oct_smooth(coarse - target, 0.5)
    i0 = int(np.searchsorted(grid, start))
    for i in range(i0, len(grid)):
        if grid[i] > 20000:
            break
        if rel[i] < -drop_db:
            return float(grid[max(i - 1, 0)])
    return 20000.0


class _Problem:
    def __init__(self, grid, d, w, cap, f_lo, f_hi, cfg: EQConfig):
        self.grid, self.d, self.w, self.cap = grid, d, w, cap
        self.sw = np.sqrt(w) / np.sqrt(PPO)
        self.f_lo, self.f_hi, self.cfg = f_lo, f_hi, cfg
        self.inside = (grid >= f_lo) & (grid <= f_hi)
        self.floor = self.cut_limit(grid) + 1.0

    def qmax(self, fc, gain=-1.0):
        """Narrow cuts are fine (resonances are audible, cutting them is safe);
        narrow boosts are not (they chase position-dependent dips and ring)."""
        c = self.cfg
        lf = np.log2(fc)
        if gain < 0:
            return float(np.interp(lf, np.log2([c.schroeder, 10 * c.schroeder]),
                                   [c.max_q_lf, max(c.max_q_hf, 3.0)]))
        return float(np.interp(lf, np.log2([c.schroeder, 3 * c.schroeder]),
                               [min(c.max_q_lf, 5.0), c.max_q_hf]))

    def cut_limit(self, f):
        c = self.cfg
        return np.interp(np.log2(f), np.log2([c.schroeder, 3 * c.schroeder]),
                         [c.max_cut, c.max_cut_hf])

    def residuals(self, filters, base=0.0, others=()):
        eq = base + eq_db(filters, self.grid, self.cfg.fs)
        parts = [self.sw * (self.d - eq),
                 # limits apply to the combined curve, so stacking filters can't dodge them
                 10 * self.sw * np.maximum(0.0, eq - self.cap),
                 10 * self.sw * np.maximum(0.0, -self.floor - eq)]
        extra = []
        for f in filters:
            if f.kind == "PK":
                extra.append(3.0 * max(0.0, np.log2(f.q / self.qmax(f.fc, f.gain))))
            extra.append(0.01 * f.gain)
        # discourage two peaking filters doing one job (wastes a band)
        pk = [f for f in list(filters) + list(others) if f.kind == "PK"]
        for i in range(len(pk)):
            for j in range(i + 1, len(pk)):
                dist = abs(np.log2(pk[i].fc / pk[j].fc))
                extra.append(20.0 * max(0.0, 1 / 6 - dist))
        parts.append(np.array(extra))
        return np.concatenate(parts)

    def cost(self, filters):
        r = self.residuals(filters)
        return float(r @ r)

    # ---- parameter packing -------------------------------------------------
    def bounds_for(self, f: Filter):
        c = self.cfg
        if f.kind == "PK":
            lo = max(self.f_lo, f.fc / np.sqrt(2))
            hi = min(self.f_hi, f.fc * np.sqrt(2))
            return (lo, hi), (-c.max_cut, c.max_boost), (c.min_q, c.max_q_lf)
        if f.kind == "LSC":
            return (max(self.f_lo, 25.0), 300.0), (-c.max_cut, c.max_boost), (0.5, 1.0)
        return (1000.0, min(self.f_hi, 12000.0)), (-6.0, c.max_boost), (0.5, 1.0)

    def fit(self, filters: list[Filter], free: list[int], base=None, others=()) -> list[Filter]:
        filters = list(filters)
        fixed = [f for i, f in enumerate(filters) if i not in free]
        base = (eq_db(fixed, self.grid, self.cfg.fs) if base is None else base)
        fixed = fixed + list(others)
        lo, hi, x0 = [], [], []
        for i in free:
            f = filters[i]
            (flo, fhi), (glo, ghi), (qlo, qhi) = self.bounds_for(f)
            if flo >= fhi:
                fhi = flo * 1.01
            b_lo = [np.log2(flo), glo, np.log2(qlo)]
            b_hi = [np.log2(fhi), ghi, np.log2(qhi)]
            x = [np.log2(f.fc), f.gain, np.log2(f.q)]
            for k in range(3):
                span = b_hi[k] - b_lo[k]
                x[k] = float(np.clip(x[k], b_lo[k] + 1e-6 * span, b_hi[k] - 1e-6 * span))
            lo += b_lo
            hi += b_hi
            x0 += x

        kinds = [filters[i].kind for i in free]

        def unpack(x):
            return [Filter(k, 2 ** x[3 * j], x[3 * j + 1], 2 ** x[3 * j + 2])
                    for j, k in enumerate(kinds)]

        def fun(x):
            return self.residuals(unpack(x), base, fixed)

        sol = least_squares(fun, np.array(x0), bounds=(np.array(lo), np.array(hi)),
                            x_scale=np.tile([0.3, 2.0, 0.5], len(free)), max_nfev=200 * len(free))
        for j, f in zip(free, unpack(sol.x)):
            filters[j] = f
        return filters

    # ---- candidates --------------------------------------------------------
    def candidates(self, r, have_kinds) -> list[Filter]:
        k = max(1, PPO // 6)
        rs = np.convolve(r, np.ones(k) / k, mode="same")
        rs = np.where(self.inside, rs, 0.0)
        sign = np.sign(rs)
        cands = []
        i, n = 0, len(rs)
        while i < n:
            if sign[i] == 0 or abs(rs[i]) < 0.3:
                i += 1
                continue
            j = i
            while j + 1 < n and sign[j + 1] == sign[i]:
                j += 1
            seg = slice(i, j + 1)
            area = float(np.sum(self.w[seg] * r[seg] ** 2))
            ipk = i + int(np.argmax(np.abs(rs[seg])))
            g = rs[ipk]
            half = abs(g) / 2
            a = ipk
            while a > i and abs(rs[a - 1]) > half:
                a -= 1
            b = ipk
            while b < j and abs(rs[b + 1]) > half:
                b += 1
            bw = max((b - a + 1) / PPO, 1 / 12)
            q = float(np.clip(np.sqrt(2 ** bw) / (2 ** bw - 1), self.cfg.min_q, self.cfg.max_q_lf))
            fc = float(self.grid[ipk])
            q = min(q, self.qmax(fc, float(g)))
            cands.append((area, Filter("PK", fc, float(g), q)))
            i = j + 1
        cands.sort(key=lambda t: -t[0])
        out = [f for _, f in cands[:6]]
        if self.cfg.shelves:
            g = self.grid
            if "LSC" not in have_kinds and self.f_lo < 90:
                m = self.inside & (g < 100)
                if m.any():
                    out.append(Filter("LSC", 100.0, float(np.mean(r[m])), 0.7))
            if "HSC" not in have_kinds and self.f_hi > 4000:
                m = self.inside & (g > 3000)
                if m.any():
                    out.append(Filter("HSC", 4000.0, float(np.mean(r[m])), 0.7))
        return out


def design_eq(grid: np.ndarray, fine_db: np.ndarray, coarse_db: np.ndarray,
              target_db: np.ndarray, cfg: EQConfig) -> EQResult:
    """target_db must already be level-aligned to fine_db."""
    notes = []
    f_lo = cfg.min_freq or detect_low_limit(grid, coarse_db, target_db)
    if cfg.min_freq is None:
        notes.append(f"speaker low-frequency limit detected at {f_lo:.0f} Hz (no boost below)")
    hf_ext = detect_high_limit(grid, coarse_db, target_db)
    f_hi = min(cfg.max_freq, hf_ext / 2 ** (1 / 3))  # stay clear of the roll-off slope
    if f_hi < cfg.max_freq:
        notes.append(f"speaker high-frequency roll-off at {hf_ext:.0f} Hz - correcting up to "
                     f"{f_hi:.0f} Hz")

    d = target_db - fine_db
    boost_lim = np.interp(np.log2(grid), np.log2([cfg.schroeder, 3 * cfg.schroeder]),
                          [cfg.max_boost, min(cfg.max_boost, cfg.max_boost_hf)])
    boost_room = np.clip(target_db - coarse_db, 0, boost_lim)
    cut_lim = np.interp(np.log2(grid), np.log2([cfg.schroeder, 3 * cfg.schroeder]),
                        [cfg.max_cut, cfg.max_cut_hf])
    d = np.where(d > 0, np.minimum(d, boost_room), np.maximum(d, -cut_lim))
    inside = (grid >= f_lo) & (grid <= f_hi)
    d = np.where(inside, d, 0.0)

    lg = np.log2(grid)
    w = np.interp(lg, np.log2([cfg.schroeder, 4 * cfg.schroeder, 8000.0]), [1.0, 0.45, 0.3])
    w = np.where(inside, w, 0.02)  # outside: don't care, only the boost cap applies
    w = np.where(grid < 15, 0.0, w)
    cap = np.where(inside, boost_lim, 0.5)

    prob = _Problem(grid, d, w, cap, f_lo, f_hi, cfg)
    filters: list[Filter] = []
    for _round in range(3):
        filters = _greedy_fill(prob, filters, d)
        pruned = _prune(prob, filters)
        if len(pruned) == len(filters):
            break
        filters = prob.fit(pruned, list(range(len(pruned)))) if pruned else pruned
    filters = _prune(prob, filters)

    # round to what the UIs accept, re-polish gains with rounded Fc/Q
    filters = sorted((f.rounded() for f in filters), key=lambda f: f.fc)
    eq = eq_db(filters, grid, cfg.fs)
    return EQResult(filters, f_lo, f_hi, d, w, eq, notes)


def _greedy_fill(prob: _Problem, filters: list[Filter], d: np.ndarray) -> list[Filter]:
    cfg, grid = prob.cfg, prob.grid
    filters = list(filters)
    cost = prob.cost(filters)
    while len(filters) < cfg.n_filters:
        base = eq_db(filters, grid, cfg.fs)
        best, best_cost = None, cost
        for cand in prob.candidates(d - base, {f.kind for f in filters}):
            f = prob.fit([cand], [0], base=base, others=filters)[0]
            c = prob.cost(filters + [f])
            if c < best_cost:
                best, best_cost = f, c
        if best is None or best_cost > cost * 0.995 - 1e-6:
            break
        filters.append(best)
        filters = prob.fit(filters, list(range(len(filters))))
        cost = prob.cost(filters)
    return filters


def _prune(prob: _Problem, filters: list[Filter], rel: float = 0.01) -> list[Filter]:
    """Drop bands that contribute (almost) nothing so they can be reused."""
    filters = [f for f in filters if abs(f.gain) >= 0.25]
    while filters:
        full = prob.cost(filters)
        costs = [prob.cost(filters[:i] + filters[i + 1:]) for i in range(len(filters))]
        i = int(np.argmin(costs))
        if costs[i] > full * (1 + rel) + 1e-6:
            break
        filters = filters[:i] + filters[i + 1:]
    return filters


def rms_error(grid, a, b, f_lo, f_hi) -> float:
    m = (grid >= f_lo) & (grid <= f_hi)
    e = (a - b)[m]
    return float(np.sqrt(np.mean((e - np.mean(e)) ** 2)))


def max_boost(filters: list[Filter], fs: float = 48000.0) -> float:
    from .analysis import log_grid
    g = log_grid(15, 20000, 96)
    return float(max(0.0, np.max(eq_db(filters, g, fs)))) if filters else 0.0


__all__ = ["EQConfig", "EQResult", "design_eq", "rms_error", "max_boost", "filter_db"]
