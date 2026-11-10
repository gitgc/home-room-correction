"""Verify a correction: re-measure with the EQ switched on and compare the
measured change against what the EQ should do.

Comparing the *change* (new response minus original response, same mic
positions) rather than the absolute response against the target cancels the
room and speakers out, so what's left is the EQ as actually applied by
Equalizer APO / SoundSource / miniDSP. Known set-up mistakes leave
recognisable fingerprints, so each is tested as an alternative explanation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .analysis import band_level, log_grid, smooth_power, variable_width
from .calibration import MicCalibration
from .peq import Filter, eq_db
from .pipeline import CHANNELS, Measurement, channel_responses

PASS_DB = 1.5  # shape error below this: correction is working as designed
WARN_DB = 3.0  # above this: something is wrong
MODES = ("channel", "combined")


@dataclass
class ChannelCheck:
    name: str
    f_lo: float
    f_hi: float
    before_db: np.ndarray  # original response
    predicted_db: np.ndarray  # original + expected EQ
    after_db: np.ndarray  # original + measured change (level-aligned)
    expected_eq: np.ndarray
    measured_eq: np.ndarray  # level-aligned measured change
    shape_err_db: float
    level_change_db: float
    expected_level_db: float
    status: str = "PASS"


@dataclass
class VerifyResult:
    mode: str
    checks: dict[str, ChannelCheck]
    findings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    status: str = "PASS"


def load_eq(folder: Path, mode: str) -> dict[str, dict]:
    js = json.loads((folder / "summary.json").read_text())
    chans = js["channels"]

    def filters(c):
        return [Filter(f["type"], f["fc"], f["gain"], f["q"]) for f in c["filters"]]

    out = {}
    for ch in CHANNELS:
        c = chans["Stereo"] if mode == "combined" else chans[ch]
        out[ch] = {"filters": filters(c), "preamp_db": c["preamp_db"],
                   "delay_ms": 0.0 if mode == "combined" else c["delay_ms"],
                   "f_lo": chans[ch]["f_lo"], "f_hi": chans[ch]["f_hi"]}
    return out


def _shape_err(measured: np.ndarray, expected: np.ndarray, m: np.ndarray) -> float:
    d = (measured - expected)[m]
    return float(np.sqrt(np.mean((d - np.mean(d)) ** 2)))


def _scaled(filters: list[Filter], k: float) -> list[Filter]:
    return [Filter(f.kind, f.fc * k, f.gain, f.q) for f in filters]


def _inverted(filters: list[Filter]) -> list[Filter]:
    return [Filter(f.kind, f.fc, -f.gain, f.q) for f in filters]


def verify_width(grid: np.ndarray) -> np.ndarray:
    """Slightly broader than design smoothing: robust to small mic re-placement."""
    return variable_width(grid, lf=1 / 12, hf=1 / 3, f_a=100, f_b=1000)


class _Through:
    """Predict a smoothed response for 'original speaker+room with these filters'.

    The EQ is applied to the *raw* averaged spectrum and then smoothed exactly like
    the new measurement, so a narrow Q10 cut and its prediction blur identically."""

    def __init__(self, resp, grid, fs, cal_db):
        self.freqs, self.power, self.grid, self.fs = resp.freqs, resp.power, grid, fs
        self.width, self.cal_db = verify_width(grid), cal_db
        self.before = self.smooth(self.power)

    def smooth(self, power):
        p = smooth_power(self.freqs, power, self.grid, self.width)
        return 10 * np.log10(np.maximum(p, 1e-30)) - self.cal_db

    def effect(self, filters: list[Filter]) -> np.ndarray:
        if not filters:
            return np.zeros_like(self.grid)
        g = 10 ** (eq_db(filters, self.freqs, self.fs) / 10)
        return self.smooth(self.power * g) - self.before


def subset_positions(m: Measurement, n: int) -> Measurement:
    """Keep the first n positions, so a quicker verify compares like with like."""
    keep = [i for i, e in enumerate(m.entries) if e["position"] < n]
    return Measurement(m.sweep, [m.entries[i] for i in keep], [m.segs[i] for i in keep],
                       m.timing, m.meta)


def compare(orig: Measurement, new: Measurement, eq: dict[str, dict], cal: MicCalibration,
            mode: str, window_ms: float = 500.0, main_weight: float = 2.0) -> VerifyResult:
    fs = orig.sweep.fs
    grid = log_grid(10, 22000)
    cal_db = cal.correction_db(grid)
    orig = subset_positions(orig, new.n_positions)
    ro = channel_responses(orig, cal, grid, window_ms, main_weight)
    rn = channel_responses(new, cal, grid, window_ms, main_weight)
    mid = (grid >= 200) & (grid <= 2000)

    checks: dict[str, ChannelCheck] = {}
    ctx = {}
    for ch in CHANNELS:
        e = eq[ch]
        th = _Through(ro[ch], grid, fs, cal_db)
        after = th.smooth(rn[ch].power) if np.array_equal(rn[ch].freqs, th.freqs) else (
            10 * np.log10(np.maximum(smooth_power(rn[ch].freqs, rn[ch].power, grid,
                                                  th.width), 1e-30)) - cal_db)
        exp = th.effect(e["filters"])
        eff = after - th.before
        m = (grid >= e["f_lo"]) & (grid <= min(e["f_hi"], 10000))
        level = float(np.median((eff - exp)[mid]))
        meas = eff - level
        err = _shape_err(eff, exp, m)
        checks[ch] = ChannelCheck(ch, e["f_lo"], e["f_hi"], th.before, th.before + exp,
                                  th.before + meas, exp, meas, err, level, e["preamp_db"])
        ctx[ch] = (th, eff, m)

    res = VerifyResult(mode, checks)
    _diagnose(res, eq, ctx, grid, orig, new, rn)
    for c in checks.values():
        c.status = "PASS" if c.shape_err_db < PASS_DB else "WARN" if c.shape_err_db < WARN_DB else "FAIL"
    worst = [c.status for c in checks.values()]
    res.status = "FAIL" if ("FAIL" in worst or res.findings) else "WARN" if "WARN" in worst else "PASS"
    return res


def _diagnose(res: VerifyResult, eq, ctx, grid, orig, new, rn) -> None:
    checks = res.checks
    swapped = False
    if res.mode == "channel" and all(k in checks for k in ("L", "R")):
        own = checks["L"].shape_err_db + checks["R"].shape_err_db
        thL, effL, mL = ctx["L"]
        thR, effR, mR = ctx["R"]
        cross = (_shape_err(effL, thL.effect(eq["R"]["filters"]), mL)
                 + _shape_err(effR, thR.effect(eq["L"]["filters"]), mR))
        if own > PASS_DB and cross < 0.6 * own:
            swapped = True
            res.findings.append("Left and right EQs look SWAPPED - the Left file is on the "
                                "right speaker. Swap them.")

    for ch, c in checks.items():
        if c.shape_err_db < PASS_DB or swapped:
            continue
        th, eff, m = ctx[ch]
        filters = eq[ch]["filters"]
        alternatives = {
            "the EQ is not active - nothing changed. Check Equalizer APO is enabled for this "
            "device, the SoundSource effect isn't bypassed, or the right miniDSP preset is "
            "selected": [],
            "the EQ is inverted (cuts became boosts)": _inverted(filters),
            "every filter sits at HALF its frequency - the coefficients were made for double "
            "the device's sample rate (e.g. 96 kHz biquads on a 48 kHz miniDSP)":
                _scaled(filters, 0.5),
            "every filter sits at DOUBLE its frequency - the coefficients were made for half "
            "the device's sample rate": _scaled(filters, 2.0),
        }
        errs = {k: _shape_err(eff, th.effect(v), m) for k, v in alternatives.items()}
        best = min(errs, key=errs.get)
        if errs[best] < 0.6 * c.shape_err_db:
            res.findings.append(f"{_name(ch)}: {best}.")
        elif np.max(np.abs((eff - c.level_change_db - c.expected_eq)[m])) > 15:
            res.findings.append(
                f"{_name(ch)}: extreme deviations (>15 dB) - a filter may be unstable. For "
                "miniDSP biquads, check the import used the file unmodified.")

    if res.mode == "channel" and all(k in checks for k in ("L", "R")):
        bal = band_level(grid, rn["L"].fine_db) - band_level(grid, rn["R"].fine_db)
        res.info.append(f"L/R level balance after correction: {bal:+.1f} dB (target 0.0)")
        if abs(bal) > 1.0:
            res.findings.append(f"Left is {bal:+.1f} dB relative to right - the level trims / "
                                "output gains aren't applied as in summary.txt.")

        t_old, t_new = orig.timing or {}, new.timing or {}
        if "r_minus_l_ms" in t_new:
            rl, rl0 = t_new["r_minus_l_ms"], t_old.get("r_minus_l_ms", 0.0)
            res.info.append(f"L/R arrival difference after correction: {rl:+.2f} ms "
                            f"(was {rl0:+.2f} ms, target 0.00)")
            if abs(rl) > 0.1:
                if abs(rl - rl0) < 0.05 and abs(rl0) > 0.1:
                    res.findings.append("The speaker delay isn't applied (arrival difference "
                                        "unchanged) - set the delay from summary.txt.")
                elif not swapped:
                    res.findings.append(f"Speakers are {abs(rl):.2f} ms out of alignment - "
                                        "check the delay setting, or the mic wasn't at the "
                                        "same main position.")

    for ch, c in checks.items():
        res.info.append(f"{_name(ch)}: level changed {c.level_change_db:+.1f} dB, expected "
                        f"{c.expected_level_db:+.1f} dB if the volume wasn't touched")


def _name(ch: str) -> str:
    return {"L": "Left", "R": "Right"}.get(ch, ch)


def report_text(res: VerifyResult, folder: Path, n_positions: int) -> str:
    what = ("per-channel EQ (Equalizer APO / miniDSP)" if res.mode == "channel"
            else "Combined_ParametricEQ.txt (SoundSource)")
    lines = [f"roomeq verify - {folder.name}", f"checked: {what}, {n_positions} position(s)", "",
             f"RESULT: {res.status}", ""]
    for ch, c in res.checks.items():
        lines.append(f"  {_name(ch):<6s} {c.status:<4s}  measured EQ vs expected: "
                     f"{c.shape_err_db:.1f} dB RMS ({c.f_lo:.0f}-{min(c.f_hi, 10000):.0f} Hz)")
    lines.append("")
    if res.findings:
        lines.append("Problems found:")
        lines += [f"  ! {f}" for f in res.findings]
        lines.append("")
    lines += [f"  {i}" for i in res.info]
    lines += ["", f"Pass < {PASS_DB} dB; up to {WARN_DB} dB is usually mic placement "
              "(re-use the same positions). Chart: verify_report.png"]
    return "\n".join(lines) + "\n"


def plot_verify(path: Path, title: str, res: VerifyResult) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = log_grid(10, 22000)
    m = (grid >= 15) & (grid <= 20000)
    n = len(res.checks)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.4 * n), squeeze=False)
    for ax, c in zip(axes[:, 0], res.checks.values()):
        ax.semilogx(grid[m], c.before_db[m], color="0.6", lw=1.0, label="before (original)")
        ax.semilogx(grid[m], c.predicted_db[m], color="#1f6fb2", lw=1.4, ls="--",
                    label="predicted with EQ")
        ax.semilogx(grid[m], c.after_db[m], color="#c0392b", lw=1.5,
                    label="measured with EQ (level-aligned)")
        off = float(np.median(c.predicted_db[(grid > 200) & (grid < 2000)])) - 22
        ax.semilogx(grid[m], c.expected_eq[m] + off, color="#2e9d55", lw=1.0, ls="--",
                    label="EQ expected")
        ax.semilogx(grid[m], c.measured_eq[m] + off, color="#2e9d55", lw=1.4,
                    label="EQ measured")
        ax.axvspan(15, c.f_lo, color="0.92", lw=0)
        ax.axvspan(min(c.f_hi, 10000), 20000, color="0.92", lw=0)
        ref = off + 22
        ax.set_ylim(ref - 34, ref + 16)
        ax.set_xlim(15, 20000)
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("dB")
        ax.set_title(f"{_name(c.name)}: measured EQ vs expected {c.shape_err_db:.1f} dB RMS "
                     f"(match: {c.status})", loc="left", fontsize=10)
        ax.legend(loc="lower left", fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("Hz")
    head = f"{title}  -  RESULT: {res.status}"
    if res.findings:
        import textwrap
        head += "\n" + "\n".join(textwrap.shorten(f, 140) for f in res.findings[:3])
    fig.suptitle(head, fontsize=11, color="#c0392b" if res.status == "FAIL" else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def result_json(res: VerifyResult) -> dict:
    return {"mode": res.mode, "status": res.status, "findings": res.findings, "info": res.info,
            "channels": {ch: {"status": c.status, "shape_err_db": c.shape_err_db,
                              "level_change_db": c.level_change_db,
                              "expected_level_db": c.expected_level_db}
                         for ch, c in res.checks.items()}}
