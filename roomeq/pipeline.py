"""Measurement storage and the analysis -> EQ -> files pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import export
from .analysis import band_level, channel_response, combine_response, log_grid
from .calibration import MicCalibration, find_calibration, load_calibration
from .optimizer import EQConfig, EQResult, design_eq, max_boost, rms_error
from .peq import Filter
from .report import plot_report
from .sweep import SweepParams
from .target import Target

CHANNELS = ("L", "R")


@dataclass
class Measurement:
    sweep: SweepParams
    entries: list[dict] = field(default_factory=list)  # position, channel, peak, pnr_db, polarity
    segs: list[np.ndarray] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def add(self, position: int, ir) -> None:
        self.entries.append({"position": position, "channel": ir.channel, "peak": ir.peak,
                             "pnr_db": ir.pnr_db, "polarity": ir.polarity})
        self.segs.append(ir.seg)

    @property
    def n_positions(self) -> int:
        return len({e["position"] for e in self.entries})

    def save(self, path: Path) -> None:
        meta = {"sweep": asdict(self.sweep), "entries": self.entries, "timing": self.timing,
                "meta": self.meta}
        np.savez_compressed(path, meta=json.dumps(meta),
                            **{f"ir{i}": s for i, s in enumerate(self.segs)})

    @classmethod
    def load(cls, path: Path) -> "Measurement":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        segs = [z[f"ir{i}"] for i in range(len(meta["entries"]))]
        return cls(SweepParams(**meta["sweep"]), meta["entries"], segs, meta["timing"], meta["meta"])


@dataclass
class Options:
    target: Target
    eq: EQConfig
    window_ms: float = 500.0
    main_weight: float = 2.0


@dataclass
class ChannelResult:
    name: str
    eq: EQResult
    fine_db: np.ndarray
    target_db: np.ndarray
    positions_db: list
    trim_db: float = 0.0
    delay_ms: float = 0.0
    preamp_db: float = 0.0
    err_before: dict = field(default_factory=dict)
    err_after: dict = field(default_factory=dict)


def resolve_calibration(m: Measurement, mic_dir: Path, orientation: int | None) -> MicCalibration:
    orient = orientation if orientation is not None else m.meta.get("mic_orientation", 90)
    stored = m.meta.get("cal_file")
    if orientation is None and stored and (mic_dir / stored).exists():
        return load_calibration(mic_dir / stored)
    return load_calibration(find_calibration(mic_dir, orient))


def _errors(grid, resp, target, f_lo, f_hi, schroeder) -> dict:
    top = min(f_hi, 10000)
    return {"full": rms_error(grid, resp, target, f_lo, top),
            "bass": rms_error(grid, resp, target, f_lo, schroeder),
            "upper": rms_error(grid, resp, target, schroeder, top)}


def _design(name, grid, resp, opts: Options) -> ChannelResult:
    shape = opts.target.shape_db(grid)
    m = (grid >= 200) & (grid <= 2000)
    target = shape + float(np.median((resp.fine_db - shape)[m]))
    eq = design_eq(grid, resp.fine_db, resp.coarse_db, target, opts.eq)
    r = ChannelResult(name, eq, resp.fine_db, target, resp.positions_db)
    s = opts.eq.schroeder
    r.err_before = _errors(grid, resp.fine_db, target, eq.f_lo, eq.f_hi, s)
    r.err_after = _errors(grid, resp.fine_db + eq.eq, target, eq.f_lo, eq.f_hi, s)
    return r


def channel_responses(m: Measurement, cal: MicCalibration, grid: np.ndarray, window_ms: float,
                      main_weight: float) -> dict:
    """Spatially averaged response per channel (main position weighted)."""
    responses = {}
    for ch in CHANNELS:
        items = []
        for e, seg in zip(m.entries, m.segs):
            if e["channel"] != ch:
                continue
            same = sum(1 for x in m.entries if x["channel"] == ch and x["position"] == e["position"])
            w = (main_weight if e["position"] == 0 else 1.0) / same
            items.append((seg, e["peak"], w, e["position"]))
        if items:
            responses[ch] = channel_response(items, m.sweep.fs, cal, grid, window_ms)
    if not responses:
        raise RuntimeError("measurement contains no impulse responses")
    return responses


def compute(m: Measurement, opts: Options, out_dir: Path, mic_dir: Path,
            orientation: int | None = None, label: str = "") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = m.sweep.fs
    opts.eq.fs = fs
    cal = resolve_calibration(m, mic_dir, orientation)
    grid = log_grid(10, 22000)

    responses = channel_responses(m, cal, grid, opts.window_ms, opts.main_weight)

    # --- level matching between channels ----------------------------------
    refs = {ch: band_level(grid, r.fine_db) for ch, r in responses.items()}
    lowest = min(refs.values())
    trims = {ch: lowest - v for ch, v in refs.items()}

    results = {ch: _design(ch, grid, r, opts) for ch, r in responses.items()}
    for ch, r in results.items():
        r.trim_db = trims[ch]

    # --- delay alignment (from the main position) --------------------------
    t = m.timing or {}
    rl = t.get("r_minus_l_ms")
    if rl is not None and "L" in results and "R" in results:
        if rl > 0:
            results["L"].delay_ms = rl
        else:
            results["R"].delay_ms = -rl

    # --- headroom: one common preamp so the stereo balance is preserved -----
    head = max(max_boost(r.eq.filters, fs) + r.trim_db for r in results.values())
    for r in results.values():
        r.preamp_db = r.trim_db - max(0.0, head)

    # --- stereo (same EQ on both channels) for SoundSource -----------------
    stereo = None
    if len(responses) == 2:
        p = sum(responses[ch].power * 10 ** (trims[ch] / 10) for ch in CHANNELS) / 2
        sresp = combine_response(responses["L"].freqs, p, grid, cal)
        stereo = _design("Stereo", grid, sresp, opts)
        stereo.preamp_db = -max_boost(stereo.eq.filters, fs)

    # --- files ------------------------------------------------------------
    created = m.meta.get("created", datetime.now().isoformat(timespec="minutes"))
    header = [f"roomeq correction - {label or out_dir.name}", f"measured {created}, "
              f"{m.n_positions} position(s), mic {cal.path.name}", f"target: {opts.target.describe()}"]
    files = []
    apo = {ch: {"filters": r.eq.filters, "preamp_db": r.preamp_db, "delay_ms": r.delay_ms}
           for ch, r in results.items()}
    files.append(export.write(out_dir / "EqualizerAPO.txt", export.apo_text(apo, header)))
    for ch, r in results.items():
        nm = {"L": "Left", "R": "Right"}[ch]
        files.append(export.write(out_dir / f"{nm}_ParametricEQ.txt",
                                  export.autoeq_text(r.eq.filters, -max_boost(r.eq.filters, fs))))
        files.append(export.write(out_dir / f"miniDSP_{nm}_biquads.txt",
                                  export.minidsp_biquads_text(r.eq.filters)))
    if stereo:
        files.append(export.write(out_dir / "Combined_ParametricEQ.txt",
                                  export.autoeq_text(stereo.eq.filters, stereo.preamp_db)))

    panels = []
    for r in list(results.values()) + ([stereo] if stereo else []):
        panels.append({"grid": grid, "measured": r.fine_db, "target": r.target_db,
                       "corrected": r.fine_db + r.eq.eq, "eq": r.eq.eq,
                       "eq_offset": float(np.median(r.target_db[(grid > 200) & (grid < 2000)])) - 20,
                       "positions": r.positions_db, "f_lo": r.eq.f_lo, "f_hi": r.eq.f_hi,
                       "title": f"{ {'L': 'Left', 'R': 'Right'}.get(r.name, 'Stereo (SoundSource)')}"
                                f" - deviation from target (RMS) before {r.err_before['full']:.1f} dB"
                                f" -> after {r.err_after['full']:.1f} dB   |   bass "
                                f"{r.err_before['bass']:.1f} -> {r.err_after['bass']:.1f} dB   |   "
                                f"above {opts.eq.schroeder:.0f} Hz {r.err_before['upper']:.1f} -> "
                                f"{r.err_after['upper']:.1f} dB"})
    files.append(plot_report(out_dir / "report.png", header[0], panels))

    summary = _summary(results, stereo, m, cal, opts, header)
    files.append(export.write(out_dir / "summary.txt", summary["text"]))
    (out_dir / "summary.json").write_text(json.dumps(summary["json"], indent=2))
    files.append(out_dir / "summary.json")
    summary["files"] = files
    return summary


def _filters_table(filters: list[Filter]) -> list[str]:
    rows = ["   #  Type      Fc (Hz)   Gain (dB)     Q"]
    for i, f in enumerate(filters, 1):
        rows.append(f"  {i:2d}  {f.kind:<5s} {f.fc:>10.1f}   {f.gain:>+8.1f}   {f.q:>5.2f}")
    return rows


def _summary(results, stereo, m: Measurement, cal, opts: Options, header) -> dict:
    headroom = max(0.0, max(r.trim_db - r.preamp_db for r in results.values()))
    lines = list(header) + [""]
    t = m.timing or {}
    warnings = []
    if t.get("polarity") and len(set(t["polarity"].values())) > 1:
        warnings.append("L and R speakers appear to have OPPOSITE POLARITY - check speaker "
                        "wiring (+/-) on one side.")
    low = [e for e in m.entries if e["pnr_db"] < 40]
    if low:
        warnings.append(f"{len(low)} sweep(s) had low signal-to-noise; results may be noisy.")
    if "r_minus_l_ms" in t:
        far = "right" if t["r_minus_l_ms"] > 0 else "left"
        lines.append(f"Speaker distance: {far} speaker is {abs(t['distance_diff_cm']):.1f} cm "
                     f"further away ({abs(t['r_minus_l_ms']):.2f} ms); clock drift "
                     f"{t.get('drift_ppm', 0):.0f} ppm.")
    lines.append("")
    js = {"created": m.meta.get("created"), "positions": m.n_positions, "mic": cal.path.name,
          "target": opts.target.describe(), "timing": t, "warnings": warnings, "channels": {}}
    for r in list(results.values()) + ([stereo] if stereo else []):
        title = {"L": "LEFT", "R": "RIGHT", "Stereo": "STEREO (same EQ both channels - SoundSource)"}
        lines.append(f"== {title.get(r.name, r.name)} ==")
        if r.name != "Stereo":
            lines.append(f"  output gain: {r.preamp_db:+.1f} dB   (level trim {r.trim_db:+.1f} dB, "
                         f"includes headroom for boosts)")
            lines.append(f"  output delay: {r.delay_ms:.2f} ms")
        else:
            lines.append(f"  preamp: {r.preamp_db:+.1f} dB")
        lines.append(f"  corrected range: {r.eq.f_lo:.0f} Hz - {r.eq.f_hi:.0f} Hz")
        for n in r.eq.notes:
            lines.append(f"  note: {n}")
        lines.append(f"  RMS deviation from target: {r.err_before['full']:.1f} dB -> "
                     f"{r.err_after['full']:.1f} dB (bass {r.err_before['bass']:.1f} -> "
                     f"{r.err_after['bass']:.1f}, above {opts.eq.schroeder:.0f} Hz "
                     f"{r.err_before['upper']:.1f} -> {r.err_after['upper']:.1f})")
        lines += _filters_table(r.eq.filters)
        lines.append("")
        js["channels"][r.name] = {
            "filters": [f.to_dict() for f in r.eq.filters], "preamp_db": r.preamp_db,
            "trim_db": r.trim_db, "delay_ms": r.delay_ms, "f_lo": r.eq.f_lo, "f_hi": r.eq.f_hi,
            "rms_before": r.err_before, "rms_after": r.err_after}
    if warnings:
        lines += ["WARNINGS:"] + [f"  ! {w}" for w in warnings] + [""]
    lines += [
        "Before/after chart: report.png (grey bands = outside the corrected range).",
        "",
        "How to use:",
        "  Equalizer APO : copy EqualizerAPO.txt into C:\\Program Files\\EqualizerAPO\\config\\",
        "                  and add the line  Include: EqualizerAPO.txt  to config.txt",
        "  SoundSource   : import Combined_ParametricEQ.txt (AutoEq format) as a custom",
        "                  parametric/headphone EQ preset on the output device",
        "  miniDSP 2x4 HD: Output 1 (L) / Output 2 (R) -> PEQ -> Advanced -> Import the",
        "                  miniDSP_*_biquads.txt file (96 kHz coefficients). Set the INPUT gain",
        f"                  to {-headroom:.1f} dB (headroom for boosts) and each OUTPUT's gain to its",
        "                  level trim and delay to the values above.",
    ]
    return {"text": "\n".join(lines) + "\n", "json": js, "warnings": warnings}
