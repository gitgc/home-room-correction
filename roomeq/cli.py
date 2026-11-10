"""roomeq command line."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .optimizer import EQConfig
from .pipeline import Measurement, Options, compute
from .sweep import SweepParams, analyse_recording, build_stimulus, exp_sweep, timing_from_sequence
from .target import PRESETS, make_target

ROOT = Path(__file__).resolve().parent.parent
MIC_DIR = ROOT / "mic"
CORRECTIONS_DIR = ROOT / "corrections"

POSITION_HINTS = [
    "MAIN listening position: mic exactly where your head goes, at ear height",
    "~30 cm LEFT of the main position, same height",
    "~30 cm RIGHT of the main position, same height",
    "~30 cm FORWARD of the main position (towards the speakers)",
    "~30 cm BEHIND the main position",
    "~15 cm ABOVE the main position",
    "~30 cm LEFT and ~20 cm FORWARD",
    "~30 cm RIGHT and ~20 cm FORWARD",
    "~15 cm BELOW the main position",
]


def _hint(i: int) -> str:
    if i < len(POSITION_HINTS):
        return POSITION_HINTS[i]
    return "anywhere within ~40 cm of the main position, not repeating an earlier spot"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "room"


# ---------------------------------------------------------------------------
def add_analysis_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("correction")
    g.add_argument("--target", default="default",
                   help=f"target curve preset ({', '.join(PRESETS)}) or a REW-style house "
                        "curve file (default: default)")
    g.add_argument("--bass-db", type=float, help="override target bass shelf (dB)")
    g.add_argument("--tilt", type=float, help="override target tilt (dB/octave above 200 Hz)")
    g.add_argument("--filters", type=int, default=10, help="number of PEQ bands (default 10)")
    g.add_argument("--min-freq", type=float,
                   help="lowest corrected frequency (default: auto-detect speaker extension)")
    g.add_argument("--max-freq", type=float, default=16000.0,
                   help="highest corrected frequency (default 16000)")
    g.add_argument("--max-boost", type=float, default=4.0, help="max boost in dB (default 4)")
    g.add_argument("--max-cut", type=float, default=15.0, help="max cut in dB (default 15)")
    g.add_argument("--schroeder", type=float, default=300.0,
                   help="room transition frequency: precise correction below, broad above")
    g.add_argument("--no-shelves", action="store_true", help="use peaking filters only")
    g.add_argument("--window-ms", type=float, default=500.0,
                   help="impulse response window (default 500 ms)")
    g.add_argument("--main-weight", type=float, default=2.0,
                   help="weight of the main listening position vs. others (default 2)")


def build_options(a) -> Options:
    target = make_target(a.target, a.bass_db, a.tilt)
    eq = EQConfig(n_filters=a.filters, min_freq=a.min_freq, max_freq=a.max_freq,
                  schroeder=a.schroeder, max_boost=a.max_boost, max_cut=a.max_cut,
                  shelves=not a.no_shelves)
    return Options(target=target, eq=eq, window_ms=a.window_ms, main_weight=a.main_weight)


def print_summary(summary: dict, out_dir: Path, open_report: bool = False) -> None:
    print()
    print(summary["text"])
    print(f"Files written to {out_dir}:")
    for f in summary["files"]:
        print(f"  {Path(f).name}")
    if open_report:
        _open_file(out_dir / "report.png")


def _open_file(path: Path) -> None:
    import os
    import subprocess
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except OSError:
        pass


# ---------------------------------------------------------------------------
def cmd_devices(a) -> int:
    from .audio import list_devices
    print(list_devices())
    return 0


def _make_io(a, fs):
    if a.simulate:
        from .audio import SimulatedIO
        return SimulatedIO(fs)
    from .audio import RealIO, resolve_device
    return RealIO(resolve_device(a.input, "input"), resolve_device(a.output, "output"), fs)


def _channel_map(a) -> dict[str, int]:
    l, r = (int(x) - 1 for x in a.channels.split(","))
    return {"L": l, "R": r}


def cmd_levels(a) -> int:
    """Play pink noise on each speaker and report what the mic hears."""
    fs = 48000
    io = _make_io(a, fs)
    cmap = _channel_map(a)
    rng = np.random.default_rng(0)
    n = int(3 * fs)
    spec = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / fs)
    spec[1:] /= np.sqrt(f[1:])
    spec[(f < 30) | (f > 16000)] = 0
    noise = np.fft.irfft(spec, n)
    noise *= 10 ** (a.level / 20) / np.sqrt(np.mean(noise ** 2))
    print(f"{io.describe()}\nPlaying pink noise at {a.level:.0f} dBFS RMS on each speaker...")
    for ch, idx in cmap.items():
        stim = np.zeros((n + fs // 2, max(cmap.values()) + 1), dtype=np.float32)
        stim[:n, idx] = noise
        rec = io.play_record(stim)
        mid = rec[int(0.8 * fs):int(0.8 * fs) + 2 * fs]
        rms = 20 * np.log10(max(np.sqrt(np.mean(mid ** 2)), 1e-12))
        pk = 20 * np.log10(max(np.max(np.abs(rec)), 1e-12))
        verdict = ("TOO LOUD - turn down" if pk > -3 else "too quiet - turn up" if rms < -45
                   else "good")
        print(f"  {ch}: mic RMS {rms:6.1f} dBFS, peak {pk:6.1f} dBFS  -> {verdict}")
    print("Aim for mic RMS between about -40 and -20 dBFS (roughly 70-80 dB SPL at the seat).")
    return 0


def run_positions(m: Measurement, io, cmap: dict[str, int], positions: int, yes: bool) -> bool:
    """Guide the user through the mic positions, filling `m`. False = aborted/empty."""
    p = m.sweep
    sweep = exp_sweep(p)
    pos = 0
    while pos < positions:
        print(f"[{pos + 1}/{positions}] {_hint(pos)}")
        if not yes:
            ans = input("  press Enter to measure, q to finish now: ").strip().lower()
            if ans == "q":
                break
        seq = ["L", "R", "L"] if pos == 0 else ["L", "R"]
        stim, starts = build_stimulus(sweep, p, seq, cmap)
        rec = io.play_record(stim)
        res = analyse_recording(rec, sweep, p, starts, seq)
        snr = ", ".join(f"{ir.channel} {ir.pnr_db:.0f} dB" for ir in res.irs)
        print(f"  mic peak {res.rec_peak_dbfs:.1f} dBFS; impulse SNR {snr}")
        for w in res.warnings:
            print(f"  ! {w}")
        if res.warnings and not yes:
            ans = input("  [r]etry this position, [c]ontinue anyway, [q]uit? ").strip().lower()
            if ans.startswith("q"):
                return False
            if ans.startswith("r") or ans == "":
                continue
        if pos == 0:
            m.timing = timing_from_sequence(res.irs, starts, p.fs)
            t = m.timing
            if t:
                print(f"  R-L arrival difference {t['r_minus_l_ms']:+.2f} ms "
                      f"({t['distance_diff_cm']:+.1f} cm)")
                if len(set(t["polarity"].values())) > 1:
                    print("  ! L and R polarity differ - check speaker wiring")
        for ir in res.irs:
            m.add(pos, ir)
        pos += 1
    if m.n_positions == 0:
        print("nothing measured")
        return False
    return True


def cmd_measure(a) -> int:
    from .calibration import find_calibration
    p = SweepParams(level_db=a.level, duration=a.sweep_length)
    try:
        cal_path = find_calibration(MIC_DIR, a.mic_orientation)
        io = _make_io(a, p.fs)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    cmap = _channel_map(a)
    name = a.name or "room"
    created = datetime.now()
    out_dir = Path(a.out) if a.out else CORRECTIONS_DIR / f"{_slug(name)}_{created:%Y-%m-%d_%H%M}"

    orient = ("pointing straight UP at the ceiling" if a.mic_orientation == 90
              else "pointing at the speakers (between them, at ear height)")
    print(f"roomeq - measuring '{name}'")
    print(f"  {io.describe()}")
    print(f"  mic calibration: {cal_path.name} -> keep the mic {orient}")
    print(f"  {a.positions} position(s), sweep {p.duration:.1f} s at {p.level_db:.0f} dBFS")
    print("  Turn OFF any existing EQ / room correction (Equalizer APO, SoundSource, miniDSP PEQ)")
    print("  for the output you are measuring. Keep the room quiet.\n")

    m = Measurement(p, meta={"name": name, "created": created.isoformat(timespec="minutes"),
                             "mic_orientation": a.mic_orientation, "cal_file": cal_path.name,
                             "devices": io.describe()})
    if not run_positions(m, io, cmap, a.positions, a.yes):
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    m.save(out_dir / "measurement.npz")
    print("\nComputing correction...")
    summary = compute(m, build_options(a), out_dir, MIC_DIR, label=name)
    print_summary(summary, out_dir, open_report=not a.no_open)
    return 0


def cmd_compute(a) -> int:
    src = Path(a.measurement)
    npz = src / "measurement.npz" if src.is_dir() else src
    m = Measurement.load(npz)
    out_dir = Path(a.out) if a.out else npz.parent
    orient = a.mic_orientation
    summary = compute(m, build_options(a), out_dir, MIC_DIR, orientation=orient,
                      label=m.meta.get("name", ""))
    print_summary(summary, out_dir, open_report=a.open)
    return 0


def _sim_eq(eq: dict, how: str, cmap: dict[str, int], fs: int) -> dict:
    """Simulated playback chain for testing verify (correct and broken set-ups)."""
    if how == "none":
        return {}
    out = {}
    for ch, e in eq.items():
        src = eq[{"L": "R", "R": "L"}[ch]] if how == "swapped" else e
        filters = src["filters"]
        if how == "inverted":
            from .peq import Filter
            filters = [Filter(f.kind, f.fc, -f.gain, f.q) for f in filters]
        out[cmap[ch]] = {"filters": filters, "preamp_db": src["preamp_db"],
                         "delay_ms": src["delay_ms"],
                         "design_fs": 2 * fs if how == "halfrate" else fs}
    return out


def cmd_verify(a) -> int:
    from dataclasses import replace

    from . import verify as V
    from .pipeline import resolve_calibration
    folder = Path(a.correction)
    if not (folder / "measurement.npz").exists() or not (folder / "summary.json").exists():
        print(f"error: {folder} is not a correction folder (needs measurement.npz and "
              "summary.json)", file=sys.stderr)
        return 1
    orig = Measurement.load(folder / "measurement.npz")
    eq = V.load_eq(folder, a.eq)
    p = replace(orig.sweep, level_db=a.level if a.level is not None else orig.sweep.level_db)
    positions = a.positions or orig.n_positions
    cmap = _channel_map(a)
    if a.simulate:
        from .audio import SimulatedIO
        io = SimulatedIO(p.fs, seed=2, eq=_sim_eq(eq, a.simulate_eq, cmap, p.fs))
    else:
        io = _make_io(a, p.fs)

    what = {"channel": "the per-channel correction (Equalizer APO / miniDSP: EQ, gains, delays)",
            "combined": "Combined_ParametricEQ.txt (SoundSource)"}[a.eq]
    orient = "pointing UP" if orig.meta.get("mic_orientation", 90) == 90 else "at the speakers"
    print(f"roomeq verify - {folder.name}")
    print(f"  {io.describe()}")
    print(f"  Switch ON {what},")
    print("  and nothing else (no other EQ/effects). Keep the volume where it was if you can.")
    print(f"  Use the same {positions} mic position(s) as the original measurement, mic {orient}.\n")

    new = Measurement(p, meta={"name": orig.meta.get("name", ""), "verify_of": folder.name,
                               "created": datetime.now().isoformat(timespec="minutes"),
                               "mic_orientation": orig.meta.get("mic_orientation", 90),
                               "cal_file": orig.meta.get("cal_file"), "devices": io.describe(),
                               "eq_mode": a.eq})
    if not run_positions(new, io, cmap, positions, a.yes):
        return 1

    cal = resolve_calibration(orig, MIC_DIR, None)
    res = V.compare(orig, new, eq, cal, a.eq)
    stamp = datetime.now()
    base = f"verify_{a.eq}_{stamp:%Y-%m-%d_%H%M}"
    out, k = folder / base, 1
    while out.exists():
        k += 1
        out = folder / f"{base}_{k}"
    out.mkdir(parents=True)
    new.save(out / "measurement.npz")
    text = V.report_text(res, folder, new.n_positions)
    (out / "verify.txt").write_text(text)
    (out / "verify.json").write_text(json.dumps(V.result_json(res), indent=2))
    V.plot_verify(out / "verify_report.png", f"roomeq verify - {folder.name}", res)
    print()
    print(text)
    print(f"Files written to {out}")
    if not a.no_open:
        _open_file(out / "verify_report.png")
    return 2 if res.status == "FAIL" else 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="roomeq", description=(
        "Measure stereo speakers at the listening position with a UMIK-1 and generate a "
        "10-band parametric room-correction EQ for Equalizer APO, SoundSource and miniDSP."))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("devices", help="list audio devices")
    sp.set_defaults(func=cmd_devices)

    def io_args(sp):
        sp.add_argument("--input", help="mic device (name substring or index; default: UMIK-1)")
        sp.add_argument("--output", help="playback device (default: system default output)")
        sp.add_argument("--channels", default="1,2",
                        help="output channels for left,right (1-based, default 1,2)")
        sp.add_argument("--simulate", action="store_true", help=argparse.SUPPRESS)

    sp = sub.add_parser("levels", help="play pink noise to set the volume before measuring")
    io_args(sp)
    sp.add_argument("--level", type=float, default=-20.0, help="noise level dBFS RMS (default -20)")
    sp.set_defaults(func=cmd_levels)

    sp = sub.add_parser("measure", help="measure and write a correction")
    sp.add_argument("--name", help="name for this system/room, e.g. 'living-room'")
    sp.add_argument("--positions", type=int, default=5,
                    help="mic positions around the seat (default 5; Dirac-style 5-9 recommended)")
    sp.add_argument("--mic-orientation", type=int, choices=(0, 90), default=90,
                    help="90 = mic pointing at ceiling (default, recommended), 0 = at speakers")
    sp.add_argument("--level", type=float, default=-12.0, help="sweep level dBFS (default -12)")
    sp.add_argument("--sweep-length", type=float, default=5.5, help="sweep seconds (default 5.5)")
    sp.add_argument("--out", help="output folder (default corrections/<name>_<date>)")
    sp.add_argument("--yes", "-y", action="store_true", help="don't wait between positions")
    sp.add_argument("--no-open", action="store_true", help="don't open the before/after chart")
    io_args(sp)
    add_analysis_args(sp)
    sp.set_defaults(func=cmd_measure)

    sp = sub.add_parser("verify", help="re-measure with the correction switched on and check "
                                       "it was loaded correctly")
    sp.add_argument("correction", help="corrections/<folder> to verify")
    sp.add_argument("--eq", choices=("channel", "combined"), default="channel",
                    help="channel = per-channel files (Equalizer APO, miniDSP; default), "
                         "combined = Combined_ParametricEQ.txt (SoundSource)")
    sp.add_argument("--positions", type=int,
                    help="mic positions (default: same number as the original measurement)")
    sp.add_argument("--level", type=float, help="sweep level dBFS (default: as original)")
    sp.add_argument("--yes", "-y", action="store_true", help="don't wait between positions")
    sp.add_argument("--no-open", action="store_true", help="don't open the verify chart")
    sp.add_argument("--simulate-eq", default="applied",
                    choices=("applied", "none", "swapped", "inverted", "halfrate"),
                    help=argparse.SUPPRESS)
    io_args(sp)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("compute", help="re-run the EQ design on a saved measurement")
    sp.add_argument("measurement", help="corrections/<folder> or a measurement.npz")
    sp.add_argument("--out", help="output folder (default: same folder)")
    sp.add_argument("--mic-orientation", type=int, choices=(0, 90),
                    help="override the calibration file used")
    sp.add_argument("--open", action="store_true", help="open the before/after chart")
    add_analysis_args(sp)
    sp.set_defaults(func=cmd_compute)
    return ap


def main(argv: list[str] | None = None) -> int:
    from .audio import DeviceError
    a = build_parser().parse_args(argv)
    try:
        return a.func(a)
    except DeviceError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\naborted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
