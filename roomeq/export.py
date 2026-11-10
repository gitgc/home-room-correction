"""Write correction files for Equalizer APO, SoundSource (AutoEq format) and miniDSP."""

from __future__ import annotations

from pathlib import Path

from .peq import Filter, biquad

MINIDSP_FS = 96000  # miniDSP 2x4 HD internal sample rate


def filter_line(i: int, f: Filter) -> str:
    fc = f"{f.fc:.1f}" if f.fc < 1000 else f"{f.fc:.0f}"
    return f"Filter {i}: ON {f.kind} Fc {fc} Hz Gain {f.gain:.1f} dB Q {f.q:.2f}"


def autoeq_text(filters: list[Filter], preamp_db: float) -> str:
    """AutoEq 'ParametricEQ.txt' format - imported by SoundSource, Wavelet, APO etc."""
    lines = [f"Preamp: {preamp_db:.1f} dB"]
    lines += [filter_line(i, f) for i, f in enumerate(filters, 1)]
    return "\n".join(lines) + "\n"


def apo_text(channels: dict[str, dict], header: list[str]) -> str:
    """channels: {"L": {"filters", "preamp_db", "delay_ms"}, ...}"""
    out = [f"# {h}" for h in header] + [""]
    for ch, c in channels.items():
        out.append(f"Channel: {ch}")
        out.append(f"Preamp: {c['preamp_db']:.1f} dB")
        if c.get("delay_ms", 0) > 0:
            out.append(f"Delay: {c['delay_ms']:.2f} ms")
        out += [filter_line(i, f) for i, f in enumerate(c["filters"], 1)]
        out.append("")
    out.append("Channel: all")
    return "\n".join(out) + "\n"


def minidsp_biquads_text(filters: list[Filter], fs: int = MINIDSP_FS, n_slots: int = 10) -> str:
    """miniDSP 'advanced' biquad import format. Note miniDSP's sign convention:
    a1 and a2 are negated relative to the usual difference equation."""
    lines = []
    for i in range(n_slots):
        if i < len(filters):
            b, a = biquad(filters[i], fs)
        else:
            b, a = [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]
        lines += [f"biquad{i + 1},", f"b0={b[0]:.15f},", f"b1={b[1]:.15f},", f"b2={b[2]:.15f},",
                  f"a1={-a[1]:.15f},", f"a2={-a[2]:.15f}" + ("," if i < n_slots - 1 else "")]
    return "\n".join(lines) + "\n"


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
