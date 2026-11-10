"""Audio I/O: play a stimulus on the speakers while recording the UMIK-1.

Input and output are opened as two separate streams (they are different
devices with independent clocks, and not every host API supports full-duplex
across devices). The offset between them is unknown but constant for the
whole recording, which is all the analysis needs.
"""

from __future__ import annotations

import threading
import time

import numpy as np


class DeviceError(RuntimeError):
    pass


def _sd():
    import sounddevice as sd
    return sd


def list_devices() -> str:
    sd = _sd()
    lines = []
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]
    for i, d in enumerate(sd.query_devices()):
        tags = []
        if "umik" in d["name"].lower():
            tags.append("UMIK")
        if i == default_in:
            tags.append("default in")
        if i == default_out:
            tags.append("default out")
        lines.append(f"  [{i:2d}] {d['name']:<40s} in:{d['max_input_channels']:<3d} "
                     f"out:{d['max_output_channels']:<3d} {int(d['default_samplerate'])} Hz"
                     + (f"  <- {', '.join(tags)}" if tags else ""))
    return "\n".join(lines)


def resolve_device(spec: str | None, kind: str) -> int:
    sd = _sd()
    devs = sd.query_devices()
    key = f"max_{kind}_channels"
    if spec is None:
        if kind == "input":
            for i, d in enumerate(devs):
                if "umik" in d["name"].lower() and d[key] > 0:
                    return i
            raise DeviceError("UMIK-1 not found. Plug it in, or pass --input (see `roomeq devices`).")
        return int(sd.query_devices(kind="output")["index"])
    if str(spec).isdigit():
        return int(spec)
    matches = [i for i, d in enumerate(devs) if spec.lower() in d["name"].lower() and d[key] > 0]
    if not matches:
        raise DeviceError(f"no {kind} device matching '{spec}' (see `roomeq devices`)")
    return matches[0]


def device_name(idx: int) -> str:
    return _sd().query_devices(idx)["name"]


class RealIO:
    def __init__(self, input_dev: int, output_dev: int, fs: int, warmup_s: float = 0.3):
        self.input_dev, self.output_dev, self.fs, self.warmup_s = input_dev, output_dev, fs, warmup_s

    def describe(self) -> str:
        return f"in: {device_name(self.input_dev)} | out: {device_name(self.output_dev)}"

    def _output_rate(self) -> int:
        """48 kHz if the playback device accepts it, otherwise its own rate."""
        sd = _sd()
        try:
            sd.check_output_settings(device=self.output_dev, samplerate=self.fs, channels=1)
            return self.fs
        except Exception:  # noqa: BLE001
            return int(sd.query_devices(self.output_dev)["default_samplerate"])

    def play_record(self, stim: np.ndarray) -> np.ndarray:
        sd = _sd()
        out_fs = self._output_rate()
        if out_fs != self.fs:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(out_fs, self.fs)
            stim = resample_poly(stim, out_fs // g, self.fs // g, axis=0)
        stim = np.ascontiguousarray(stim, dtype=np.float32)
        n, nch = stim.shape
        chunks: list[np.ndarray] = []
        problems: list[str] = []
        pos = 0
        done = threading.Event()
        hint = threading.Timer(5.0, lambda: print(
            "  ...still waiting for the microphone. On macOS, allow microphone access for your "
            "terminal app in System Settings > Privacy & Security > Microphone, then re-run.",
            flush=True))
        hint.daemon = True
        hint.start()

        def in_cb(indata, frames, t, status):
            if status:
                problems.append(str(status))
            if not chunks:
                hint.cancel()
            chunks.append(indata[:, 0].copy())

        def out_cb(outdata, frames, t, status):
            nonlocal pos
            if status:
                problems.append(str(status))
            k = min(frames, n - pos)
            outdata[:k] = stim[pos:pos + k]
            outdata[k:] = 0
            pos += k
            if pos >= n:
                raise sd.CallbackStop

        try:
            with sd.InputStream(device=self.input_dev, channels=1, samplerate=self.fs,
                                dtype="float32", callback=in_cb):
                time.sleep(self.warmup_s)
                with sd.OutputStream(device=self.output_dev, channels=nch, samplerate=out_fs,
                                     dtype="float32", callback=out_cb,
                                     finished_callback=done.set):
                    if not done.wait(timeout=n / out_fs + 10):
                        raise DeviceError("playback did not finish")
                time.sleep(0.5)
        finally:
            hint.cancel()
        if any("overflow" in p or "underflow" in p for p in problems):
            print(f"  warning: audio glitches during measurement: {sorted(set(problems))}")
        rec = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if not np.any(rec):
            raise DeviceError(
                "the microphone returned pure digital silence. On macOS this means microphone "
                "access is blocked: System Settings > Privacy & Security > Microphone -> enable "
                "your terminal app.")
        return rec


class SimulatedIO:
    """A synthetic room (speaker roll-off, room modes, reflections, reverb, noise).

    Lets the whole pipeline be exercised without speakers or a mic.
    """

    def __init__(self, fs: int = 48000, seed: int = 1, eq: dict | None = None):
        self.fs = fs
        self.rng = np.random.default_rng(seed)
        self.position = 0
        # optional EQ "loaded" in the playback chain, per output channel index:
        # {ch: {"filters": [...], "preamp_db": x, "delay_ms": y, "design_fs": fs}}
        self.eq = eq or {}

    def _apply_eq(self, x: np.ndarray, ch: int) -> np.ndarray:
        e = self.eq.get(ch)
        if not e:
            return x
        from scipy.signal import lfilter
        from .peq import biquad
        for f in e["filters"]:
            b, a = biquad(f, e.get("design_fs", self.fs))
            x = lfilter(b, a, x)
        x = x * 10 ** (e.get("preamp_db", 0.0) / 20)
        d = int(round(e.get("delay_ms", 0.0) * self.fs / 1000))
        return np.concatenate([np.zeros(d), x[:len(x) - d]]) if d else x

    def describe(self) -> str:
        return "simulated room"

    def room_ir(self, channel: int) -> np.ndarray:
        """IR for the current mic position. Geometry is fixed per position index (so a
        later session re-using the same positions sees the same room), plus a little
        session-to-session mic placement error."""
        from scipy.signal import lfilter
        from .peq import Filter, biquad
        fs = self.fs
        geo = np.random.default_rng(1000 * self.position + 7 * channel + 17)
        place = self.rng
        n = int(0.8 * fs)
        x = np.zeros(n)
        delay = int(fs * (0.010 + 0.0006 * channel))  # R speaker 20 cm further away
        x[delay] = 1.0
        for t_ms, g in [(2.6, 0.45), (4.1, 0.35), (6.3, 0.3), (9.5, 0.25)]:
            t_ms += geo.normal(0, 0.3) + place.normal(0, 0.02)
            x[delay + int(fs * t_ms / 1000)] += g * geo.choice([-1, 1]) * 0.8
        tail = geo.normal(0, 1, n) * np.exp(-np.arange(n) / (0.35 / 6.9 * fs)) * 0.02
        tail[:delay + int(0.005 * fs)] = 0
        tail = lfilter([0.3], [1, -0.7], tail)
        x += tail

        def jitter(s):
            return geo.normal(0, s) + place.normal(0, 0.15 * s)

        sections = [
            Filter("PK", 41 + jitter(1), 7 + jitter(1.5), 7),
            Filter("PK", 58, -7 + jitter(2) - 2 * channel, 5),
            Filter("PK", 87 + jitter(2), 8 + jitter(1.5), 9),
            Filter("PK", 130, 4 + jitter(1), 5),
            Filter("PK", 220, -3, 3),
            Filter("PK", 2500, 2.5, 1.2),
            Filter("HSC", 9000, -2.5, 0.7),
        ]
        for f in sections:
            b, a = biquad(f, fs)
            x = lfilter(b, a, x)
        # speaker roll-off: 2nd-order high-pass at 45 Hz
        w0 = 2 * np.pi * 45 / fs
        al = np.sin(w0) / (2 * 0.7)
        cw = np.cos(w0)
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]) / (1 + al)
        a = np.array([1, -2 * cw / (1 + al), (1 - al) / (1 + al)])
        return lfilter(b, a, x) * 0.3

    def play_record(self, stim: np.ndarray) -> np.ndarray:
        from scipy.signal import fftconvolve
        latency = int(self.fs * (0.3 + 0.017))
        rec = np.zeros(stim.shape[0] + latency + int(0.8 * self.fs))
        for ch in range(stim.shape[1]):
            if not np.any(stim[:, ch]):
                continue
            y = fftconvolve(self._apply_eq(stim[:, ch].astype(float), ch), self.room_ir(ch))
            rec[latency:latency + len(y)] += y[:len(rec) - latency]
        rec += self.rng.normal(0, 10 ** (-80 / 20), len(rec))
        self.position += 1
        return rec.astype(np.float32)
