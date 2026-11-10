"""Target (house) curves for in-room steady-state response.

A flat in-room response sounds thin and bright: in a real room a well-behaved
speaker measures with a gently falling treble and some bass lift. The default
here follows the shape of Harman's preferred in-room curve and Dirac Live's
default target: a bass shelf below ~100 Hz and roughly -0.8 dB/octave above
200 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .peq import Filter, filter_db

PRESETS = {
    # name: (bass shelf dB, tilt dB/oct)
    "default": (4.0, -0.8),
    "warm": (6.0, -1.0),
    "neutral": (2.0, -0.5),
    "flat": (0.0, 0.0),
}


@dataclass
class Target:
    name: str
    bass_db: float = 4.0
    bass_fc: float = 105.0
    tilt_db_oct: float = -0.8
    tilt_from: float = 200.0
    file: Path | None = None

    def shape_db(self, grid: np.ndarray) -> np.ndarray:
        """Target relative to 1 kHz = 0 dB."""
        if self.file is not None:
            f, d = load_target_file(self.file)
            out = np.interp(np.log10(grid), np.log10(f), d)
            return out - np.interp(3.0, np.log10(f), d)
        shelf = filter_db(Filter("LSC", self.bass_fc, self.bass_db, 0.7), grid, 48000)
        tilt = self.tilt_db_oct * np.log2(np.maximum(grid, self.tilt_from) / 1000.0)
        return shelf + tilt

    def describe(self) -> str:
        if self.file is not None:
            return f"file {self.file}"
        return (f"{self.name}: bass shelf {self.bass_db:+.1f} dB @ {self.bass_fc:.0f} Hz, "
                f"tilt {self.tilt_db_oct:+.2f} dB/oct above {self.tilt_from:.0f} Hz")


def make_target(spec: str, bass_db: float | None = None, tilt: float | None = None) -> Target:
    if spec in PRESETS:
        b, t = PRESETS[spec]
        return Target(spec, bass_db=b if bass_db is None else bass_db,
                      tilt_db_oct=t if tilt is None else tilt)
    path = Path(spec)
    if not path.exists():
        raise ValueError(f"target '{spec}' is neither a preset ({', '.join(PRESETS)}) nor a file")
    return Target(path.stem, file=path)


def load_target_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """REW-style house curve: lines of 'frequency dB', comments with * or #."""
    f, d = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line[0] in "*#\"":
            continue
        parts = line.replace(",", " ").split()
        try:
            f.append(float(parts[0]))
            d.append(float(parts[1]))
        except (ValueError, IndexError):
            continue
    order = np.argsort(f)
    return np.array(f)[order], np.array(d)[order]
