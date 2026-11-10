"""Microphone calibration files (miniDSP UMIK-1 / REW format)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MicCalibration:
    path: Path
    freqs: np.ndarray
    db: np.ndarray
    sens_factor_db: float | None = None
    serial: str | None = None

    def correction_db(self, f: np.ndarray) -> np.ndarray:
        """Mic deviation (dB) at frequencies f. Subtract this from a measurement."""
        lf = np.log10(np.clip(np.asarray(f, dtype=float), 1e-3, None))
        return np.interp(lf, np.log10(self.freqs), self.db, left=self.db[0], right=self.db[-1])


_NUM = re.compile(r"^\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)[\s,;]+([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)")


def load_calibration(path: str | Path) -> MicCalibration:
    path = Path(path)
    freqs, vals = [], []
    sens = serial = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.search(r"Sens Factor\s*=\s*([-+]?\d+(?:\.\d+)?)", line, re.I)
        if m:
            sens = float(m.group(1))
        m = re.search(r"SERNO:\s*(\w+)", line, re.I)
        if m:
            serial = m.group(1)
        m = _NUM.match(line)
        if m and not line.lstrip().startswith(('"', "*", "#")):
            freqs.append(float(m.group(1)))
            vals.append(float(m.group(2)))
    if len(freqs) < 10:
        raise ValueError(f"{path}: does not look like a calibration file")
    order = np.argsort(freqs)
    return MicCalibration(path, np.array(freqs)[order], np.array(vals)[order], sens, serial)


def find_calibration(mic_dir: str | Path, orientation: int) -> Path:
    """Pick the 0° or 90° file from the mic folder by filename."""
    files = sorted(Path(mic_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no calibration files in {mic_dir}")
    is90 = [f for f in files if "90" in f.stem.lower()]
    is0 = [f for f in files if f not in is90]
    pick = is90 if orientation == 90 else is0
    if not pick:
        raise FileNotFoundError(f"no {orientation}° calibration file in {mic_dir}")
    return pick[0]
