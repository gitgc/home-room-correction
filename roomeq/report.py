"""Before/after plot."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_report(path: Path, title: str, panels: list[dict]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 4.2 * len(panels)), squeeze=False)
    for ax, p in zip(axes[:, 0], panels):
        g = p["grid"]
        m = (g >= 15) & (g <= 20000)
        for k, pos in enumerate(p.get("positions", [])):
            ax.semilogx(g[m], pos[m], color="0.78", lw=0.6,
                        label="individual mic positions" if k == 0 else None)
        ax.semilogx(g[m], p["measured"][m], color="#c0392b", lw=1.4, label="BEFORE: measured (all positions averaged)")
        ax.semilogx(g[m], p["target"][m], color="k", lw=1.2, ls="--", label="target")
        ax.semilogx(g[m], p["corrected"][m], color="#1f6fb2", lw=1.6, label="AFTER: predicted with EQ")
        ax.semilogx(g[m], p["eq"][m] + p["eq_offset"], color="#2e9d55", lw=1.0,
                    label=f"EQ curve (offset {p['eq_offset']:+.0f} dB)")
        ax.axvspan(15, p["f_lo"], color="0.92", lw=0)
        ax.axvspan(p["f_hi"], 20000, color="0.92", lw=0)
        ref = float(np.median(p["target"][(g > 200) & (g < 2000)]))
        ax.set_ylim(ref - 30, ref + 18)
        ax.set_xlim(15, 20000)
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000])
        ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
        ax.grid(True, which="both", alpha=0.3)
        ax.set_ylabel("dB")
        ax.set_title(p["title"], loc="left", fontsize=10)
        ax.legend(loc="lower left", fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel("Hz")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
