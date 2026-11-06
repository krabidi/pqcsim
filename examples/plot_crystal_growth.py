"""Crystal-growth velocity of three phase-change materials vs temperature, with published measurements.

Curves: pqcsim.devices.growth with the parameters stated in each paper (no fitting here).
Markers: values reported in the papers (see docs/validation.md, sections 1 and 3).
Writes docs/figures/crystal_growth.png.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT)]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pqcsim.devices.growth import orava2012_gst, salinga2013_aist, sebastian2014_gst

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

T = np.linspace(400, 900, 2001)
series = [
    ("AgInSbTe, melt-quenched (Salinga 2013)", salinga2013_aist(), BLUE),
    ("GST, supercooled liquid (Orava 2012)", orava2012_gst(), ORANGE),
    ("GST, melt-quenched (Sebastian 2014)", sebastian2014_gst(), AQUA),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for label, model, color in series:
    u = model.velocity(T)
    ok = u > 0
    ax.plot(T[ok], u[ok], color=color, lw=2, label=label, zorder=3)

marker = dict(ms=8, mec=SURFACE, mew=2, zorder=5, ls="none")
# Salinga 2013: measured ~100 nm/s at 418 K and 3.4 m/s at 553 K; maximum 10-100 m/s (two-pulse experiment)
ax.plot([418, 553], [1e-7, 3.4], "o", color=BLUE, **marker)
ax.errorbar([716], [np.sqrt(10 * 100)], yerr=[[np.sqrt(10 * 100) - 10], [100 - np.sqrt(10 * 100)]],
            fmt="none", ecolor=BLUE, elinewidth=2, capsize=4, zorder=4)
# Orava 2012: maximum just above 3 m/s at 0.76 T_m
ax.plot([0.76 * 900], [3.0], "s", color=ORANGE, **marker)
# Sebastian 2014: measured temperature of maximum growth 749-752 K
ax.fill_betweenx([0.05, 3], 749, 752, color=AQUA, alpha=0.6, lw=0, zorder=4)

ax.set_yscale("log")
ax.set_xlim(400, 900)
ax.set_ylim(1e-14, 1e5)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Crystal growth velocity (m/s)")
ax.grid(True, which="major", color=GRID, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax.annotate("measured\n~100 nm/s", (418, 1e-7), xytext=(430, 1e-5), color=INK2, fontsize=8.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("measured maximum\n10–100 m/s", (716, 100), xytext=(560, 1e4), color=INK2, fontsize=8.5,
            ha="right", va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("reported maximum\n≈3 m/s at 0.76 $T_m$", (684, 3.0), xytext=(790, 1e4), color=INK2,
            fontsize=8.5, va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.annotate("measured peak\ntemperature 749–752 K", (752, 0.05), xytext=(765, 2e-3), color=INK2, fontsize=8.5,
            va="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
ax.set_title("Model curves (lines) vs published measurements (markers)", color=INK, fontsize=11, loc="left")
fig.tight_layout()
out = ROOT / "docs" / "figures" / "crystal_growth.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor=SURFACE)
print(out)
