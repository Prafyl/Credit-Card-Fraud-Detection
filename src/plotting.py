"""Shared chart styling, so every figure in reports/figures/ looks like one system.

Palette validated with the dataviz colour checks: blue vs red separate at CVD deltaE 21.6
and normal-vision deltaE 32.3, both well clear of the 8 / 15 floors, and both clear 3:1
contrast against the chart surface.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "reports" / "figures"

# Series colours: identity, not status. Legit is the neutral baseline, fraud the accent.
LEGIT = "#2a78d6"
FRAUD = "#e34948"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Correlations are signed, so the heatmap needs a diverging scale: two opposite hues
# with a neutral grey midpoint, so "no correlation" reads as nothing.
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", [LEGIT, "#f0efec", FRAUD])


def use_house_style() -> None:
    """Apply the shared rcParams once, at import time."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",  # solid hairlines; dashed grids read as thresholds
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def save(fig, name: str) -> None:
    """Write a figure to reports/figures/<name>.png and close it."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


use_house_style()
