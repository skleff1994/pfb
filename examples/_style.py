"""Shared matplotlib style for the example figures.

PNG previews go to examples/figures/. Vector PDFs (the figures of the theory note and report) go to the
directory $PFB_DOC_FIG (default: examples/figures/pdf).
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
DOC_FIG = Path(os.environ.get("PFB_DOC_FIG") or FIG_DIR / "pdf")
DOC_FIG.mkdir(parents=True, exist_ok=True)

from pfb.plot import COLOR, MUTED  # noqa: E402,F401  one colour per hypothesis

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": "#0b0b0b",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "lines.linewidth": 2.0, "legend.frameon": False, "font.size": 10,
})


def save(fig, name, pdf_stem=None):
    """Save a PNG preview to examples/figures/name and, if pdf_stem is given, a vector PDF to DOC_FIG ($PFB_DOC_FIG)."""
    path = FIG_DIR / name
    fig.savefig(path)
    print(f"saved {path}")
    if pdf_stem is not None:
        pdf = DOC_FIG / f"{pdf_stem}.pdf"
        fig.savefig(pdf)
        print(f"saved {pdf}")
