# machine_learning/utils/io/analysis/visualization/theme.py
#
# Centralized design tokens and matplotlib rcParams presets.
# Import ``Theme`` and call ``Theme.apply()`` once before any plotting.

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════════

# Survival-related feature highlight colors
SURV_COLOR:   str = "#E8605D"
SURV_LIGHT:   str = "#F4A9A7"
NOSURV_COLOR: str = "#3A7CA5"
NOSURV_LIGHT: str = "#8DC0DE"

# Canvas / text
BG_COLOR:     str = "#FAFAF8"
GRID_COLOR:   str = "#E6E6E2"
TEXT_DARK:    str = "#2B2B2B"
TEXT_MID:     str = "#6B6B6B"
STRIPE_COLOR: str = "#F3F3F0"
DEEP_BLUE:    str = "#1B4F72"

# Confusion-matrix colormap
CM_CMAP = LinearSegmentedColormap.from_list(
    "styled_blues",
    [BG_COLOR, NOSURV_LIGHT, NOSURV_COLOR, DEEP_BLUE],
    N=256,
)

# ROC curve line styles
LINE_COLORS: list[str] = [NOSURV_COLOR, NOSURV_LIGHT, DEEP_BLUE, SURV_COLOR, SURV_LIGHT]
LINE_STYLES: list      = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
CLASS_COLORS: list[str] = [NOSURV_COLOR, SURV_COLOR, NOSURV_LIGHT, SURV_LIGHT]


# ══════════════════════════════════════════════════════════════════════════════
#  RCPARAMS PRESETS
# ══════════════════════════════════════════════════════════════════════════════

_NOTEBOOK_PARAMS: dict = {
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         120,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}

_PUBLICATION_PARAMS: dict = {
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "DejaVu Sans", "Arial"],
    "axes.facecolor":     BG_COLOR,
    "figure.facecolor":   BG_COLOR,
    "text.color":         TEXT_DARK,
    "axes.edgecolor":     GRID_COLOR,
    "axes.grid":          False,
    "xtick.color":        TEXT_MID,
    "ytick.color":        TEXT_MID,
    "figure.dpi":         120,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}


class Theme:
    """
    Apply consistent matplotlib styling across all analysis plots.

    Usage
    -----
    ::

        from analysis.visualization.theme import Theme

        Theme.apply()                # default notebook style
        Theme.apply("publication")   # publication-ready (sans-serif, styled bg)
    """

    @staticmethod
    def apply(preset: str = "notebook") -> None:
        """
        Apply an rcParams preset globally.

        Parameters
        ----------
        preset : {'notebook', 'publication'}
        """
        params = (
            _PUBLICATION_PARAMS if preset == "publication"
            else _NOTEBOOK_PARAMS
        )
        plt.rcParams.update(params)

    @staticmethod
    def reset() -> None:
        """Restore matplotlib defaults."""
        plt.rcdefaults()
