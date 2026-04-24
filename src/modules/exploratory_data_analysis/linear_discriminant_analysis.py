"""
linear_discriminant_analysis.py
================================
Parametrized LDA analysis class.  All column names, bracket lists, colours,
and file-output paths are constructor arguments so the same class works on
both the full dataset and any limited/subset dataset.

Usage — full dataset (original defaults):
    lda = LDAAnalysis(df_credit_sales)
    pipe, X_lda, evr, sep_df = lda.run()

Usage — limited dataset (only financial cols available, two brackets):
    lda = LDAAnalysis(
        df_limited,
        bracket_col    = 'dtp_bracket',
        dtp_features   = [],                          # not present
        fin_features   = ['credit_sale_amount',
                          'amount_paid_cumsum'],
        extra_features = ['payment_ratio'],
        bracket_order  = ['30_days', '60_days'],
        output_path    = 'lda_limited.png',
        title          = 'LDA — Limited dataset',
    )
    pipe, X_lda, evr, sep_df = lda.run()

After fitting, inject LD components back into your dataframe:
    for i in range(lda.X_lda_.shape[1]):
        df[f'LD{i+1}'] = lda.X_lda_[:, i]
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Default config (mirrors the original hard-coded constants)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_BRACKET_COL = "dtp_bracket"

_DEFAULT_DTP_FEATURES = ["dtp_1", "dtp_2", "dtp_3", "dtp_4", "dtp_avg", "dtp_wavg"]

_DEFAULT_FIN_FEATURES = [
    "credit_sale_amount",
    "opening_balance",
    "amount_due_cumsum",
    "amount_paid_cumsum",
]

_DEFAULT_EXTRA_FEATURES = ["payment_ratio"]

_DEFAULT_BRACKET_ORDER = ["30_days", "60_days", "90_days"]  # on_time excluded

_DEFAULT_BRACKET_COLORS: Dict[str, str] = {
    "on_time": "#2ecc71",
    "30_days": "#5dade2",
    "60_days": "#f0b27a",
    "90_days": "#e07b72",
}

_DEFAULT_OUTPUT_PATH = "lda_delinquent_only.png"

_DEFAULT_TITLE = "LDA Analysis — Delinquent Only (30 / 60 / 90 days)  |  on_time excluded"


# ─────────────────────────────────────────────────────────────────────────────
# Class
# ─────────────────────────────────────────────────────────────────────────────


class LDAAnalysis:
    """
    Parametrized Linear Discriminant Analysis with before/after visualisation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.  Only the columns listed in the feature/bracket
        parameters need to be present.

    bracket_col : str
        Name of the target / class column.

    dtp_features : list[str]
        DTP-style features that are NOT log1p-transformed.
        Pass an empty list if your dataset does not have them.

    fin_features : list[str]
        Financial features that WILL be log1p-transformed.
        Pass an empty list if none are available.

    extra_features : list[str]
        Additional features appended after fin_features; NOT log1p-transformed.
        Defaults to ['payment_ratio'].  Pass [] to omit.

    bracket_order : list[str]
        The class labels to keep, in display order.

    bracket_colors : dict[str, str]
        Mapping from bracket label → hex colour.  Any label in bracket_order
        that is missing from this dict gets a generic fallback colour.

    output_path : str
        File path for the saved PNG.  Set to None to skip saving.

    title : str
        Suptitle shown on the figure.

    dpi : int
        Resolution of the saved figure (default 150).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        bracket_col: str = _DEFAULT_BRACKET_COL,
        dtp_features: List[str] = _DEFAULT_DTP_FEATURES,
        fin_features: List[str] = _DEFAULT_FIN_FEATURES,
        extra_features: List[str] = _DEFAULT_EXTRA_FEATURES,
        bracket_order: List[str] = _DEFAULT_BRACKET_ORDER,
        bracket_colors: Dict[str, str] = _DEFAULT_BRACKET_COLORS,
        output_path: Optional[str] = _DEFAULT_OUTPUT_PATH,
        title: str = _DEFAULT_TITLE,
        dpi: int = 150,
    ) -> None:
        self.df = df
        self.bracket_col = bracket_col
        self.dtp_features = list(dtp_features)
        self.fin_features = list(fin_features)
        self.extra_features = list(extra_features)
        self.bracket_order = list(bracket_order)
        self.bracket_colors = dict(bracket_colors)
        self.output_path = output_path
        self.title = title
        self.dpi = dpi

        # Derived lists — computed once on init
        self.all_features: List[str] = (
            self.dtp_features + self.fin_features + self.extra_features
        )

        # Fill in any missing colours with a generic palette
        _fallback = [
            "#9b59b6", "#1abc9c", "#e74c3c", "#3498db", "#f39c12",
        ]
        for i, label in enumerate(self.bracket_order):
            if label not in self.bracket_colors:
                self.bracket_colors[label] = _fallback[i % len(_fallback)]

        # Public result attributes — populated after run()
        self.pipe_: Optional[Pipeline] = None
        self.X_lda_: Optional[np.ndarray] = None
        self.evr_: Optional[np.ndarray] = None
        self.sep_df_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Tuple[Pipeline, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Fit LDA and produce the diagnostic figure.

        Returns
        -------
        pipe     : fitted sklearn Pipeline (StandardScaler → LDA)
        X_lda    : projected data array  (n_samples × n_components)
        evr      : explained variance ratio array
        sep_df   : per-feature Fisher separation before/after table
        """
        X_raw, y = self._load_and_prepare()
        X_t = self._log1p_transform(X_raw)
        pipe, X_lda, evr = self._fit_lda(X_t, y)

        self.pipe_ = pipe
        self.X_lda_ = X_lda
        self.evr_ = evr

        self._print_report(evr)

        sep_before = self._compute_class_separation(X_raw, y)
        sep_after = self._compute_class_separation(X_t, y)
        sep_df = self._build_sep_df(sep_before, sep_after)
        self.sep_df_ = sep_df
        print(sep_df.to_string(index=False))

        self._draw_figure(X_raw, X_t, X_lda, y, evr, sep_before, sep_after)

        return pipe, X_lda, evr, sep_df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Project new data using the already-fitted pipeline.
        Raises RuntimeError if run() has not been called yet.
        """
        if self.pipe_ is None:
            raise RuntimeError("Call run() before transform().")
        X_raw, _ = self._load_and_prepare(df)
        X_t = self._log1p_transform(X_raw)
        return self.pipe_.transform(X_t)

    def inject_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add LD1, LD2, … columns to *df* in-place and return it.
        Uses the index from the fitted data to align rows.
        """
        if self.X_lda_ is None:
            raise RuntimeError("Call run() before inject_components().")
        X_raw, _ = self._load_and_prepare()
        for i in range(self.X_lda_.shape[1]):
            df.loc[X_raw.index, f"LD{i + 1}"] = self.X_lda_[:, i]
        return df

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _load_and_prepare(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Drop NAs, filter to bracket_order classes, return (X, y)."""
        if df is None:
            df = self.df
        cols_needed = self.all_features + [self.bracket_col]
        # Keep only columns that actually exist in df
        cols_present = [c for c in cols_needed if c in df.columns]
        missing = set(cols_needed) - set(cols_present)
        if missing:
            warnings.warn(
                f"LDAAnalysis: the following configured columns are missing "
                f"from the dataframe and will be skipped: {sorted(missing)}",
                stacklevel=3,
            )
        # Rebuild feature lists to only use present columns
        feats_present = [c for c in self.all_features if c in df.columns]

        df_clean = df[feats_present + [self.bracket_col]].dropna()
        df_clean = df_clean[df_clean[self.bracket_col].isin(self.bracket_order)]
        X = df_clean[feats_present].astype(float).copy()
        y = df_clean[self.bracket_col].copy()
        return X, y

    def _log1p_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """log1p-transform financial features; leave others untouched."""
        X_t = X.astype(float).copy()
        for col in self.fin_features:
            if col in X_t.columns:
                shift = abs(X_t[col].min()) if X_t[col].min() < 0 else 0
                X_t[col] = np.log1p(X_t[col] + shift)
        return X_t

    def _fit_lda(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
        """Scale → LDA; return (pipeline, projected array, explained var ratio)."""
        n_components = min(len(self.bracket_order) - 1, X.shape[1])
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lda",
                    LinearDiscriminantAnalysis(
                        n_components=n_components, solver="svd"
                    ),
                ),
            ]
        )
        X_lda = pipe.fit_transform(X, y)
        evr = pipe.named_steps["lda"].explained_variance_ratio_
        return pipe, X_lda, evr

    def _compute_class_separation(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fisher between/within variance ratio per feature."""
        rows = []
        for col in X.columns:
            overall_mean = X[col].mean()
            classes = y.unique()
            between = (
                sum(
                    (y == c).sum() * (X.loc[y == c, col].mean() - overall_mean) ** 2
                    for c in classes
                )
                / len(y)
            )
            within = (
                sum(X.loc[y == c, col].var() * (y == c).sum() for c in classes)
                / len(y)
            )
            rows.append(
                {
                    "feature": col,
                    "separation": between / within if within > 0 else 0,
                }
            )
        return pd.DataFrame(rows).sort_values("separation", ascending=False)

    @staticmethod
    def _build_sep_df(
        sep_before: pd.DataFrame, sep_after: pd.DataFrame
    ) -> pd.DataFrame:
        merged = sep_before.rename(columns={"separation": "before"}).merge(
            sep_after.rename(columns={"separation": "after"}), on="feature"
        )
        merged["improvement_%"] = (
            (merged["after"] - merged["before"])
            / merged["before"].replace(0, np.nan)
            * 100
        )
        return merged

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    def _print_report(self, evr: np.ndarray) -> None:
        print(f"\n{'=' * 55}")
        print("  LDA — Explained separation variance")
        print(f"{'=' * 55}")
        for i, v in enumerate(evr, 1):
            print(f"  LD{i}: {'█' * int(v * 40):<40}  {v * 100:.1f}%")
        print(f"{'=' * 55}\n")

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _draw_raw_hist(
        self,
        ax: plt.Axes,
        col: pd.Series,
        y: pd.Series,
        feat: str,
        show_legend: bool = False,
    ) -> None:
        col = col.astype(float)
        lo, hi = col.quantile(0.01), col.quantile(0.99)
        for bracket in self.bracket_order:
            vals = col[y == bracket].clip(lo, hi)
            ax.hist(
                vals,
                bins=60,
                density=True,
                alpha=0.55,
                color=self.bracket_colors[bracket],
                label=bracket,
            )
        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if show_legend:
            ax.legend(fontsize=7, loc="upper right")

    def _draw_ld_hist(
        self,
        ax: plt.Axes,
        X_lda: np.ndarray,
        y: pd.Series,
        evr: np.ndarray,
        ld_idx: int,
        show_legend: bool = False,
    ) -> None:
        for bracket in self.bracket_order:
            mask = (y == bracket).values
            ax.hist(
                X_lda[mask, ld_idx],
                bins=80,
                density=True,
                alpha=0.55,
                color=self.bracket_colors[bracket],
                label=bracket,
            )
        ax.set_title(
            f"LD{ld_idx + 1}  ({evr[ld_idx] * 100:.1f}% sep. var.)",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel(f"LD{ld_idx + 1} score", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if show_legend:
            ax.legend(fontsize=7, loc="upper right")

    def _draw_ld_scatter(
        self,
        ax: plt.Axes,
        X_lda: np.ndarray,
        y: pd.Series,
        evr: np.ndarray,
    ) -> None:
        for bracket in self.bracket_order:
            mask = (y == bracket).values
            ax.scatter(
                X_lda[mask, 0],
                X_lda[mask, 1],
                s=5,
                alpha=0.28,
                color=self.bracket_colors[bracket],
                label=bracket,
                rasterized=True,
            )
        ax.set_xlabel(f"LD1  ({evr[0] * 100:.1f}%)", fontsize=8)
        ld2_label = f"LD2  ({evr[1] * 100:.1f}%)" if len(evr) > 1 else "LD2"
        ax.set_ylabel(ld2_label, fontsize=8)
        ax.set_title("LD1 vs LD2 — 2-D projection", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, markerscale=3)
        ax.tick_params(labelsize=7)

    @staticmethod
    def _col_header(ax: plt.Axes, text: str, color: str) -> None:
        ax.annotate(
            text,
            xy=(0.5, 1.24),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc=color, alpha=0.85, ec="none"),
        )

    def _draw_separation_bar(
        self,
        ax: plt.Axes,
        sep_before: pd.DataFrame,
        sep_after: pd.DataFrame,
    ) -> None:
        features = sep_before["feature"].tolist()
        x, w = np.arange(len(features)), 0.35
        b_vals = sep_before.set_index("feature").loc[features, "separation"].values
        a_vals = sep_after.set_index("feature").loc[features, "separation"].values
        ax.bar(x - w / 2, b_vals, w, label="Before (raw)", color="#aab7b8")
        ax.bar(x + w / 2, a_vals, w, label="After (log1p+scaled)", color="#5dade2")
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Fisher's separation ratio", fontsize=9)
        ax.set_title(
            "Per-feature class separation — before vs after log1p transform",
            fontsize=10,
        )
        ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Main figure
    # ------------------------------------------------------------------

    def _draw_figure(
        self,
        X_raw: pd.DataFrame,
        X_t: pd.DataFrame,
        X_lda: np.ndarray,
        y: pd.Series,
        evr: np.ndarray,
        sep_before: pd.DataFrame,
        sep_after: pd.DataFrame,
    ) -> None:
        all_feats = [c for c in self.all_features if c in X_raw.columns]
        n_feats = len(all_feats)
        n_ld = X_lda.shape[1]

        NROWS = n_feats + 1  # feature rows + Fisher bar row
        NCOLS = 2

        fig, axes = plt.subplots(
            NROWS,
            NCOLS,
            figsize=(14, NROWS * 2.6),
            gridspec_kw={"hspace": 0.80, "wspace": 0.28},
        )

        # Ensure axes is always 2-D even with 1 row
        if NROWS == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle(self.title, fontsize=13, fontweight="bold", y=1.001)

        # Column header badges
        self._col_header(axes[0, 0], "◀  BEFORE  —  Raw feature distributions", "#922b21")
        self._col_header(axes[0, 1], "AFTER  —  LDA discriminant components  ▶", "#1a5276")

        # LEFT: raw histograms for every feature
        for i, feat in enumerate(all_feats):
            self._draw_raw_hist(axes[i, 0], X_raw[feat], y, feat, show_legend=(i == 0))

        # RIGHT: LD histograms
        for ld_idx in range(n_ld):
            self._draw_ld_hist(
                axes[ld_idx, 1], X_lda, y, evr, ld_idx, show_legend=(ld_idx == 0)
            )

        # RIGHT: LD1 vs LD2 scatter (only if ≥ 2 components)
        if n_ld >= 2 and n_ld < n_feats:
            self._draw_ld_scatter(axes[n_ld, 1], X_lda, y, evr)
            blank_start = n_ld + 1
        else:
            blank_start = n_ld

        # Turn off unused right-column cells
        for i in range(blank_start, n_feats):
            axes[i, 1].axis("off")

        # Fisher bar — spans both columns in the last row
        br = n_feats
        for c in range(NCOLS):
            axes[br, c].axis("off")

        pos_l = axes[br, 0].get_position()
        pos_r = axes[br, 1].get_position()
        bar_ax = fig.add_axes(
            (
                pos_l.x0,
                pos_l.y0,
                pos_r.x1 - pos_l.x0,
                pos_l.height * 1.8,
            )
        )
        self._draw_separation_bar(bar_ax, sep_before, sep_after)

        if self.output_path:
            plt.savefig(self.output_path, dpi=self.dpi, bbox_inches="tight")
            print(f"\nSaved → {self.output_path}")

        plt.show()