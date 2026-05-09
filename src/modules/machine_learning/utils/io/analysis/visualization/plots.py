# machine_learning/utils/io/analysis/visualization/plots.py
#
# Plot dispatch and rendering functions consumed by ResultSet.plot(),
# ChartCollection.plot(), and FeatureResult.plot().
#
# Each public function receives a data object (ResultSet, ChartCollection,
# or FeatureResult) and renders the appropriate matplotlib figure.

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..registry import (
    ALL_MODELS,
    CLASS_NAMES,
    FAMILY_MAP,
    FAMILY_PALETTE,
    MODEL_DISPLAY,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    STRATEGY_PALETTE,
    SURVIVAL_FEATURES,
)
from .theme import (
    BG_COLOR,
    CM_CMAP,
    CLASS_COLORS,
    GRID_COLOR,
    LINE_COLORS,
    LINE_STYLES,
    NOSURV_COLOR,
    NOSURV_LIGHT,
    STRIPE_COLOR,
    SURV_COLOR,
    SURV_LIGHT,
    TEXT_DARK,
    TEXT_MID,
)

if TYPE_CHECKING:
    from ..result_set import ChartCollection, FeatureResult, ResultSet


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_or_show(fig, save: Optional[str], **savekw) -> None:
    if save:
        fig.savefig(save, bbox_inches="tight", dpi=savekw.get("dpi", 180),
                    facecolor=savekw.get("facecolor", BG_COLOR))
        print(f"✓ Saved {save}")
    plt.show()


def _pretty_feature(name: str) -> str:
    """Make raw feature names human-readable."""
    swaps = {
        "dtp": "DTP", "xgb": "XGB", "ada": "Ada", "rf": "RF",
        "avg": "Avg", "wavg": "W.Avg", "std": "Std",
        "cumsum": "Cumul.", "prob": "Prob", "cum": "Cum",
        "surv": "Surv", "ts": "TS",
    }
    out = name.replace("_", " ")
    for k, v in swaps.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.IGNORECASE)
    tokens = out.split()
    final = []
    for t in tokens:
        if t == t.upper() and len(t) > 1:
            final.append(t)
        elif t[0].isupper():
            final.append(t)
        else:
            final.append(t.capitalize())
    return " ".join(final)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT SET PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_result_set(
    rs: ResultSet,
    kind: Optional[str] = None,
    save: Optional[str] = None,
    metric: str = "enhanced_f1_macro",
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Auto-dispatch plot for a ResultSet.

    - ``kind="bar"`` or default for small sets → horizontal bar chart
    - ``kind="heatmap"`` → model × strategy heatmap
    - ``kind="grouped_bar"`` → grouped bar chart by strategy
    """
    if kind is None:
        kind = "bar"

    if kind == "bar":
        return _plot_ranked_bar(rs, metric, save, figsize, title, **kwargs)
    elif kind == "heatmap":
        return _plot_heatmap(rs, metric, save, figsize, title, **kwargs)
    elif kind == "grouped_bar":
        return _plot_grouped_bar(rs, metric, save, figsize, title, **kwargs)
    else:
        raise ValueError(f"Unknown plot kind {kind!r}")


def _plot_ranked_bar(
    rs: ResultSet,
    metric: str,
    save: Optional[str],
    figsize: Optional[tuple],
    title: Optional[str],
    **kwargs,
) -> plt.Figure:
    """Horizontal bar chart of ranked experiments."""
    df = rs._df.copy()
    if "model_display" not in df.columns and "model" in df.columns:
        df["model_display"] = df["model"].map(MODEL_DISPLAY)
    if "family" not in df.columns and "model" in df.columns:
        df["family"] = df["model"].map(FAMILY_MAP)

    df = df.sort_values(metric, ascending=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=figsize or (10, max(4, n * 0.4)))

    labels = []
    for _, row in df.iterrows():
        display = str(row.get("model_display", row.get("model", "")))
        strat = str(row.get("strategy_label", ""))
        labels.append(f"{display} / {strat}")

    colors = [FAMILY_PALETTE.get(str(row.get("family", "")), "#999")
              for _, row in df.iterrows()]

    ax.barh(range(n), df[metric].values, color=colors,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8.5)

    metric_label = metric.replace("_macro", "").replace("_", " ").title()
    ax.set_xlabel(metric_label)
    ax.set_title(title or f"Ranked by {metric_label}", fontweight="bold")

    # Value labels
    for i, val in enumerate(df[metric].values):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=7.5)

    # Family legend
    patches = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_PALETTE.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save)
    return fig


def _plot_heatmap(
    rs: ResultSet,
    metric: str,
    save: Optional[str],
    figsize: Optional[tuple],
    title: Optional[str],
    **kwargs,
) -> plt.Figure:
    """Model × strategy heatmap."""
    import seaborn as sns

    pivot = rs.pivot(
        index="model", columns="strategy_label", values=metric, aggfunc="max"
    )
    pivot = pivot.reindex(index=ALL_MODELS, columns=STRATEGY_ORDER)
    pivot = pivot.rename(index=MODEL_DISPLAY, columns=STRATEGY_LABELS)

    fig, ax = plt.subplots(figsize=figsize or (12, 8))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".3f",
        cmap="YlOrRd", linewidths=0.4, linecolor="white",
        annot_kws={"size": 8},
    )
    metric_label = metric.replace("_macro", "").replace("_", " ").title()
    ax.set_title(title or f"{metric_label} — Model × Strategy", fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("Balance Strategy")

    plt.tight_layout()
    _save_or_show(fig, save)
    return fig


def _plot_grouped_bar(
    rs: ResultSet,
    metric: str,
    save: Optional[str],
    figsize: Optional[tuple],
    title: Optional[str],
    **kwargs,
) -> plt.Figure:
    """Grouped bar chart: model × strategy."""
    df = rs._df.copy()
    n_models = len(ALL_MODELS)
    n_strategies = len(STRATEGY_ORDER)
    x = np.arange(n_models)
    width = 0.10
    offsets = np.linspace(
        -(n_strategies - 1) / 2 * width,
        (n_strategies - 1) / 2 * width,
        n_strategies,
    )

    fig, ax = plt.subplots(figsize=figsize or (16, 5))
    for i, strat in enumerate(STRATEGY_ORDER):
        vals = []
        for m in ALL_MODELS:
            row = df[(df["model"] == m) & (df["strategy_label"] == strat)]
            vals.append(row[metric].values[0] if len(row) else np.nan)
        color = STRATEGY_PALETTE.get(strat, "#999")
        ax.bar(x + offsets[i], vals, width,
               label=STRATEGY_LABELS.get(strat, strat),
               color=color, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in ALL_MODELS],
        rotation=35, ha="right", fontsize=8.5,
    )
    metric_label = metric.replace("_macro", "").replace("_", " ").title()
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"{metric_label} by Model & Strategy", fontweight="bold")
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save_or_show(fig, save)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  CHART COLLECTION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_charts(
    collection: ChartCollection,
    save: Optional[str] = None,
    figsize: Optional[tuple] = None,
    **kwargs,
) -> plt.Figure:
    """Dispatch to the right chart plotter."""
    chart_type = collection._chart_type
    if chart_type == "confusion_matrix":
        return _plot_confusion_matrices(collection, save, figsize, **kwargs)
    elif chart_type == "roc_curve":
        return _plot_roc_curves(collection, save, figsize, **kwargs)
    else:
        raise ValueError(f"Plotting not yet implemented for {chart_type!r}")


def _plot_confusion_matrices(
    collection: ChartCollection,
    save: Optional[str],
    figsize: Optional[tuple],
    **kwargs,
) -> plt.Figure:
    """Render confusion matrices in a grid layout."""
    items = list(collection)
    n = len(items)
    if n == 0:
        raise ValueError("No confusion matrices to plot")

    class_labels = kwargs.get("class_labels", CLASS_NAMES)

    if n <= 2:
        fig, axes = plt.subplots(1, n, figsize=figsize or (6 * n, 6), facecolor=BG_COLOR)
        if n == 1:
            axes = [axes]
    elif n == 3:
        fig = plt.figure(figsize=figsize or (13, 12), facecolor=BG_COLOR)
        gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.40,
                              left=0.08, right=0.95, top=0.86, bottom=0.10)
        axes = [
            fig.add_subplot(gs[0, 0:2]),
            fig.add_subplot(gs[0, 2:4]),
            fig.add_subplot(gs[1, 1:3]),
        ]
    else:
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig, axes_grid = plt.subplots(rows, cols,
                                       figsize=figsize or (6 * cols, 6 * rows),
                                       facecolor=BG_COLOR)
        axes = list(axes_grid.flat) if hasattr(axes_grid, "flat") else [axes_grid]

    for idx, item in enumerate(items):
        ax = axes[idx]
        cm = np.array(item.data, dtype=float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        nc = cm.shape[0]

        ax.imshow(cm_norm, cmap=CM_CMAP, vmin=0, vmax=1, aspect="equal")

        for edge in range(nc + 1):
            ax.axhline(edge - 0.5, color="white", linewidth=1.8, zorder=2)
            ax.axvline(edge - 0.5, color="white", linewidth=1.8, zorder=2)

        for i in range(nc):
            for j in range(nc):
                pct = cm_norm[i, j]
                count = int(cm[i, j])
                color = "white" if pct > 0.50 else TEXT_DARK
                ax.text(j, i - 0.12, f"{pct:.2f}", ha="center", va="center",
                        color=color, fontsize=8.5, fontweight="bold", zorder=3)
                ax.text(j, i + 0.22, f"({count})", ha="center", va="center",
                        color=color, fontsize=7, alpha=0.75, zorder=3)

        clean = [re.sub(r"\s*\(\d+\)\s*$", "", c) for c in class_labels[:nc]]
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(clean, rotation=30, ha="right", fontsize=7.5, color=TEXT_MID)
        ax.set_yticklabels(clean, fontsize=7.5, color=TEXT_MID)
        ax.tick_params(axis="both", length=0, pad=6)
        ax.set_xlabel("Predicted", fontsize=9, color=TEXT_MID, labelpad=8)
        ax.set_ylabel("Actual", fontsize=9, color=TEXT_MID, labelpad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

        title_text = f"#{idx+1}  {item.model_display}  ·  {item.strategy}"
        ax.set_title(title_text, fontsize=10.5, fontweight="bold",
                     color=TEXT_DARK, pad=20, loc="left")

    fig.suptitle(f"Confusion Matrices — Top-{n}", fontsize=15,
                 fontweight="bold", color=TEXT_DARK, y=0.98)

    plt.tight_layout()
    _save_or_show(fig, save, facecolor=BG_COLOR)
    return fig


def _plot_roc_curves(
    collection: ChartCollection,
    save: Optional[str],
    figsize: Optional[tuple],
    **kwargs,
) -> plt.Figure:
    """Render ROC curves for the selected experiments."""
    items = list(collection)
    n = len(items)
    if n == 0:
        raise ValueError("No ROC curves to plot")

    FPR_GRID = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(figsize=figsize or (8, 8), facecolor=BG_COLOR)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=TEXT_MID,
            label="Random (AUC = 0.50)")

    for idx, item in enumerate(items):
        roc_data = item.data
        if not isinstance(roc_data, dict):
            continue

        tpr_interps = []
        for cls_key in sorted(roc_data.keys(), key=lambda k: int(k)):
            cls_data = roc_data[cls_key]
            tpr_interps.append(
                np.interp(FPR_GRID, np.array(cls_data["fpr"]),
                          np.array(cls_data["tpr"]))
            )
        macro_tpr = np.mean(tpr_interps, axis=0)

        label = (f"{item.model_display} / {item.strategy}"
                 f"\n(AUC = {item.auc:.4f})")
        ax.plot(
            FPR_GRID, macro_tpr,
            linestyle=LINE_STYLES[idx % len(LINE_STYLES)],
            color=LINE_COLORS[idx % len(LINE_COLORS)],
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("False Positive Rate", fontsize=11, color=TEXT_MID, labelpad=8)
    ax.set_ylabel("True Positive Rate", fontsize=11, color=TEXT_MID, labelpad=8)
    ax.set_title("Macro-Avg OvR ROC", fontsize=12.5, fontweight="bold",
                 color=TEXT_DARK, loc="left", pad=10)
    ax.legend(fontsize=10, loc="lower right", frameon=False)

    plt.tight_layout()
    _save_or_show(fig, save, facecolor=BG_COLOR)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE RESULT PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_features(
    feat: FeatureResult,
    save: Optional[str] = None,
    figsize: Optional[tuple] = None,
    highlight_survival: bool = True,
    **kwargs,
) -> plt.Figure:
    """Render feature importance horizontal bar charts in a grid."""
    import matplotlib.transforms as mtransforms

    data = feat.as_dict()
    models = list(data.keys())
    n = len(models)
    if n == 0:
        raise ValueError("No feature data to plot")

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=figsize or (14, 7 * rows),
                              facecolor=BG_COLOR)
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for idx, model_key in enumerate(models):
        ax = axes_flat[idx]
        pairs = data[model_key]
        features_raw = [f for f, _ in pairs]
        weights = np.array([w for _, w in pairs], dtype=float)
        if weights.max() > 0:
            weights = weights / weights.max()

        n_feats = len(features_raw)
        y_pos = np.arange(n_feats)

        # Alternating stripes
        for i in range(n_feats):
            if i % 2 == 0:
                ax.axhspan(i - 0.4, i + 0.4, color=STRIPE_COLOR, zorder=0)

        is_surv = [f in SURVIVAL_FEATURES for f in features_raw]

        for i, (w, surv) in enumerate(zip(weights, is_surv)):
            base_c = SURV_COLOR if (surv and highlight_survival) else NOSURV_COLOR
            light_c = SURV_LIGHT if (surv and highlight_survival) else NOSURV_LIGHT
            ax.barh(i, w, height=0.62, color=light_c, alpha=0.35, zorder=1, left=0.002)
            ax.barh(i, w, height=0.58, color=base_c, alpha=0.88, zorder=2,
                    edgecolor="white", linewidth=0.4)
            if w >= 0.05:
                ax.text(w + 0.015, i, f"{w:.2f}", va="center", fontsize=7,
                        color=TEXT_MID, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([""] * n_feats)
        ax.tick_params(axis="y", length=0, pad=0)
        ax.invert_yaxis()

        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for rank, (f, y) in enumerate(zip(features_raw, y_pos), 1):
            marker = " ●" if (f in SURVIVAL_FEATURES and highlight_survival) else ""
            label = f"{rank:>2}.  {_pretty_feature(f)}{marker}"
            ax.text(-0.38, y, label, transform=trans,
                    ha="left", va="center", fontsize=7.5,
                    fontfamily="monospace", color=TEXT_DARK,
                    clip_on=False, zorder=5)

        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
        ax.set_xticklabels(["0", ".25", ".50", ".75", "1.0"],
                           fontsize=7.5, color=TEXT_MID)
        ax.xaxis.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        display = MODEL_DISPLAY.get(model_key, model_key)
        ax.set_title(display, fontsize=11, fontweight="bold",
                     color=TEXT_DARK, pad=20, loc="left")

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Legend
    if highlight_survival:
        patches = [
            mpatches.Patch(facecolor=SURV_COLOR, alpha=0.88, edgecolor="white",
                           label="  Survival features (Cox PH)"),
            mpatches.Patch(facecolor=NOSURV_COLOR, alpha=0.88, edgecolor="white",
                           label="  Non-survival features"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=2,
                   fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Feature Importances", ha="center", fontsize=15,
                 fontweight="bold", color=TEXT_DARK, y=1.01)

    plt.tight_layout(h_pad=3.5, w_pad=2.5, rect=[0, 0.02, 1, 0.97])
    _save_or_show(fig, save, facecolor=BG_COLOR)
    return fig
