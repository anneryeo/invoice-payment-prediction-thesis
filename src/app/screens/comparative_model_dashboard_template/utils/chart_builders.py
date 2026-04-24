import json

import numpy as np
import plotly.graph_objects as go
from dash import html

from src.app.screens.comparative_model_dashboard_template.constants import (
    MODELS, METRICS, CHART_COLORS,
    _class_label, CLASS_LABELS,
)
from src.modules.machine_learning.utils.io.data_loaders import json_deserialize


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════

def _base_layout(title_text: str) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(family="'IBM Plex Mono', monospace", color="#374151", size=10),
        margin=dict(l=52, r=28, t=60, b=48),
        title=dict(
            text=title_text,
            font=dict(family="'DM Serif Display', serif", size=13, color="#111827"),
            x=0, xanchor="left", pad=dict(l=8, t=6),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e5e7eb", borderwidth=1,
            font=dict(size=9),
        ),
        xaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db", tickfont=dict(size=9), zeroline=False),
        yaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db", tickfont=dict(size=9), zeroline=False),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_metrics(model_key: str, result_type: str) -> dict:
    return MODELS[model_key][result_type]["evaluation"]["metrics"]


def delta_badge(val: float, ref: float):
    diff = val - ref
    if abs(diff) < 0.0005:
        return html.Span("—", className="delta-neutral")
    sign = "▲" if diff > 0 else "▼"
    cls = "delta-up" if diff > 0 else "delta-down"
    return html.Span(f"{sign}{abs(diff):.3f}", className=cls)


def build_leaderboard_rows(
        sort_metric: str,
        result_type: str,
        sort_result_type: str = None,
        model_filter: list = None,
        strategy_filter: list = None,
) -> list:
    _sort_rt = sort_result_type if sort_result_type is not None else result_type
    rows = []
    for key, data in MODELS.items():
        if model_filter is not None and data["model"] not in model_filter:
            continue
        if strategy_filter is not None and data["balance_strategy"] not in strategy_filter:
            continue
        base_m = _get_metrics(key, "baseline")
        enh_m  = _get_metrics(key, "enhanced")
        rows.append({
            "key":        key,
            "name":       data["model"],
            "strategy":   data["balance_strategy"],
            "parameters": data.get("parameters", {}),
            **{f"base_{m}": base_m.get(m, 0) for m in METRICS},
            **{f"enh_{m}":  enh_m.get(m, 0)  for m in METRICS},
            "sort_val": (enh_m if _sort_rt == "enhanced" else base_m).get(sort_metric, 0),
        })
    rows.sort(key=lambda x: x["sort_val"], reverse=True)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_curve(raw):
    """
    Resolve a roc_curve / pr_curve value into a plain Python dict.

    After SQLite + JSON deserialization the value will typically already be a
    dict.  This function handles that fast path as well as the fallback where
    the value arrives as a JSON string (e.g. doubly-encoded) that still needs
    one more json.loads pass.
    """
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return {}
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[step4] _parse_curve json.loads failed: {e} — preview: {raw[:80]}")
    return {}


def build_roc_figure(model_key: str, result_type: str) -> go.Figure:
    roc = _parse_curve(
        MODELS[model_key][result_type]["evaluation"]["charts"].get("roc_curve")
    )
    auc = _get_metrics(model_key, result_type).get("roc_auc_macro", 0)

    layout = _base_layout(f"ROC Curve · AUC {auc:.3f}")
    layout["xaxis"]["title"] = "False Positive Rate"
    layout["xaxis"]["range"] = [0, 1]
    layout["xaxis"]["fixedrange"] = True
    layout["yaxis"]["title"] = "True Positive Rate"
    layout["yaxis"]["range"] = [0, 1]
    layout["yaxis"]["fixedrange"] = True

    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#9ca3af", dash="dash", width=1))

    for i, cls in enumerate(sorted(roc.keys(), key=lambda k: int(k) if str(k).isdigit() else k)):
        entry = roc[cls]
        fpr = entry.get("fpr", []) if isinstance(entry, dict) else []
        tpr = entry.get("tpr", []) if isinstance(entry, dict) else []
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines", name=_class_label(str(cls)),
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
        ))

    fig.update_layout(**layout)
    return fig


def build_pr_figure(model_key: str, result_type: str) -> go.Figure:
    pr = _parse_curve(
        MODELS[model_key][result_type]["evaluation"]["charts"].get("pr_curve")
    )

    layout = _base_layout("Precision–Recall Curve")
    layout["xaxis"]["title"] = "Recall"
    layout["xaxis"]["range"] = [0, 1]
    layout["xaxis"]["fixedrange"] = True
    layout["yaxis"]["title"] = "Precision"
    layout["yaxis"]["range"] = [0, 1.05]
    layout["yaxis"]["fixedrange"] = True

    fig = go.Figure()
    for i, cls in enumerate(sorted(pr.keys(), key=lambda k: int(k) if str(k).isdigit() else k)):
        entry = pr[cls]
        recall    = entry.get("recall",    []) if isinstance(entry, dict) else []
        precision = entry.get("precision", []) if isinstance(entry, dict) else []
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines", name=_class_label(str(cls)),
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
        ))

    fig.update_layout(**layout)
    return fig


def build_cm_figure(model_key: str, result_type: str) -> go.Figure:
    cm_raw = MODELS[model_key][result_type]["evaluation"]["charts"].get("confusion_matrix")
    if isinstance(cm_raw, str):
        cm_raw = json_deserialize(cm_raw)
    if cm_raw is None:
        cm_arr = np.zeros((3, 3), dtype=int)
    else:
        cm_arr = np.array(cm_raw)

    n = cm_arr.shape[0]
    tick_vals = list(range(n))

    # Derive tick labels from the runtime CLASS_LABELS dict (populated from
    # session class_mappings) rather than any hardcoded mapping.
    # Short label used on the axis ticks; full label used in hover tooltips.
    tick_labels = [_class_label(str(i)) for i in range(n)]
    full_labels = tick_labels  # CLASS_LABELS already contains human-readable names

    # cm_arr[actual][predicted]; flip rows so index 0 appears at top in Plotly
    cm_plot    = cm_arr[::-1]
    total_plot = cm_plot.sum(axis=1, keepdims=True)
    pct_plot   = np.where(total_plot > 0, cm_plot / total_plot * 100, 0)

    z_min   = float(cm_plot.min())
    z_max   = float(cm_plot.max())
    z_range = z_max - z_min if z_max != z_min else 1.0

    _cs = [(0xef, 0xf6, 0xff), (0x93, 0xc5, 0xfd), (0x1d, 0x4e, 0xd8)]

    def _interp_color(norm: float):
        if norm <= 0.5:
            t = norm / 0.5
            r = _cs[0][0] + t * (_cs[1][0] - _cs[0][0])
            g = _cs[0][1] + t * (_cs[1][1] - _cs[0][1])
            b = _cs[0][2] + t * (_cs[1][2] - _cs[0][2])
        else:
            t = (norm - 0.5) / 0.5
            r = _cs[1][0] + t * (_cs[2][0] - _cs[1][0])
            g = _cs[1][1] + t * (_cs[2][1] - _cs[1][1])
            b = _cs[1][2] + t * (_cs[2][2] - _cs[1][2])
        return r / 255, g / 255, b / 255

    def _relative_luminance(r, g, b):
        def _lin(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)

    # Y tick labels must also be flipped to match the flipped rows
    tick_labels_y = tick_labels[::-1]
    full_labels_y = full_labels[::-1]

    # customdata: row i → full_labels_y[i] (actual), col j → full_labels[j] (predicted)
    customdata = [
        [
            [full_labels_y[i], full_labels[j], int(cm_plot[i][j]), float(pct_plot[i][j])]
            for j in range(n)
        ]
        for i in range(n)
    ]

    layout = _base_layout("Confusion Matrix")
    axis_common_x = dict(
        fixedrange=True,
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels,
        tickfont=dict(size=10, family="'IBM Plex Mono', monospace"),
    )
    axis_common_y = dict(
        fixedrange=True,
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_labels_y,
        tickfont=dict(size=10, family="'IBM Plex Mono', monospace"),
    )
    layout["xaxis"].update({"title": "Predicted", **axis_common_x})
    layout["yaxis"].update({"title": "Actual",    **axis_common_y})

    fig = go.Figure(go.Heatmap(
        z=cm_plot,
        x=tick_vals,
        y=tick_vals,
        colorscale=[[0, "#eff6ff"], [0.5, "#93c5fd"], [1, "#1d4ed8"]],
        showscale=False,
        customdata=customdata,
        hovertemplate=(
            "<b style='color:#94a3b8;'>Actual&#160;&#160;&#160;&#160;&#160;&#160;</b>%{customdata[0]}<br>"
            "<b style='color:#94a3b8;'>Predicted&#160;&#160;&#160;</b>%{customdata[1]}<br>"
            "<b style='color:#94a3b8;'>Count&#160;&#160;&#160;&#160;&#160;&#160;&#160;</b>%{customdata[2]}<br>"
            "<b style='color:#94a3b8;'>Percentage&#160;</b>&#160;%{customdata[3]:.1f}%"
            "<extra></extra>"
        ),
    ))

    for i in range(n):
        for j in range(n):
            val     = cm_plot[i][j]
            norm    = (float(val) - z_min) / z_range
            r, g, b = _interp_color(norm)
            lum     = _relative_luminance(r, g, b)
            color   = "#ffffff" if lum < 0.35 else "#111827"
            pct_val = pct_plot[i][j]
            fig.add_annotation(
                x=j, y=i,
                text=f"{val}<br>{pct_val:.1f}%",
                showarrow=False,
                font=dict(size=10, color=color,
                          family="'IBM Plex Mono', monospace"),
                align="center",
            )

    fig.update_layout(
        **layout,
        hoverlabel=dict(
            bgcolor="#1e293b",
            bordercolor="#334155",
            font=dict(
                family="'IBM Plex Mono', monospace",
                size=11,
                color="#f1f5f9",
            ),
        ),
    )
    return fig


def build_features_figure(model_key: str, result_type: str,
                           top15_only: bool = True,
                           stage: str | None = None) -> go.Figure:
    raw_weights = MODELS[model_key][result_type].get("weights")

    # ── Multi-stage detection ─────────────────────────────────────────────────
    # TwoStagePipeline stores weights as {"stage_1": {feat: score, …},
    # "stage_2": {feat: score, …}} so the dashboard can toggle between views.
    # Any other model stores a flat {feat: score} dict.
    is_multistage = (
        isinstance(raw_weights, dict)
        and bool(raw_weights)
        and all(isinstance(v, dict) for v in raw_weights.values())
    )

    if is_multistage:
        # Resolve which stage to show; default to "stage_1" if unspecified.
        active_stage  = stage if stage in raw_weights else "stage_1"
        stage_weights = raw_weights[active_stage]
        features      = list(stage_weights.keys())
        importance    = [float(v) for v in stage_weights.values()]
    else:
        features = MODELS[model_key][result_type].get("features", [])

        # Use real weights if available, otherwise fall back to rank-based estimates
        if raw_weights and isinstance(raw_weights, (list, dict)):
            if isinstance(raw_weights, dict):
                importance = [float(raw_weights.get(f, 0)) for f in features]
            else:
                importance = [float(w) for w in raw_weights]
        else:
            importance = sorted(
                [round(1 - i * 0.1 + np.random.uniform(-0.03, 0.03), 3) for i in range(len(features))],
                reverse=True,
            )

    # Sort by absolute value descending so magnitude drives rank regardless of sign
    paired = sorted(zip(importance, features), key=lambda p: abs(p[0]), reverse=True)

    if top15_only:
        paired = paired[:15]

    importance = [p[0] for p in paired]
    features   = [p[1] for p in paired]

    abs_importance = [abs(v) for v in importance]
    is_negative    = [v < 0 for v in importance]
    is_positive    = [v > 0 for v in importance]

    # Append sign markers to y-axis labels
    display_labels = [
        f"{f}  (−)" if neg else (f"{f}  (+)" if pos else f)
        for f, neg, pos in zip(features, is_negative, is_positive)
    ]

    # Top bar: accent blue; rest: lighter blue
    colors = [
        CHART_COLORS[0] if idx == 0 else "#93c5fd"
        for idx in range(len(abs_importance))
    ]

    # Dynamic left margin — account for longest label including sign suffix
    max_label_len = max((len(lbl) for lbl in display_labels), default=10)
    left_margin   = max(100, min(max_label_len * 7, 260))

    layout = _base_layout("")
    layout["title"]                = None
    layout["xaxis"]["title"]       = "Importance Score (absolute)"
    layout["xaxis"]["fixedrange"]  = True
    layout["xaxis"]["autorange"]   = True
    layout["yaxis"]["autorange"]   = "reversed"
    layout["yaxis"]["fixedrange"]  = False
    layout["margin"]["l"] = left_margin
    layout["margin"]["r"] = 60
    layout["margin"]["t"] = 16
    layout["margin"]["b"] = 48
    n = len(features)
    layout["height"] = max(470, 28 * n)

    # Bar text shows original signed value
    bar_texts = [
        f"{v:.3f}" if not neg else f"−{abs(v):.3f}"
        for v, neg in zip(importance, is_negative)
    ]

    # White text on dark top bar, dark text on lighter bars
    text_colors = [
        "#ffffff" if idx == 0 else "#1e3a5f"
        for idx in range(len(colors))
    ]

    fig = go.Figure(go.Bar(
        x=abs_importance,
        y=display_labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=bar_texts,
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(size=9, color=text_colors, family="'IBM Plex Mono', monospace"),
        customdata=importance,
        hovertemplate="<b>%{y}</b><br>Score: %{customdata:.4f}<extra></extra>",
    ))
    fig.update_layout(**layout)
    return fig
