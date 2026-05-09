# Chart Design Guideline

**Matplotlib Visualization Style System**

A reference for producing consistent, publication-quality charts across model evaluation reports. Covers color palette, typography, layout grids, spacing tokens, and component patterns for confusion matrices, bar charts, line plots, and more.

*Design Tokens · Layout Specs · Component Patterns · Code Snippets*

---

## Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Color Palette](#2-color-palette)
3. [Typography](#3-typography)
4. [Figure & Layout Specifications](#4-figure--layout-specifications)
5. [Component: Confusion Matrix](#5-component-confusion-matrix)
6. [Component: Bar Chart](#6-component-bar-chart)
7. [Component: Line / Trend Chart](#7-component-line--trend-chart)
8. [Badges, Legends & Annotations](#8-badges-legends--annotations)
9. [Spacing & Padding Quick-Reference](#9-spacing--padding-quick-reference)
10. [Matplotlib rcParams Snippet](#10-matplotlib-rcparams-snippet)

---

## 1. Design Philosophy

Every chart should feel like it belongs in a single, cohesive report. The system is built on three principles:

- **Warmth over sterility.** A warm off-white background (`#FAFAF8`) replaces pure white, reducing glare and giving a softer, editorial feel.
- **Ink economy.** Spines are removed. Grid lines are pale and sparse. Data carries the visual weight — chrome does not.
- **Consistent accents.** Two hue families — blue (`#3A7CA5`) for primary data and coral-red (`#E8605D`) for highlights/alerts — thread through every chart type.

> When in doubt, remove an element rather than add one. If something doesn't directly help the reader interpret data, it shouldn't be in the chart.

---

## 2. Color Palette

### 2.1 Core Tokens

| Swatch | Token Name     | Hex       | Usage                                                  |
|--------|----------------|-----------|--------------------------------------------------------|
| 🔴     | SURV_COLOR     | `#E8605D` | Highlight / alert accent, badge backgrounds, survival class |
| 🟠     | SURV_LIGHT     | `#F4A9A7` | Light tint of accent — secondary highlights            |
| 🔵     | NOSURV_COLOR   | `#3A7CA5` | Primary data color, header fills, high-confidence cells |
| 🔹     | NOSURV_LIGHT   | `#8DC0DE` | Moderate-confidence cells, secondary data elements     |
| ⬜     | BG_COLOR       | `#FAFAF8` | Figure & axes background — warm off-white              |
| 🔲     | GRID_COLOR     | `#E6E6E2` | Grid lines, light borders, dividers                    |
| ⬛     | TEXT_DARK      | `#2B2B2B` | Titles, primary labels, bold annotations               |
| 🩶     | TEXT_MID       | `#6B6B6B` | Subtitles, axis labels, secondary text                 |
| ◻️     | STRIPE_COLOR   | `#F3F3F0` | Alternate-row shading in tables / bar charts           |
| 🟦     | DEEP_BLUE      | `#1B4F72` | Colormap endpoint — maximum intensity                  |

### 2.2 Colormap (Confusion Matrices)

A four-stop sequential colormap blends from `BG_COLOR` through `NOSURV_LIGHT` and `NOSURV_COLOR` to `DEEP_BLUE`. Build it with:

```python
LinearSegmentedColormap.from_list(
    "styled_blues",
    [BG_COLOR, NOSURV_LIGHT, NOSURV_COLOR, "#1B4F72"],
    N=256
)
```

Always set `vmin=0`, `vmax=1` when displaying row-normalized matrices so the colormap range is consistent across subplots.

### 2.3 Usage Rules

- Never use pure black (`#000000`) or pure white (`#FFFFFF`) — substitute `TEXT_DARK` and `BG_COLOR`.
- Limit each chart to the two hue families (blue, coral-red). If a third hue is needed, derive it from a midpoint blend or use a neutral gray.
- For cell annotations on dark backgrounds (`pct > 0.50`), switch text to white. On light backgrounds, use `TEXT_DARK`.

---

## 3. Typography

| Property              | Value                                                    |
|-----------------------|----------------------------------------------------------|
| Font stack            | Helvetica Neue → Helvetica → DejaVu Sans → Arial        |
| Figure suptitle       | `fontsize=15`, `fontweight="bold"`, `color=TEXT_DARK`    |
| Figure subtitle       | `fontsize=9.5`, `color=TEXT_MID`, `fontstyle="italic"`   |
| Subplot title         | `fontsize=10.5`, `fontweight="bold"`, `color=TEXT_DARK`  |
| Axis labels           | `fontsize=9`, `color=TEXT_MID`, `labelpad=8`             |
| Tick labels           | `fontsize=7.5`, `color=TEXT_MID`                         |
| Cell values (bold)    | `fontsize=8.5`, `fontweight="bold"`                      |
| Cell values (count)   | `fontsize=7`, `alpha=0.75`                               |
| Badge text            | `fontsize=8.5`, `fontweight="bold"`                      |
| Legend text            | `fontsize=9`                                             |

Keep the font stack consistent via `rcParams` at the top of every script. Never use serif fonts or monospace for chart labels.

---

## 4. Figure & Layout Specifications

### 4.1 Standard Figure Sizes

| Property                    | Value            |
|-----------------------------|------------------|
| Single chart (full-width)   | `figsize=(13, 6)`  |
| Side-by-side (2 charts)     | `figsize=(13, 6)`  |
| 2×2 grid or 2-row layout   | `figsize=(13, 12)` |
| Tall single chart           | `figsize=(10, 10)` |
| DPI for saved files         | 180              |
| Save facecolor              | `BG_COLOR` (`#FAFAF8`) |

### 4.2 GridSpec Defaults

| Property             | Value                                        |
|----------------------|----------------------------------------------|
| hspace (vertical gap)  | 0.28 – 0.35 (use lower end for tighter layouts) |
| wspace (horizontal gap) | 0.35 – 0.45                                  |
| left margin          | 0.08                                         |
| right margin         | 0.95                                         |
| top margin           | 0.86 (leave room for suptitle + subtitle)    |
| bottom margin        | 0.10 (leave room for legend + xlabel)        |

### 4.3 Title Placement

The suptitle sits at `y=0.96` in figure coordinates. The subtitle (italic) sits at `y=0.935`. Subplot titles are placed at `y=1.10` in axes coordinates, left-aligned. Metric badges sit at the same y but right-aligned.

Always use `fig.text()` for figure-level titles, not `fig.suptitle()`, to retain full control over positioning.

---

## 5. Component: Confusion Matrix

### 5.1 Structure

- Use `ax.imshow()` with the custom `styled_blues` colormap, `aspect='equal'`, `vmin=0`, `vmax=1`.
- Draw white grid lines (`linewidth=1.8`) between cells to create visual separation — not black borders.
- Each cell shows two lines: the normalized proportion (bold, `fontsize=8.5`) offset at `y – 0.12`, and the raw count in parentheses (`fontsize=7`, `alpha=0.75`) offset at `y + 0.22`.

### 5.2 Layout (Top-3 Grid)

For a 2-row layout (2 top + 1 centered bottom), use a 2×4 GridSpec. Top row occupies cols 0:2 and 2:4. Bottom row occupies cols 1:3 (centered). Key spacing values:

| Property             | Value              |
|----------------------|--------------------|
| GridSpec shape       | 2 rows × 4 cols    |
| hspace               | 0.28               |
| wspace               | 0.40               |
| top                  | 0.86               |
| bottom               | 0.10               |
| Tick rotation (x-axis) | 30°, `ha="right"` |
| Tick pad             | 6                  |
| Spines               | All hidden         |

### 5.3 Text Contrast Rule

When the normalized cell value exceeds `0.50`, annotation text switches to white. Below `0.50`, it uses `TEXT_DARK`. This ensures legibility against both light and dark cells of the sequential colormap.

---

## 6. Component: Bar Chart

### 6.1 Horizontal Bars (Feature Importance)

- Use `barh()` with `NOSURV_COLOR` as the primary fill. Add a thin white `edgecolor` for separation between bars.
- Alternate row backgrounds between `BG_COLOR` and `STRIPE_COLOR` using `axhspan()` to create subtle banding.
- Place value annotations to the right of each bar with a small offset (3–5 pts), using `fontsize=8`, `color=TEXT_MID`.

### 6.2 Vertical Bars

Follow the same principles: remove top and right spines, use `GRID_COLOR` for y-axis gridlines (`linewidth=0.6`, `alpha=0.7`), and keep bar widths between 0.6 and 0.8 to avoid clutter. Place value labels above bars.

### 6.3 Grouped / Stacked Bars

Use the two-hue families: `NOSURV_COLOR` and `SURV_COLOR` for the primary comparison. For additional groups, use the light variants (`NOSURV_LIGHT`, `SURV_LIGHT`). Never exceed four groups — if more are needed, consider a different chart type.

---

## 7. Component: Line / Trend Chart

| Property              | Value                                                      |
|-----------------------|------------------------------------------------------------|
| Primary line          | `NOSURV_COLOR`, `linewidth=2`, solid                       |
| Secondary line        | `SURV_COLOR`, `linewidth=2`, solid                         |
| Reference / baseline  | `TEXT_MID`, `linewidth=1`, dashed (`"--"`)                 |
| Confidence band       | `fill_between` with `alpha=0.12`, matching line color      |
| Marker style          | Circular (`"o"`), `markersize=5`, `edgecolor="white"`, `linewidth=1.5` |
| Grid                  | y-axis only, `GRID_COLOR`, `linewidth=0.6`                |
| Spines                | Only bottom and left visible, `color=GRID_COLOR`           |

For multi-line plots, keep the legend outside the plot area (`bbox_to_anchor` below or to the right) and use `frameon=False` with `fontsize=9`.

---

## 8. Badges, Legends & Annotations

### 8.1 Metric Badges

Badges are pill-shaped labels placed in the upper-right corner of each subplot to surface a key metric (e.g., F1 score, AUC). They use:

| Property    | Value                                       |
|-------------|---------------------------------------------|
| boxstyle    | `"round,pad=0.3,rounding_size=0.5"`         |
| facecolor   | `SURV_COLOR` with `alpha=0.12`              |
| edgecolor   | same as facecolor                           |
| text color  | `SURV_COLOR` (full opacity)                 |
| fontsize    | `8.5`, `fontweight="bold"`                  |
| position    | axes coords `(1.0, 1.10)`, `ha="right"`    |

### 8.2 Legends

- Use `fig.legend()` for figure-level legends, placed at `bbox_to_anchor=(0.5, -0.005)` with `loc='lower center'`.
- Always `frameon=False`. Set `columnspacing=2.5` to prevent label collision. `fontsize=9`.
- Patch handles: `handlelength=1.8`, `handleheight=1.2`.

Ensure at least 0.10 figure-coordinate units between the lowest axes content and the legend anchor to prevent overlap with x-axis labels.

### 8.3 Annotations

Use `ax.annotate()` with `arrowprops=dict(arrowstyle='->', color=TEXT_MID, lw=0.8)` for callouts. Text should use `fontsize=8`, `color=TEXT_MID`. Keep annotation lines short and unobtrusive.

---

## 9. Spacing & Padding Quick-Reference

| Property                 | Value                                          |
|--------------------------|------------------------------------------------|
| Suptitle → subtitle gap  | ~2.5% figure height (0.96 → 0.935)            |
| Subtitle → top axes      | ≥ 4.5% figure height (`top=0.86`)             |
| Axes xlabel → legend     | ≥ 10% figure height (`bottom=0.10`)           |
| Inter-row (hspace)       | 0.28 – 0.35                                   |
| Inter-col (wspace)       | 0.35 – 0.45                                   |
| Axis labelpad            | 8 pts                                          |
| Tick pad                 | 6 pts                                          |
| Grid linewidth           | 0.6, alpha 0.7                                 |
| Cell divider linewidth   | 1.8 (white)                                    |
| Spine treatment          | All hidden (`set_visible=False`)               |
| Save settings            | `bbox_inches="tight"`, `dpi=180`, `facecolor=BG_COLOR` |

---

## 10. Matplotlib rcParams Snippet

Paste this block at the top of every plotting script to enforce the design system globally:

```python
from matplotlib import rcParams

# ■■ Design Tokens ■■
SURV_COLOR   = "#E8605D"
SURV_LIGHT   = "#F4A9A7"
NOSURV_COLOR = "#3A7CA5"
NOSURV_LIGHT = "#8DC0DE"
BG_COLOR     = "#FAFAF8"
GRID_COLOR   = "#E6E6E2"
TEXT_DARK    = "#2B2B2B"
TEXT_MID     = "#6B6B6B"
STRIPE_COLOR = "#F3F3F0"

rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  [
        "Helvetica Neue", "Helvetica",
        "DejaVu Sans", "Arial"
    ],
    "axes.facecolor":   BG_COLOR,
    "figure.facecolor": BG_COLOR,
    "text.color":       TEXT_DARK,
    "axes.edgecolor":   GRID_COLOR,
    "axes.grid":        False,
    "xtick.color":      TEXT_MID,
    "ytick.color":      TEXT_MID,
})
```

---

*For questions or updates to this guideline, refer to the source chart scripts (feature importance, confusion matrices) as the canonical implementations.*
