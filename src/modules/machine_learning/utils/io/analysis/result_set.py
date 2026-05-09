# machine_learning/utils/io/analysis/result_set.py
#
# ResultSet — immutable, chainable query object over experiment results.
#
# Every filtering/ranking method returns a *new* ResultSet so chains are
# safe and predictable.  Terminal methods (.charts(), .features(), .compare(),
# .plot()) produce output without mutating the set.

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .registry import (
    ALL_MODELS,
    FAMILY_MAP,
    MODEL_DISPLAY,
    ORDINAL_BASE_MAP,
    STRATEGY_LABELS,
    STRATEGY_ORDER,
    TWO_STAGE_BASE_MAP,
    strategy_label as _build_strategy_label,
)

if TYPE_CHECKING:
    from .analyzer import ResultsAnalyzer


# ══════════════════════════════════════════════════════════════════════════════
#  CHART ITEM
# ══════════════════════════════════════════════════════════════════════════════

class ChartItem:
    """A single chart (confusion matrix, ROC curve, etc.) for one experiment."""

    __slots__ = ("experiment_id", "model", "strategy", "phase",
                 "chart_type", "data", "f1", "auc")

    def __init__(
        self,
        experiment_id: int,
        model: str,
        strategy: str,
        phase: str,
        chart_type: str,
        data,
        f1: float = 0.0,
        auc: float = 0.0,
    ) -> None:
        self.experiment_id = experiment_id
        self.model = model
        self.strategy = strategy
        self.phase = phase
        self.chart_type = chart_type
        self.data = data
        self.f1 = f1
        self.auc = auc

    @property
    def model_display(self) -> str:
        return MODEL_DISPLAY.get(self.model, self.model)

    def __repr__(self) -> str:
        return (
            f"ChartItem({self.model_display}, {self.strategy}, "
            f"{self.chart_type}, F1={self.f1:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  CHART COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

class ChartCollection:
    """
    Lazy collection of chart items for a set of experiments.

    Fetches chart data from the database only when items are accessed.
    """

    def __init__(
        self,
        result_set: ResultSet,
        chart_type: str,
        phase: str = "enhanced",
    ) -> None:
        self._result_set = result_set
        self._chart_type = chart_type
        self._phase = phase
        self._items: Optional[list[ChartItem]] = None

    def _hydrate(self) -> list[ChartItem]:
        if self._items is not None:
            return self._items

        repo = self._result_set._analyzer.repo
        df = self._result_set.df
        items: list[ChartItem] = []

        for _, row in df.iterrows():
            exp_id = int(row["id"])
            charts = repo.load_charts(exp_id, self._phase, self._chart_type)
            chart_data = charts.get(self._chart_type)
            if chart_data is None:
                continue

            items.append(ChartItem(
                experiment_id=exp_id,
                model=str(row["model"]),
                strategy=str(row.get("strategy_label", row.get("balance_strategy", ""))),
                phase=self._phase,
                chart_type=self._chart_type,
                data=chart_data,
                f1=float(row.get("enhanced_f1_macro", 0) or 0),
                auc=float(row.get("enhanced_roc_auc_macro", 0) or 0),
            ))

        self._items = items
        return items

    def __len__(self) -> int:
        return len(self._hydrate())

    def __getitem__(self, idx: int) -> ChartItem:
        return self._hydrate()[idx]

    def __iter__(self):
        return iter(self._hydrate())

    def __repr__(self) -> str:
        return f"ChartCollection({self._chart_type}, n={len(self)})"

    def plot(self, save: Optional[str] = None, **kwargs):
        """
        Render the chart collection using the appropriate visualization.

        Dispatches to confusion matrix, ROC, or PR curve plotters based on
        ``chart_type``.
        """
        from .visualization import plots
        return plots.plot_charts(self, save=save, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE RESULT
# ══════════════════════════════════════════════════════════════════════════════

class FeatureResult:
    """
    Feature importance data for a set of experiments, with helpers for
    filtering to survival/non-survival features and plotting.
    """

    def __init__(
        self,
        result_set: ResultSet,
        phase: str = "enhanced",
    ) -> None:
        self._result_set = result_set
        self._phase = phase
        self._data: Optional[dict[str, list[tuple[str, float]]]] = None

    def _hydrate(self) -> dict[str, list[tuple[str, float]]]:
        """model_slug → [(feature_name, weight), ...] sorted by weight desc."""
        if self._data is not None:
            return self._data

        from .registry import SURVIVAL_FEATURES

        repo = self._result_set._analyzer.repo
        df = self._result_set.df
        result: dict[str, list[tuple[str, float]]] = {}

        for _, row in df.iterrows():
            exp_id = int(row["id"])
            feat = repo.load_features(exp_id, self._phase)
            features = feat.get("features", [])
            weights = feat.get("weights")

            if not features:
                continue

            model_key = str(row["model"])

            if isinstance(weights, dict):
                pairs = [(f, float(weights.get(f, 0))) for f in features]
            elif isinstance(weights, list) and len(weights) == len(features):
                pairs = list(zip(features, [float(w) for w in weights]))
            else:
                pairs = [(f, 0.0) for f in features]

            pairs.sort(key=lambda x: x[1], reverse=True)
            result[model_key] = pairs

        self._data = result
        return result

    def top(self, n: int = 20) -> FeatureResult:
        """Return a new FeatureResult limited to the top-n features per model."""
        self._hydrate()
        trimmed = FeatureResult(self._result_set, self._phase)
        trimmed._data = {k: v[:n] for k, v in self._data.items()}
        return trimmed

    def survival_share(self) -> dict[str, float]:
        """Return fraction of features that are survival-derived, per model."""
        from .registry import SURVIVAL_FEATURES
        data = self._hydrate()
        shares: dict[str, float] = {}
        for model, pairs in data.items():
            total = len(pairs)
            surv = sum(1 for f, _ in pairs if f in SURVIVAL_FEATURES)
            shares[model] = surv / total if total else 0.0
        return shares

    def as_dict(self) -> dict[str, list[tuple[str, float]]]:
        """Return the raw feature data as a dict."""
        return self._hydrate()

    def plot(self, save: Optional[str] = None, **kwargs):
        """Plot feature importance bars."""
        from .visualization import plots
        return plots.plot_features(self, save=save, **kwargs)

    def __repr__(self) -> str:
        data = self._hydrate()
        models = ", ".join(MODEL_DISPLAY.get(m, m) for m in data)
        return f"FeatureResult(models=[{models}])"


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON RESULT
# ══════════════════════════════════════════════════════════════════════════════

class ComparisonResult:
    """
    Result of a statistical comparison between two ResultSets.
    """

    def __init__(
        self,
        label_a: str,
        label_b: str,
        test: str,
        statistic: float,
        p_value: float,
        extra: Optional[dict] = None,
    ) -> None:
        self.label_a = label_a
        self.label_b = label_b
        self.test = test
        self.statistic = statistic
        self.p_value = p_value
        self.significant = p_value < 0.05
        self.extra = extra or {}

    def __repr__(self) -> str:
        sig = "← SIGNIFICANT (p<0.05)" if self.significant else "(not significant)"
        lines = [
            f"{self.test}: {self.label_a} vs {self.label_b}",
            f"  statistic = {self.statistic:.4f}",
            f"  p-value   = {self.p_value:.4f}  {sig}",
        ]
        if "n01" in self.extra:
            lines.append(
                f"  Discordant: n₀₁={self.extra['n01']}, n₁₀={self.extra['n10']}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT SET
# ══════════════════════════════════════════════════════════════════════════════

class ResultSet:
    """
    Immutable filtered view of experiments.

    Every filtering/ranking method returns a new ``ResultSet`` — the
    original is never mutated.  Chain freely::

        analyzer.family("Two-Stage").top(5, by="f1").charts("roc_curve").plot()

    Parameters
    ----------
    df : pd.DataFrame
        The (possibly filtered) experiment DataFrame with metric columns.
    analyzer : ResultsAnalyzer
        Back-reference for DB access (chart hydration, feature loading).
    """

    def __init__(self, df: pd.DataFrame, analyzer: ResultsAnalyzer) -> None:
        self._df = df.copy()
        self._analyzer = analyzer

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def df(self) -> pd.DataFrame:
        """The underlying DataFrame (read-only copy)."""
        return self._df.copy()

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        cols = ["model_display", "family", "strategy_label",
                "enhanced_f1_macro", "enhanced_roc_auc_macro"]
        cols = [c for c in cols if c in self._df.columns]
        preview = self._df[cols].head(10)
        header = f"ResultSet ({len(self._df)} experiments)"
        return f"{header}\n{preview.to_string()}"

    # ── Filtering (returns new ResultSet) ─────────────────────────────────────

    def family(self, *families: str) -> ResultSet:
        """Filter to experiments belonging to the given model families."""
        mask = self._df["family"].isin(families)
        return ResultSet(self._df[mask], self._analyzer)

    def model(self, *models: str) -> ResultSet:
        """Filter to experiments for specific model slugs."""
        mask = self._df["model"].isin(models)
        return ResultSet(self._df[mask], self._analyzer)

    def strategy(self, *strategies: str) -> ResultSet:
        """Filter to experiments using specific balance strategies."""
        mask = self._df["strategy_label"].isin(strategies)
        return ResultSet(self._df[mask], self._analyzer)

    def phase(self, phase_name: str) -> ResultSet:
        """
        Return self (no-op filter) — phase selection is handled by column
        names (``enhanced_*`` vs ``baseline_*``).  This method exists for
        API symmetry and may be used in future to set a default phase for
        downstream operations.
        """
        # Phase is encoded in column names, not rows — no filtering needed.
        # Store as metadata for downstream chart/feature calls.
        rs = ResultSet(self._df, self._analyzer)
        rs._phase = phase_name
        return rs

    def where(self, predicate: Optional[Callable] = None, **kwargs) -> ResultSet:
        """
        Generic filter.

        Pass a callable that receives a DataFrame and returns a boolean mask,
        or use Django-style keyword lookups::

            rs.where(enhanced_f1_macro__gt=0.55)
            rs.where(lambda df: df["f1_lift"] > 0.05)

        Supported suffixes: ``__gt``, ``__lt``, ``__gte``, ``__lte``, ``__eq``,
        ``__ne``, ``__in``.
        """
        df = self._df

        if predicate is not None:
            mask = predicate(df)
            df = df[mask]

        for key, value in kwargs.items():
            if "__" in key:
                col, op = key.rsplit("__", 1)
            else:
                col, op = key, "eq"

            if col not in df.columns:
                raise KeyError(f"Column {col!r} not found in DataFrame")

            ops = {
                "gt":  lambda c, v: c > v,
                "lt":  lambda c, v: c < v,
                "gte": lambda c, v: c >= v,
                "lte": lambda c, v: c <= v,
                "eq":  lambda c, v: c == v,
                "ne":  lambda c, v: c != v,
                "in":  lambda c, v: c.isin(v),
            }
            if op not in ops:
                raise ValueError(f"Unknown operator {op!r}. Use: {list(ops)}")

            df = df[ops[op](df[col], value)]

        return ResultSet(df, self._analyzer)

    # ── Ranking (returns new ResultSet) ───────────────────────────────────────

    def top(self, n: int = 10, by: str = "enhanced_f1_macro") -> ResultSet:
        """Return the top-N experiments ranked by a metric column."""
        sorted_df = self._df.sort_values(by, ascending=False).head(n)
        sorted_df = sorted_df.reset_index(drop=True)
        sorted_df.index += 1  # 1-based ranking
        return ResultSet(sorted_df, self._analyzer)

    def best_per(self, group_col: str, by: str = "enhanced_f1_macro") -> ResultSet:
        """
        Return the best experiment per unique value of ``group_col``.

        Common usage::

            rs.best_per("model")           # best config per model type
            rs.best_per("family")          # best config per family
            rs.best_per("strategy_label")  # best config per strategy
        """
        idx = self._df.groupby(group_col)[by].idxmax()
        best = self._df.loc[idx].sort_values(by, ascending=False).reset_index(drop=True)
        best.index += 1
        return ResultSet(best, self._analyzer)

    # ── Aggregation ───────────────────────────────────────────────────────────

    def aggregate(
        self,
        func: str = "max",
        group_by: Sequence[str] = ("model", "strategy_label"),
    ) -> ResultSet:
        """
        Aggregate metrics across parameter sets.

        Parameters
        ----------
        func : {'max', 'mean', 'min', 'median'}
        group_by : sequence of column names
        """
        metric_cols = [c for c in self._df.columns
                       if any(m in c for m in ("f1_macro", "roc_auc_macro",
                                                "accuracy", "precision_macro",
                                                "recall_macro"))]
        group_cols = [c for c in group_by if c in self._df.columns]
        agg_df = (
            self._df.groupby(group_cols, as_index=False)[metric_cols]
            .agg(func)
        )
        # Re-attach display columns
        if "model" in agg_df.columns:
            agg_df["model_display"] = agg_df["model"].map(MODEL_DISPLAY)
            agg_df["family"] = agg_df["model"].map(FAMILY_MAP)
        return ResultSet(agg_df, self._analyzer)

    def pivot(
        self,
        index: str = "model",
        columns: str = "strategy_label",
        values: str = "enhanced_f1_macro",
        aggfunc: str = "max",
    ) -> pd.DataFrame:
        """
        Create a pivot table from the current selection.

        Returns a raw DataFrame (not a ResultSet) since pivoted data has a
        different shape. Use ``.plot(kind="heatmap")`` on the analyzer for
        visualization.
        """
        return self._df.pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
        )

    # ── Variant lift comparison ───────────────────────────────────────────────

    def variant_lift(self) -> pd.DataFrame:
        """
        Compare ordinal/two-stage variants against their base classifiers.

        Returns a DataFrame with columns: variant, base, type, metric,
        base_val, variant_val, delta.
        """
        rows: list[dict] = []

        def _best(model_name: str, metric: str) -> float:
            sub = self._df[self._df["model"] == model_name]
            return float(sub[metric].max()) if not sub.empty else float("nan")

        for mapping, variant_type in [
            (ORDINAL_BASE_MAP, "Ordinal"),
            (TWO_STAGE_BASE_MAP, "Two-Stage"),
        ]:
            for variant_model, base_model in mapping.items():
                for metric in ("enhanced_f1_macro", "enhanced_roc_auc_macro"):
                    base_val = _best(base_model, metric)
                    var_val = _best(variant_model, metric)
                    rows.append({
                        "variant":      MODEL_DISPLAY.get(variant_model, variant_model),
                        "base":         MODEL_DISPLAY.get(base_model, base_model),
                        "type":         variant_type,
                        "metric":       metric.replace("enhanced_", "").replace("_macro", ""),
                        "base_val":     base_val,
                        "variant_val":  var_val,
                        "delta":        var_val - base_val,
                    })

        return pd.DataFrame(rows)

    # ── Terminal operations ───────────────────────────────────────────────────

    def charts(
        self,
        chart_type: str,
        phase: str = "enhanced",
    ) -> ChartCollection:
        """
        Lazily fetch chart data for the selected experiments.

        Parameters
        ----------
        chart_type : {'confusion_matrix', 'roc_curve', 'pr_curve'}
        phase : {'enhanced', 'baseline'}
        """
        return ChartCollection(self, chart_type, phase)

    def features(self, phase: str = "enhanced") -> FeatureResult:
        """
        Fetch feature importance data for the selected experiments.
        """
        return FeatureResult(self, phase)

    def compare(
        self,
        other: ResultSet,
        test: str = "spearman",
        metric: str = "enhanced_f1_macro",
    ) -> ComparisonResult:
        """
        Compare this ResultSet against another using a statistical test.

        Parameters
        ----------
        other : ResultSet
            The second group to compare against.
        test : {'spearman', 'mannwhitney'}
            Statistical test to use.  ``'mcnemar'`` requires re-running
            the ML pipeline and is not yet supported in the fluent API —
            use the standalone ``run_mcnemar_pair()`` function instead.
        metric : str
            Column name of the metric to compare.
        """
        from scipy import stats

        a_vals = self._df[metric].dropna().values
        b_vals = other._df[metric].dropna().values

        label_a = self._summary_label()
        label_b = other._summary_label()

        if test == "spearman":
            result = stats.spearmanr(a_vals, b_vals)
            stat = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
            pval = float(result.pvalue) if hasattr(result, "pvalue") else float(result[1])
            return ComparisonResult(label_a, label_b, "Spearman ρ", stat, pval)

        elif test == "mannwhitney":
            stat_val, pval = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            return ComparisonResult(label_a, label_b, "Mann-Whitney U", float(stat_val), float(pval))

        else:
            raise ValueError(f"Unknown test {test!r}. Use 'spearman' or 'mannwhitney'.")

    def rank_correlation(
        self,
        col_a: str = "enhanced_f1_macro",
        col_b: str = "enhanced_roc_auc_macro",
    ) -> ComparisonResult:
        """
        Compute Spearman rank correlation between two metric columns
        within this ResultSet.
        """
        from scipy import stats

        a = self._df[col_a].dropna()
        b = self._df[col_b].dropna()
        common = a.index.intersection(b.index)
        result = stats.spearmanr(a.loc[common], b.loc[common])
        stat = float(result.statistic) if hasattr(result, "statistic") else float(result[0])
        pval = float(result.pvalue) if hasattr(result, "pvalue") else float(result[1])
        return ComparisonResult(col_a, col_b, "Spearman ρ", stat, pval)

    # ── Plot dispatch ─────────────────────────────────────────────────────────

    def plot(self, kind: Optional[str] = None, save: Optional[str] = None, **kwargs):
        """
        Render the current selection as a chart.

        When ``kind`` is ``None``, auto-selects based on the result shape:
        - Small ranked sets → horizontal bar chart
        - Aggregated data   → grouped bar chart

        Parameters
        ----------
        kind : {'bar', 'grouped_bar', 'heatmap'}, optional
        save : str, optional
            File path to save the figure.
        """
        from .visualization import plots
        return plots.plot_result_set(self, kind=kind, save=save, **kwargs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _summary_label(self) -> str:
        """Generate a short label describing this ResultSet."""
        if len(self._df) == 1:
            row = self._df.iloc[0]
            return MODEL_DISPLAY.get(str(row.get("model", "")), str(row.get("model", "")))
        families = self._df["family"].unique() if "family" in self._df.columns else []
        if len(families) == 1:
            return str(families[0])
        return f"{len(self._df)} experiments"
