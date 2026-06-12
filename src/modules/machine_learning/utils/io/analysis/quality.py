# machine_learning/utils/io/analysis/quality.py
#
# Data quality checks extracted from Section 2 of ML_Results_Analysis.py.
# Returns a structured QualityReport instead of printing to stdout.

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .registry import STRATEGY_ORDER, ALL_MODELS, MODEL_DISPLAY


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY REPORT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityReport:
    """
    Structured result of running data-quality checks on an experiment DataFrame.

    Attributes
    ----------
    passed : bool
        ``True`` when no issues were found.
    issues : list[str]
        Human-readable descriptions of every issue detected.
    null_counts : dict[str, int]
        Number of NaN values per primary metric column.
    strategy_counts : pd.Series
        Experiment count per balance strategy.
    model_counts : pd.Series
        Experiment count per model type.
    hybrid_nulls : int
        Number of hybrid rows with a NULL ``undersample_threshold``.
    mean_lift : float
        Mean ``enhanced_f1_macro - baseline_f1_macro`` across experiments.
    negative_lifts : int
        Number of experiments with F1 lift below −0.02.
    """

    passed:           bool           = True
    issues:           list[str]      = field(default_factory=list)
    null_counts:      dict[str, int] = field(default_factory=dict)
    strategy_counts:  pd.Series      = field(default_factory=lambda: pd.Series(dtype=int))
    model_counts:     pd.Series      = field(default_factory=lambda: pd.Series(dtype=int))
    hybrid_nulls:     int            = 0
    mean_lift:        float          = 0.0
    negative_lifts:   int            = 0

    def __repr__(self) -> str:
        status = "✓ PASSED" if self.passed else f"⚠ {len(self.issues)} ISSUE(S)"
        lines = [f"QualityReport: {status}"]
        if self.issues:
            for iss in self.issues:
                lines.append(f"  · {iss}")
        lines.append(f"  Mean F1 lift: {self.mean_lift:+.4f}")
        lines.append(f"  Negative lifts (< −0.02): {self.negative_lifts}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_quality_checks(df: pd.DataFrame) -> QualityReport:
    """
    Run all quality checks against an experiment DataFrame and return a
    :class:`QualityReport`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at minimum: ``enhanced_f1_macro``, ``enhanced_roc_auc_macro``,
        ``baseline_f1_macro``, ``baseline_roc_auc_macro``, ``balance_strategy``,
        ``undersample_threshold``, ``model``, ``strategy_label``, ``f1_lift``.
    """
    report = QualityReport()
    issues: list[str] = []

    # ── NaN check on primary metrics ──────────────────────────────────────────
    for col in [
        "enhanced_f1_macro", "enhanced_roc_auc_macro",
        "baseline_f1_macro", "baseline_roc_auc_macro",
    ]:
        n_null = int(df[col].isna().sum()) if col in df.columns else -1
        report.null_counts[col] = n_null
        if n_null > 0:
            issues.append(f"NaN in {col}: {n_null} rows")

    # ── Row counts per strategy ───────────────────────────────────────────────
    if "strategy_label" in df.columns:
        report.strategy_counts = (
            df.groupby("strategy_label").size()
              .reindex(STRATEGY_ORDER, fill_value=0)
        )

    # ── Row counts per model ──────────────────────────────────────────────────
    if "model" in df.columns:
        report.model_counts = (
            df.groupby("model").size()
              .reindex(ALL_MODELS, fill_value=0)
        )
        for m, cnt in report.model_counts.items():
            if cnt == 0:
                issues.append(f"Model {MODEL_DISPLAY.get(str(m), str(m))}: ZERO rows")

    # ── Hybrid NULL check ─────────────────────────────────────────────────────
    if "balance_strategy" in df.columns and "undersample_threshold" in df.columns:
        hybrid_mask = df["balance_strategy"] == "hybrid"
        report.hybrid_nulls = int(
            df.loc[hybrid_mask, "undersample_threshold"].isna().sum()
        )
        if report.hybrid_nulls > 0:
            issues.append(f"Hybrid threshold NULL: {report.hybrid_nulls}")

    # ── Lift check ────────────────────────────────────────────────────────────
    if "f1_lift" in df.columns:
        report.mean_lift = float(df["f1_lift"].mean())
        report.negative_lifts = int((df["f1_lift"] < -0.02).sum())
        if report.negative_lifts > 50:
            issues.append(f"Many rows with negative F1 lift: {report.negative_lifts}")

    report.issues = issues
    report.passed = len(issues) == 0
    return report
