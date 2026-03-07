import os
import sys
import warnings
import numpy as np
import pandas as pd
from contextlib import contextmanager
from itertools import product

from lifelines import CoxPHFitter
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored
from tqdm import tqdm

from machine_learning.utils.data.clean_survival_inputs import clean_survival_inputs


# ── module-level helpers ────────────────────────────────────────────────────
# These must be module-level (not methods) so joblib can pickle them for
# parallel workers.

@contextmanager
def _suppress_fd_stderr():
    """
    Redirect OS-level stderr (fd=2) to /dev/null for the duration of the
    context. Silences Intel MKL/LAPACK messages (e.g. DGELSY) that are
    written directly to the OS file descriptor and bypass Python's warning
    and exception systems entirely.
    """
    original_fd = sys.stderr.fileno()
    saved_fd    = os.dup(original_fd)
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, original_fd)
        os.close(devnull_fd)
        yield
    finally:
        os.dup2(saved_fd, original_fd)
        os.close(saved_fd)


def _evaluate_params(df_fit, pen, l1, method, robust, step, kf_splits, extra_kwargs=None):
    """
    Evaluate one hyperparameter combination across all CV folds.
    Module-level so joblib can pickle it for parallel workers.

    Returns
    -------
    mean_c_index : float or None
    params : dict or None
    diag : dict
        Diagnostic counters for the Excel report.
    """
    extra_kwargs      = extra_kwargs or {}
    c_indices         = []
    n_folds_failed    = 0
    n_overflow        = 0
    n_nan_scores      = 0
    n_inf_scores      = 0
    fit_exception_msg = ""

    for train_idx, val_idx in kf_splits:
        df_train = df_fit.iloc[train_idx]
        df_val   = df_fit.iloc[val_idx]

        try:
            cph = CoxPHFitter(
                penalizer=pen,
                l1_ratio=l1,
                baseline_estimation_method=method,
                **extra_kwargs
            )

            with _suppress_fd_stderr(), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cph.fit(
                    df_train,
                    duration_col="T",
                    event_col="E",
                    robust=robust,
                    fit_options={"step_size": step}
                )

            n_overflow += sum(
                1 for w in caught
                if issubclass(w.category, RuntimeWarning)
                and "overflow" in str(w.message).lower()
            )

            # spline exposes predict_log_partial_hazard, not predict_partial_hazard
            if method == "spline":
                log_risk    = cph.predict_log_partial_hazard(df_val)
                risk_scores = np.exp(log_risk.clip(upper=500))
            else:
                risk_scores = cph.predict_partial_hazard(df_val)

            n_nan_scores += int(risk_scores.isna().sum())
            n_inf_scores += int(np.isinf(risk_scores).sum())

            valid_mask = risk_scores.notna() & np.isfinite(risk_scores)
            if valid_mask.sum() < 2:
                n_folds_failed += 1
                continue

            c_index = concordance_index_censored(
                df_val.loc[valid_mask.values, "E"].astype(bool),
                df_val.loc[valid_mask.values, "T"],
                risk_scores[valid_mask].values
            )[0]
            c_indices.append(c_index)

        except Exception as exc:
            n_folds_failed += 1
            msg = str(exc)
            if "DGELSY" in msg or "Parameter 4" in msg:
                fit_exception_msg = f"MKL_DGELSY: rank-deficient spline matrix. {msg[:80]}"
            else:
                fit_exception_msg = msg[:120]
            continue

    diag = {
        "n_folds_ok":          len(c_indices),
        "n_folds_failed":      n_folds_failed,
        "n_overflow_warnings": n_overflow,
        "n_nan_risk_scores":   n_nan_scores,
        "n_inf_risk_scores":   n_inf_scores,
        "c_index_std":         float(np.std(c_indices)) if len(c_indices) > 1 else np.nan,
        "fit_exception":       fit_exception_msg,
    }

    if not c_indices:
        return None, None, diag

    return float(np.mean(c_indices)), {
        "penalizer": pen, "l1_ratio": l1,
        "baseline_estimation_method": method,
        "robust": robust, "step_size": step,
        **extra_kwargs,
    }, diag


# ── main class ──────────────────────────────────────────────────────────────

class CoxHyperparameterTuner:
    """
    Tunes CoxPHFitter hyperparameters via parallelized cross-validation with
    pre-flight data validation, spline dry-run blacklisting, and an optional
    Excel diagnostics report.

    Usage
    -----
    tuner = CoxHyperparameterTuner(save_report_path="results/cox_tuning.xlsx")
    tuner.fit(X_surv, T, E)

    print(tuner.best_params_)
    print(tuner.best_c_index_)
    print(tuner.results_df_)

    Parameters
    ----------
    penalizer_grid : list of float
    l1_ratios : list of float
    baseline_methods : list of str
        Any of "breslow", "spline", "efron". "efron" is auto-removed when
        tied_obs_ratio > efron_obs_threshold. "spline" knots are validated
        via dry-run before the grid search.
    robust_options : list of bool
    step_sizes : list of float
    n_baseline_knots_grid : list of int
        Knot counts for spline. Royston et al. recommend 4; range 2–8.
    spline_min_penalizer : float
        Spline combos with penalizer below this are skipped (divergence risk).
    efron_obs_threshold : float
        Fraction of events involved in ties above which efron is removed.
    n_splits : int
    random_state : int
    n_jobs : int
        Joblib parallel workers. -1 = all cores.
    save_report_path : str or None
        Path for the Excel report. Accepts directory or full .xlsx path.
    """

    # ── constants ────────────────────────────────────────────────────────────
    _SPLINE_MIN_PENALIZER  = 0.1
    _EFRON_OBS_THRESHOLD   = 0.03

    def __init__(
        self,
        penalizer_grid         = [0.001, 0.01, 0.1, 1, 10, 100],
        l1_ratios              = [0, 0.25, 0.5, 0.75, 1],
        baseline_methods       = ["breslow", "spline"],
        robust_options         = [True, False],
        step_sizes             = [0.5, 0.75, 0.95],
        n_baseline_knots_grid  = [2, 3, 4, 6, 8],
        spline_min_penalizer   = 0.1,
        efron_obs_threshold    = 0.03,
        n_splits               = 5,
        random_state           = 42,
        n_jobs                 = -1,
        save_report_path       = None,
    ):
        self.penalizer_grid        = penalizer_grid
        self.l1_ratios             = l1_ratios
        self.baseline_methods      = baseline_methods
        self.robust_options        = robust_options
        self.step_sizes            = step_sizes
        self.n_baseline_knots_grid = n_baseline_knots_grid
        self.spline_min_penalizer  = spline_min_penalizer
        self.efron_obs_threshold   = efron_obs_threshold
        self.n_splits              = n_splits
        self.random_state          = random_state
        self.n_jobs                = n_jobs
        self.save_report_path      = save_report_path

        # Set after fit()
        self.best_params_    = None
        self.best_c_index_   = None
        self.results_df_     = None
        self.safe_methods_   = None
        self.safe_knots_grid_ = None
        self.tie_ratio_      = None
        self._df_fit         = None
        self._kf_splits      = None

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, X_surv, T, E):
        """
        Run the full tuning pipeline.

        Steps
        -----
        1. Clean and validate input data
        2. Check for tied event times → may remove efron
        3. Build CV folds
        4. Dry-run validate spline knot counts → blacklist bad knots
        5. Build hyperparameter grid
        6. Parallel CV search
        7. Collect results and select best params
        8. Save Excel report (if save_report_path set)

        Returns
        -------
        self  (allows chaining: tuner.fit(X, T, E).best_params_)
        """
        self._df_fit = self._clean_data(X_surv, T, E)
        self.safe_methods_ = self._check_ties()
        self._kf_splits    = self._build_folds()
        self.safe_knots_grid_ = self._validate_spline()
        combos             = self._build_grid()
        self._run_search(combos)

        if self.save_report_path:
            self._save_report()

        return self

    # ── pipeline steps ───────────────────────────────────────────────────────

    def _clean_data(self, X_surv, T, E) -> pd.DataFrame:
        """Clean inputs and return a ready-to-fit df_fit."""
        _, _, _, df_fit = clean_survival_inputs(X_surv, T, E)
        return df_fit

    def _check_ties(self) -> list:
        """
        Detect tied event times and remove efron if tied_obs_ratio exceeds
        the threshold. Returns the list of safe baseline methods.
        """
        event_times  = self._df_fit.loc[self._df_fit["E"] == 1, "T"]
        n_events     = len(event_times)
        time_counts  = event_times.value_counts()
        n_tied_times = int((time_counts > 1).sum())
        n_tied_obs   = int(time_counts[time_counts > 1].sum())

        tied_time_ratio  = n_tied_times / max(n_events, 1)
        self.tie_ratio_  = n_tied_obs   / max(n_events, 1)

        print(
            f"[CoxHyperparameterTuner] Tie check: {n_events} events | "
            f"{n_tied_times} tied time points ({tied_time_ratio:.1%}) | "
            f"{n_tied_obs} observations in ties ({self.tie_ratio_:.1%})."
        )

        safe_methods = list(self.baseline_methods)

        if n_tied_times > 0 and "efron" in safe_methods:
            if self.tie_ratio_ > self.efron_obs_threshold:
                safe_methods = [m for m in safe_methods if m != "efron"]
                print(
                    f"[CoxHyperparameterTuner] WARNING: tied obs ratio "
                    f"{self.tie_ratio_:.1%} > threshold "
                    f"{self.efron_obs_threshold:.0%} — 'efron' removed."
                )
            else:
                print(
                    f"[CoxHyperparameterTuner] Tied obs ratio {self.tie_ratio_:.1%} "
                    f"below threshold — efron kept, watch for fold failures."
                )

        if not safe_methods:
            safe_methods = ["breslow"]
            print("[CoxHyperparameterTuner] Fallback: no valid methods, using breslow.")

        return safe_methods

    def _build_folds(self) -> list:
        """Build and return the CV fold split indices."""
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        return list(kf.split(self._df_fit))

    def _validate_spline(self) -> list:
        """
        Dry-run each knot count on the hardest CV fold (fewest events).
        Blacklists any knot count that causes a fit or prediction error,
        preventing MKL DGELSY errors from ever firing during the grid search.
        Returns the list of safe knot counts.
        """
        if "spline" not in self.safe_methods_:
            return self.n_baseline_knots_grid

        fold_train_dfs = [self._df_fit.iloc[ti] for ti, _ in self._kf_splits]
        hardest_fold   = min(fold_train_dfs, key=lambda d: int(d["E"].sum()))
        n_fold_events  = int(hardest_fold["E"].sum())

        print(
            f"[CoxHyperparameterTuner] Dry-run validating spline knots "
            f"{self.n_baseline_knots_grid} on hardest fold ({n_fold_events} events)..."
        )

        safe_knots   = []
        failed_knots = []

        for knots in sorted(self.n_baseline_knots_grid):
            try:
                with _suppress_fd_stderr(), warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    probe = CoxPHFitter(
                        baseline_estimation_method="spline",
                        n_baseline_knots=knots,
                        penalizer=self.spline_min_penalizer,
                        l1_ratio=0.0,
                    )
                    probe.fit(
                        hardest_fold,
                        duration_col="T",
                        event_col="E",
                        fit_options={"step_size": 0.95}
                    )
                    # Also verify prediction — spline uses a different internal
                    # class that may not expose all prediction methods
                    probe.predict_log_partial_hazard(
                        hardest_fold.drop(columns=["T", "E"])
                    )
                safe_knots.append(knots)
            except Exception:
                failed_knots.append(knots)

        if failed_knots:
            print(
                f"[CoxHyperparameterTuner] Dry-run BLACKLISTED knots {failed_knots} "
                f"— excluded from grid."
            )
        if safe_knots:
            print(
                f"[CoxHyperparameterTuner] Dry-run PASSED knots: {safe_knots}."
            )
        else:
            print(
                f"[CoxHyperparameterTuner] WARNING: all spline knots failed — "
                f"removing spline from search."
            )
            self.safe_methods_ = [m for m in self.safe_methods_ if m != "spline"]
            if not self.safe_methods_:
                self.safe_methods_ = ["breslow"]

        return safe_knots

    def _build_grid(self) -> list:
        """
        Build the full list of hyperparameter combos as
        (pen, l1, method, robust, step, extra_kwargs) tuples.

        Rules applied
        -------------
        - spline  : crossed with safe_knots_grid_, skips pen < spline_min_penalizer
        - breslow : full penalizer_grid, no extra_kwargs
        - efron   : full penalizer_grid, no extra_kwargs (if still in safe_methods_)
        """
        combos    = []
        base_axes = list(product(
            self.penalizer_grid, self.l1_ratios,
            self.robust_options, self.step_sizes
        ))

        for pen, l1, robust, step in base_axes:
            for method in self.safe_methods_:
                if method == "spline":
                    if pen < self.spline_min_penalizer:
                        continue
                    for knots in self.safe_knots_grid_:
                        combos.append((pen, l1, method, robust, step,
                                       {"n_baseline_knots": knots}))
                else:
                    combos.append((pen, l1, method, robust, step, {}))

        print(
            f"[CoxHyperparameterTuner] Grid: {len(combos)} combinations "
            f"× {self.n_splits} folds | workers={self.n_jobs}"
        )
        return combos

    def _run_search(self, combos: list) -> None:
        """
        Execute the parallel CV search over combos.
        Populates self.results_df_, self.best_params_, self.best_c_index_.
        """
        raw_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_evaluate_params)(
                self._df_fit, pen, l1, method, robust, step,
                self._kf_splits, extra_kwargs
            )
            for pen, l1, method, robust, step, extra_kwargs
            in tqdm(combos, desc="Cox CV tuning", unit="combo")
        )

        rows         = []
        best_c_index = -np.inf
        best_params  = None

        for (pen, l1, method, robust, step, extra_kwargs), \
                (mean_c, params, diag) in zip(combos, raw_results):
            rows.append({
                "penalizer":        pen,
                "l1_ratio":         l1,
                "baseline_method":  method,
                "robust":           robust,
                "step_size":        step,
                "n_baseline_knots": extra_kwargs.get("n_baseline_knots", "N/A"),
                "c_index_mean":     mean_c if mean_c is not None else np.nan,
                **diag,
            })
            if mean_c is not None and mean_c > best_c_index:
                best_c_index = mean_c
                best_params  = params

        if best_params is None:
            print("[CoxHyperparameterTuner] No valid params — falling back to defaults.")
            best_params = {
                "penalizer": 0.1, "l1_ratio": 0,
                "baseline_estimation_method": "breslow",
                "robust": True, "step_size": 0.95,
            }
            best_c_index = np.nan

        self.results_df_   = pd.DataFrame(rows).sort_values(
            "c_index_mean", ascending=False
        ).reset_index(drop=True)
        self.best_params_  = best_params
        self.best_c_index_ = best_c_index

        print(
            f"[CoxHyperparameterTuner] Best → {self.best_params_} | "
            f"C-index: {self.best_c_index_:.4f}"
        )

    def _save_report(self) -> None:
        """Build and save the 3-sheet Excel diagnostics report."""
        path = self.save_report_path
        if os.path.isdir(path) or path.endswith(("/", "\\")):
            path = os.path.join(path, "cox_tuning_report.xlsx")
        elif not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        _build_excel_report(self.results_df_, path)

    # ── convenience ──────────────────────────────────────────────────────────

    def _check_is_fitted(self):
        if self.best_params_ is None:
            raise RuntimeError("Call fit() before accessing results.")

    @property
    def summary(self) -> pd.DataFrame:
        """Top-10 results sorted by C-index."""
        self._check_is_fitted()
        return self.results_df_.head(10)


# ── Excel report (module-level, called by _save_report) ─────────────────────

def _health_flag(row) -> str:
    flags = []
    if row["n_overflow_warnings"] > 0:
        flags.append(f"overflow×{row['n_overflow_warnings']}")
    if row["n_nan_risk_scores"]   > 0:
        flags.append(f"NaN_scores×{row['n_nan_risk_scores']}")
    if row["n_inf_risk_scores"]   > 0:
        flags.append(f"inf_scores×{row['n_inf_risk_scores']}")
    if row["n_folds_failed"]      > 0:
        flags.append(f"fold_fail×{row['n_folds_failed']}")
    if row.get("fit_exception", ""):
        flags.append("exception")
    return "; ".join(flags) if flags else "OK"


def _build_excel_report(df: pd.DataFrame, save_path: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule

    df = df.copy()
    df["health_flag"] = df.apply(_health_flag, axis=1)

    wb = Workbook()

    thin_side   = Side(style="thin", color="CCCCCC")
    thin_border = Border(left=thin_side, right=thin_side,
                         top=thin_side,  bottom=thin_side)
    center      = Alignment(horizontal="center", vertical="center", wrap_text=True)
    hdr_fill    = PatternFill("solid", fgColor="1F4E79")
    hdr_font    = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    best_fill   = PatternFill("solid", fgColor="BDD7EE")
    alt_fill    = PatternFill("solid", fgColor="F5F5F5")
    ok_fill     = PatternFill("solid", fgColor="E2EFDA")
    warn_fill   = PatternFill("solid", fgColor="FFEB9C")
    bad_fill    = PatternFill("solid", fgColor="FFC7CE")
    sec_fill    = PatternFill("solid", fgColor="D6E4F0")

    # ── Sheet 1: All Results ───────────────────────────────────────────────
    ws = wb.active
    ws.title = "All Results"

    HEADERS = [
        "Rank", "penalizer", "l1_ratio", "baseline_method", "n_baseline_knots",
        "robust", "step_size", "c_index_mean", "c_index_std",
        "folds_ok", "folds_failed", "overflow_warnings",
        "nan_risk_scores", "inf_risk_scores", "fit_exception", "health_flag",
    ]
    COL_WIDTHS = [6, 12, 10, 18, 16, 8, 10, 14, 12, 10, 12, 18, 16, 16, 42, 32]

    for ci, h in enumerate(HEADERS, 1):
        cell = ws.cell(row=1, column=ci, value=h)
        cell.font = hdr_font; cell.fill = hdr_fill
        cell.alignment = center; cell.border = thin_border
    ws.row_dimensions[1].height = 30

    for ri, row in df.iterrows():
        er   = ri + 2
        rank = ri + 1
        flag = row["health_flag"]

        vals = [
            rank, row["penalizer"], row["l1_ratio"], row["baseline_method"],
            row.get("n_baseline_knots", "N/A"),
            str(row["robust"]), row["step_size"],
            round(row["c_index_mean"], 6) if pd.notna(row["c_index_mean"]) else "FAILED",
            round(row["c_index_std"],  6) if pd.notna(row["c_index_std"])  else "",
            row["n_folds_ok"], row["n_folds_failed"],
            row["n_overflow_warnings"], row["n_nan_risk_scores"],
            row["n_inf_risk_scores"], row["fit_exception"] or "", flag,
        ]

        for ci, val in enumerate(vals, 1):
            cell = ws.cell(row=er, column=ci, value=val)
            cell.font      = Font(name="Arial", size=9)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border    = thin_border
            cell.fill      = best_fill if rank == 1 else (
                alt_fill if ri % 2 == 0 else PatternFill()
            )

        fc = ws.cell(row=er, column=len(HEADERS))
        if flag == "OK":
            fc.fill = ok_fill;   fc.font = Font(name="Arial", size=9, color="375623")
        elif "exception" in flag or "fold_fail" in flag:
            fc.fill = bad_fill;  fc.font = Font(name="Arial", size=9, color="9C0006")
        else:
            fc.fill = warn_fill; fc.font = Font(name="Arial", size=9, color="7D4900")

    n_data = len(df) + 1
    cc = get_column_letter(8)
    ws.conditional_formatting.add(
        f"{cc}2:{cc}{n_data}",
        ColorScaleRule(
            start_type="min",  start_color="FFC7CE",
            mid_type="num",    mid_value=0.6, mid_color="FFEB9C",
            end_type="max",    end_color="C6EFCE",
        )
    )
    for i, w in enumerate(COL_WIDTHS, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    # ── Sheet 2: Best Result ───────────────────────────────────────────────
    ws2  = wb.create_sheet("Best Result")
    best = df.iloc[0]
    ws2.column_dimensions["A"].width = 28
    ws2.column_dimensions["B"].width = 28

    ws2.merge_cells("A1:B1")
    ws2["A1"] = "Cox PH Tuning — Best Result"
    ws2["A1"].font      = Font(name="Arial", bold=True, size=13, color="1F4E79")
    ws2["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws2.row_dimensions[1].height = 28

    summary = [
        ("Parameter",          "Value"),
        ("penalizer",          best["penalizer"]),
        ("l1_ratio",           best["l1_ratio"]),
        ("baseline_method",    best["baseline_method"]),
        ("n_baseline_knots",   best.get("n_baseline_knots", "N/A")),
        ("robust",             str(best["robust"])),
        ("step_size",          best["step_size"]),
        ("", ""),
        ("Metric",             "Value"),
        ("C-index (mean CV)",  round(best["c_index_mean"], 6)),
        ("C-index (std  CV)",  round(best["c_index_std"],  6) if pd.notna(best["c_index_std"]) else "N/A"),
        ("Folds OK",           best["n_folds_ok"]),
        ("Folds Failed",       best["n_folds_failed"]),
        ("", ""),
        ("Diagnostics",        "Value"),
        ("Overflow Warnings",  best["n_overflow_warnings"]),
        ("NaN Risk Scores",    best["n_nan_risk_scores"]),
        ("Inf Risk Scores",    best["n_inf_risk_scores"]),
        ("Health Flag",        best["health_flag"]),
        ("Fit Exception",      best["fit_exception"] or "None"),
    ]
    section_rows = {2, 10, 16}
    for ri, (lbl, val) in enumerate(summary, 2):
        a = ws2.cell(row=ri, column=1, value=lbl)
        b = ws2.cell(row=ri, column=2, value=val)
        if ri in section_rows:
            for c in (a, b):
                c.fill = sec_fill
                c.font = Font(name="Arial", bold=True, size=10, color="1F4E79")
        else:
            a.font = Font(name="Arial", bold=True, size=10)
            b.font = Font(name="Arial", size=10)
        for c in (a, b):
            c.alignment = Alignment(vertical="center")
            c.border    = thin_border

    # ── Sheet 3: Diagnostics Overview ─────────────────────────────────────
    ws3 = wb.create_sheet("Diagnostics Overview")
    ws3.column_dimensions["A"].width = 36
    ws3.column_dimensions["B"].width = 20

    ws3.merge_cells("A1:B1")
    ws3["A1"] = "Diagnostics Overview"
    ws3["A1"].font      = Font(name="Arial", bold=True, size=13, color="1F4E79")
    ws3["A1"].alignment = Alignment(horizontal="center")
    ws3.row_dimensions[1].height = 28

    total = len(df)
    diag_summary = [
        ("Metric",                               "Count"),
        ("Total combinations evaluated",         total),
        ("Combinations — no issues (OK)",        (df["health_flag"] == "OK").sum()),
        ("Combinations — overflow warnings",     (df["n_overflow_warnings"] > 0).sum()),
        ("Combinations — NaN risk scores",       (df["n_nan_risk_scores"]   > 0).sum()),
        ("Combinations — Inf risk scores",       (df["n_inf_risk_scores"]   > 0).sum()),
        ("Combinations — failed folds",          (df["n_folds_failed"]      > 0).sum()),
        ("Combinations — exceptions",            (df["fit_exception"].astype(str).str.len() > 0).sum()),
        ("", ""),
        ("Best  C-index (mean CV)",              round(df["c_index_mean"].max(),    6)),
        ("Worst C-index (mean CV)",              round(df["c_index_mean"].min(),    6)),
        ("Median C-index (mean CV)",             round(df["c_index_mean"].median(), 6)),
        ("Mean  C-index std across combos",      round(df["c_index_std"].mean(),    6)),
    ]
    for ri, (lbl, val) in enumerate(diag_summary, 2):
        a = ws3.cell(row=ri, column=1, value=lbl)
        b = ws3.cell(row=ri, column=2, value=val)
        if ri == 2:
            for c in (a, b):
                c.fill = sec_fill
                c.font = Font(name="Arial", bold=True, size=10, color="1F4E79")
        else:
            a.font = Font(name="Arial", bold=True, size=10)
            b.font = Font(name="Arial", size=10)
        for c in (a, b):
            c.alignment = Alignment(vertical="center")
            c.border    = thin_border

    wb.save(save_path)
    print(f"[CoxHyperparameterTuner] Report saved → {save_path}")