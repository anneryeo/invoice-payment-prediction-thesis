"""Phase 6 — Appendices (Tasks 6.1–6.4).

- 6.1: page break before every appendix heading (A–H)
- 6.4: intro + observation text for the DSO figures in Appendices A–C
- 6.2: new Appendix G — institutional authorization letter placeholder
- 6.3: new Appendix H — dataset description and 44-variable dictionary
"""
from docx_helpers import (load, save, find_par, par_exists, insert_par_after,
                          insert_par_before, insert_table_after,
                          page_break_before, BODY)
from docx.oxml.ns import qn

doc = load()
assert not par_exists(doc, "APPENDIX G:"), "phase 6 already applied"

# ---------------------------------------------------------------- Task 6.4
DSO_INTRO = (
    "Appendix {x} presents the Days Sales Outstanding (DSO) per payment plan "
    "type for calendar year {year}. DSO is computed as the average number of "
    "days between invoice due date and full settlement, segmented by the "
    "student's chosen installment plan (A through E, or none).")
DSO_OBS = (
    "Across the three years presented in Appendices A through C, the plan "
    "types that extend the longest credit windows — the extended monthly "
    "installment arrangements and invoices with no assigned plan — "
    "consistently exhibit the highest DSO, while Plan A (full upfront payment) "
    "settles fastest. The implication is that settlement speed tracks the "
    "length of the credit window a plan extends: the more payment is deferred, "
    "the longer receivables remain outstanding. This ordering is consistent "
    "with the ordinal plan-risk encoding used as a model feature (Section "
    "3.3), and it explains why plan-type variables carry predictive signal in "
    "the payment-bracket models.")

for letter, year in [("A", 2023), ("B", 2024), ("C", 2025)]:
    head = find_par(doc, f"APPENDIX {letter}: Days Sales Outstanding",
                    style="Heading 2")
    intro = insert_par_after(doc, head._p,
                             DSO_INTRO.format(x=letter, year=year), BODY)
    # find the image paragraph following the intro
    el = intro._p.getnext()
    while el is not None and not (el.tag == qn("w:p")
                                  and el.findall(".//" + qn("w:drawing"))):
        el = el.getnext()
    assert el is not None, f"no figure found in Appendix {letter}"
    obs = insert_par_after(doc, el, DSO_OBS if letter == "C" else (
        "The figure shows the same plan-level ordering observed across all "
        "three reported years: longer credit windows correspond to higher "
        "DSO. A consolidated observation across Appendices A–C follows the "
        "final figure in Appendix C."), BODY)

# ------------------------------------------------------- Tasks 6.2 and 6.3
refs_head = find_par(doc, "REFERENCES", style="Heading 1")

g_head = insert_par_before(doc, refs_head._p,
    "APPENDIX G: Institutional Communication and Authorization Letter",
    "Heading 2")
insert_par_before(doc, refs_head._p, (
    "Letter on file — co-author to provide. This appendix contains the written "
    "authorization from the partner institution permitting the research team "
    "to access and process student financial records for the purposes of this "
    "study, subject to the conditions described in Section 3.7.1. The letter "
    "is issued on the school's letterhead and signed by an authorized "
    "signatory, and it specifies the scope of data access granted, the "
    "restrictions on data use (research purposes only, mandatory "
    "pseudonymization, and no public release of raw records), and the date of "
    "authorization. This communication is distinct from any non-disclosure "
    "agreement: it constitutes the school's formal authorization to use its "
    "data for research."), BODY)

h_head = insert_par_before(doc, refs_head._p,
    "APPENDIX H: Dataset Description and Variable Dictionary", "Heading 2")
h_desc = insert_par_before(doc, refs_head._p, (
    "The pseudonymized dataset comprises 11,440 raw invoice records exported "
    "from the partner institution's ERP system, of which 6,527 labeled records "
    "were retained for modeling after filtering for sufficient payment "
    "history. Records span academic years 2019–2025, with payment activity "
    "observed through March 31, 2026, and originate from three source tables: "
    "revenues (itemized receivables and payments), enrollees (per-year "
    "enrollment and installment plan), and the chart of accounts (category "
    "classifications). The table below lists every variable used in the "
    "machine learning pipeline, its category, type, description, and source."), BODY)

ROWS = [
    ["Feature Name", "Category", "Type", "Description", "Source"],
    # Identity / index
    ["school_year", "Identity (not a model input)", "Categorical",
     "Academic year of the invoice", "Revenues"],
    ["student_id_pseudonymized", "Identity (not a model input)", "Integer",
     "Anonymized student identifier", "Revenues (pseudonymized)"],
    ["category_name", "Identity (not a model input)", "Categorical",
     "Fee category (e.g., tuition, miscellaneous)", "Chart of accounts"],
    # Raw financial
    ["gross_receivables", "Raw financial", "Numeric",
     "Total billed amount before deductions", "Revenues"],
    ["amount_discounted", "Raw financial", "Numeric",
     "Discount applied to the invoice", "Revenues"],
    ["adjustments", "Raw financial", "Numeric",
     "Post-billing adjustments (credit or debit)", "Revenues"],
    ["credit_sale_amount", "Raw financial", "Numeric",
     "Net receivable after discount and adjustments", "Derived"],
    ["due_date", "Raw financial", "Date",
     "Contractual payment due date", "Revenues"],
    ["date_fully_paid", "Raw financial", "Date",
     "Date the invoice was fully settled (empty if unpaid)", "Revenues"],
    # DTP
    ["dtp_1", "Days-to-payment history", "Numeric",
     "Days to full payment, most recent prior invoice", "Derived"],
    ["dtp_2", "Days-to-payment history", "Numeric",
     "Days to full payment, 2nd most recent prior invoice", "Derived"],
    ["dtp_3", "Days-to-payment history", "Numeric",
     "Days to full payment, 3rd most recent prior invoice", "Derived"],
    ["dtp_4", "Days-to-payment history", "Numeric",
     "Days to full payment, 4th most recent prior invoice", "Derived"],
    ["dtp_avg", "Days-to-payment history", "Numeric",
     "Simple average of available DTP values", "Derived"],
    ["dtp_wavg", "Days-to-payment history", "Numeric",
     "Weighted average of DTP values (recent invoices weighted higher)",
     "Derived"],
    ["dtp_2_trend", "Days-to-payment history", "Numeric",
     "Slope of DTP across the 2 most recent invoices", "Derived"],
    ["dtp_3_trend", "Days-to-payment history", "Numeric",
     "Slope of DTP across the 3 most recent invoices", "Derived"],
    ["days_since_last_payment", "Days-to-payment history", "Numeric",
     "Calendar days since the student's last full payment", "Derived"],
    ["dtp_rolling_std", "Days-to-payment history", "Numeric",
     "Rolling standard deviation of recent DTP values", "Derived"],
    ["dtp_max", "Days-to-payment history", "Numeric",
     "Maximum DTP observed across recent invoices", "Derived"],
    # Financial / cumulative
    ["amount_due_cumsum", "Financial / cumulative", "Numeric",
     "Cumulative total billed to the student to date", "Derived"],
    ["amount_paid_cumsum", "Financial / cumulative", "Numeric",
     "Cumulative total paid by the student to date", "Derived"],
    ["opening_balance", "Financial / cumulative", "Numeric",
     "Unpaid balance carried over from prior periods", "Derived"],
    ["opening_balance_flag", "Financial / cumulative", "Binary",
     "1 if the student has a non-zero opening balance", "Derived"],
    ["payment_ratio", "Financial / cumulative", "Numeric",
     "Ratio of cumulative paid to cumulative billed", "Derived"],
    # Behavioral
    ["prev_bracket", "Behavioral / historical", "Ordinal",
     "Payment bracket of the most recent prior invoice", "Derived"],
    ["early_payer_flag", "Behavioral / historical", "Binary",
     "1 if the student has a history of paying before the due date",
     "Derived"],
    ["on_time_streak", "Behavioral / historical", "Integer",
     "Consecutive on-time payments preceding the current invoice", "Derived"],
    # Payment plan
    ["plan_type_Plan - A", "Payment plan", "Binary (one-hot)",
     "Full payment upfront", "Enrollees"],
    ["plan_type_Plan - B", "Payment plan", "Binary (one-hot)",
     "Installment plan with up to 3 payments (1–30 days)", "Enrollees"],
    ["plan_type_Plan - C", "Payment plan", "Binary (one-hot)",
     "Installment plan with up to 6 payments (31–60 days)", "Enrollees"],
    ["plan_type_Plan - D", "Payment plan", "Binary (one-hot)",
     "Monthly installment plan (61+ days)", "Enrollees"],
    ["plan_type_Plan - E", "Payment plan", "Binary (one-hot)",
     "Special COVID-era plan (one school year only)", "Enrollees"],
    ["plan_type_nan", "Payment plan", "Binary (one-hot)",
     "No plan assigned; student not enrolled", "Enrollees"],
    ["plan_type_risk_score", "Payment plan", "Ordinal",
     "Plan type encoded by associated payment risk (A=0 … none=5)",
     "Derived"],
    # Temporal
    ["due_month", "Temporal", "Integer (1–12)",
     "Calendar month of the due date", "Derived"],
    ["due_quarter", "Temporal", "Integer (1–4)",
     "Fiscal quarter of the due date", "Derived"],
    # Survival
    ["partial_hazard", "Survival analysis", "Numeric",
     "Exponentiated linear predictor from the Cox PH model", "Cox PH model"],
    ["log_partial_hazard", "Survival analysis", "Numeric",
     "Natural log of the partial hazard", "Cox PH model"],
    ["expected_survival_time", "Survival analysis", "Numeric",
     "Expected days to full payment under the Cox model", "Cox PH model"],
    ["survival_probability", "Survival analysis", "Numeric",
     "Probability of remaining unpaid beyond a reference time point",
     "Cox PH model"],
    ["cumulative_hazard", "Survival analysis", "Numeric",
     "Cumulative hazard at the reference time point", "Cox PH model"],
    # Target
    ["dtp_bracket", "Target variable", "Ordinal (4 classes)",
     "Payment bracket label: on-time, 1–30, 31–60, or 61+ days late",
     "Derived"],
    ["censor", "Target variable", "Binary",
     "1 if the invoice was still unpaid at the observation cutoff",
     "Derived"],
]
tbl = insert_table_after(doc, h_desc._p, ROWS, style="Table Grid")
insert_par_after(doc, tbl._tbl, (
    "The raw data files underlying these variables are not publicly available "
    "due to the privacy constraints described in Section 3.7.1; access may be "
    "requested through institutional channels, subject to the authorization "
    "conditions documented in Appendix G."), BODY)

# ---------------------------------------------------------------- Task 6.1
for letter in "ABCDEFGH":
    head = find_par(doc, f"APPENDIX {letter}:", style="Heading 2")
    page_break_before(head)

save(doc)
print("Phase 6 applied.")
