# Round 2 Layout Fix Report

**Date:** 2026-06-28  
**Scope:** UI/UX layout corrections for three Level-2 sub-DFD drawio files  
**Branch:** beley

---

## Motivation

The five Level-2 sub-DFD files generated in round1 had overlapping text, edge labels running through process boxes, and cramped horizontal spacing that made the diagrams hard to read in draw.io. The three most affected files were rebuilt with corrected coordinates.

---

## Files Regenerated

| File | Status |
|------|--------|
| `Level-2 DFD - 7.5 cox survival analysis tuning.drawio` | Rebuilt |
| `Level-2 DFD - 8.0 model building.drawio` | Rebuilt |
| `Level-2 DFD - 8.5 survival feature generation.drawio` | Rebuilt |
| `Level-2 DFD - 1.0 data importation.drawio` | Unchanged (clean) |
| `Level-2 DFD - 5.0 data cleaning.drawio` | Unchanged (acceptable) |

---

## Changes Per Diagram

### 7.5 Cox Survival Analysis Tuning

| Issue | Fix |
|-------|-----|
| `Fit CoxnetSurvivalAnalysis` clipped in 155 px box | 7.5.3 widened to 200 px |
| `Harrell concordance` clipped in 130 px box | 7.5.4 widened to 185 px |
| Edge labels overlapping process boxes | All internal edge labels offset -12 px above edge line |
| Output labels "Best (alpha, l1_ratio)" and "9 optimal time points" merged | Separate exit styles: `exitY=1` with staggered entry at D4 (`entryY=0.3` / `0.7`) |
| Canvas too small | Enlarged to 1400×900 |

### 8.0 Model Building

| Issue | Fix |
|-------|-----|
| "Fitted base/ordinal/two-stage models" labels stacked on left border | Changed from 3-column horizontal to 3-row vertical layout |
| Insufficient gap between D4 and container for 3 label lines | Each process now at a distinct y-band; exit edges use `exitY=0.5` with staggered `entryY=0.2/0.5/0.8` at D4 |
| Process boxes too narrow for content | Each process now spans full container width (750 px) |
| Canvas too small | Enlarged to 1400×900 |

### 8.5 Survival Feature Generation

| Issue | Fix |
|-------|-----|
| "Scalar features" edge from 8.5.4 routed through 8.5.3 bounding box | Edge now exits bottom of 8.5.4 (`exitY=1`) and enters right of 8.5.5 (`entryX=1`), routing around 8.5.3 |
| S(t)/H(t) edge labels overlapping box borders | Labels offset -12 px above edge line |
| `Fit CoxnetSurvivalAnalysis` clipped in 170 px box | 8.5.1 widened to 215 px |
| `E[T]` lines cramped | 8.5.4 widened to 180 px |
| Canvas too small | Enlarged to 1400×900 |

---

## Coordinate Summary

### 7.5 (container: x=210, y=260, w=950, h=230)
| Process | x | y | w | h |
|---------|---|---|---|---|
| 7.5.1 | 20 | 40 | 160 | 80 |
| 7.5.2 | 205 | 40 | 145 | 80 |
| 7.5.3 | 375 | 40 | **200** | 80 |
| 7.5.4 | 600 | 40 | **185** | 80 |
| 7.5.5 | 205 | 160 | 155 | 60 |
| 7.5.6 | 385 | 160 | 155 | 60 |

### 8.0 (container: x=210, y=260, w=800, h=390)
| Process | x | y | w | h |
|---------|---|---|---|---|
| 8.1 | 20 | 30 | 750 | 90 |
| 8.2 | 20 | 150 | 750 | 90 |
| 8.3 | 20 | 270 | 750 | 90 |

### 8.5 (container: x=210, y=260, w=990, h=250)
| Process | x | y | w | h |
|---------|---|---|---|---|
| 8.5.1 | 20 | 40 | **215** | 75 |
| 8.5.2 | 260 | 40 | 165 | 75 |
| 8.5.3 | 450 | 40 | 165 | 75 |
| 8.5.4 | 640 | 40 | **180** | 75 |
| 8.5.5 | 260 | 155 | 230 | 60 |

---

## Tests

New test file: `round2_layout_fixes/tests/test_layout_fixes.py`

Tests verify:
- Canvas is 1400×900 (not the old 1169×827)
- All processes present with correct IDs
- Key process boxes are wide enough to accommodate long labels
- Edge label y-offsets are non-zero (labels raised off edge lines)
- 8.0 processes are vertically stacked (y-separation > 80 px)
- 8.5 "Scalar features" edge uses `exitY=1` (bottom-exit routing)

Run: `python -m pytest round2_layout_fixes/tests/ -v`
