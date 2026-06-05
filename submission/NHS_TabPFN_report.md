---
title: "Forecasting NHS Avoidable Deaths: a Day-of-Week Baseline with a TabPFN Residual Model"
output:
  pdf_document: default
  html_document: default
---

# Forecasting NHS Avoidable Deaths: a Baseline-Anchored TabPFN Residual Model

## 1. Problem and key insight

The target — daily estimated avoidable deaths — behaves as a **slowly-drifting level
plus largely-unpredictable noise** (development period: 898 days, mean 0.67, variance
0.123). After removing a rolling-mean level, the residual's autocorrelation at the first
*usable* lag (3 days, set by the reporting lag) is only ≈0.05 — essentially noise — with
one clear exception: a **strong day-of-week pattern** (lag-7 autocorrelation ≈0.30).

Two rules shape the design. The target has a **3-day reporting lag**: at forecast origin
D it is known only to D−3. The 220 explanatory variables, however, are **not lagged** —
they are available to day D. Howlett et al. (2026) show avoidable deaths are driven by
medical-patient ED boarding/crowding, so operational system-pressure metrics are the
*proximal causes* of the target and are fresher than its own lagged values.

## 2. Method

Each of the 10 horizons is forecast in three layers:

1. **Day-of-week-adjusted baseline.** A 45-day rolling mean of the lagged (≥3-day) target
   sets the level; an additive day-of-week deviation, estimated over all prior history,
   adds the weekly pattern. This baseline alone substantially beats the previous
   absolute-value model.
2. **TabPFN residual model.** For each horizon a TabPFN regressor predicts the *deviation
   from the baseline*. Modelling the residual (not the absolute value) lets TabPFN focus on
   the predictable exogenous signal — pressure-driven surges — instead of re-learning the
   level.
3. **Shrinkage blend:** `forecast = baseline + α · TabPFN_residual`, with α tuned per
   prize block. α may be 0, collapsing safely to the baseline.

The two prize blocks are configured **independently**, and the residual is fit per horizon,
so each block uses the feature set and α that suit it.

## 3. Feature selection (which signals, how many)

Features were chosen by **mutual information with the baseline residual** — the part of the
target the level misses. The strongest residual predictors are clinically sensible
system-pressure metrics, and they differ by horizon:

- **Days 1–5** — *acute, real-time* pressure: OPEL, available beds not utilised, Category-2
  ambulance response, 999 call-answering/stack pressure, A&E occupancy and Decisions-To-Admit.
- **Days 6–10** — *structural backlog*: P1 acute waiting list, DtA-waiting-for-capacity.

We tested feature-set size directly. A compact 10-feature set was **worse** than the
~30-feature curated set: TabPFN was not overwhelmed by ~30 well-chosen columns and used the
extra signal. PCA on the operational block was rejected because it blends individually
meaningful drivers into mixed components. Calendar (day-of-week, season, bank holidays) and
recent level/deviation features are included in both blocks.

## 4. What actually helps (robust validation)

Validation is competition-style rolling-origin (predict D+1..D+10 using target ≤ D−3),
evaluated over **484 origins (182 in winter)** to avoid small-sample noise. Early
small-sample runs over-stated the residual's value; the robust evaluation is the basis for
the final α.

| Model | MSE 1–5 (all) | MSE 1–5 (winter) | MSE 6–10 (all) | MSE 6–10 (winter) |
|---|---|---|---|---|
| Previous absolute-value TabPFN¹ | 0.104 | – | 0.120 | – |
| Day-of-week-adjusted baseline | 0.084 | 0.131 | 0.090 | 0.143 |
| **Final (baseline + TabPFN residual)** | **0.078** | **0.118** | **0.090** | **0.143** |

¹ Previous pipeline's own rolling validation (mixed season).

**Finding:** the TabPFN residual robustly improves **days 1–5** (winter MSE 0.131 → 0.118,
≈10%; α=0.8), because fresh acute-pressure signals genuinely anticipate near-term surges. For
**days 6–10** the operational signals are too stale to beat the smooth level, so that block
uses the baseline (α=0). The largest, most reliable gain over the old pipeline comes from
the day-of-week baseline; TabPFN adds a further edge on the short-horizon prize.

## 5. Implementation and compliance

- **Engine:** local `tabpfn` (CPU) by default, so results reproduce with no API key or
  internet (`--api` switches to the cloud client). One 10-day forecast is ≤5 TabPFN fits on
  ~365 rows — well under the 1-hour limit.
- **Recalibration:** the residual models retrain on each new forecast origin using only data
  available then — permitted by the rules and needing no code change.
- **No external data; only supplied variables; fixed seeds.** Config in `data/model_config.json`.

## 6. Files
`NHS_TabPFN_data_prep.R` → `data/processed_tabpfn.csv`; `nhs_forecast_core.py` (baseline +
residual primitives); `NHS_TabPFN_pipeline.py` (`--mode validate|assess`); forecasts in
`submission/`.
