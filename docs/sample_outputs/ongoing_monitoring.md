# Ongoing Monitoring Plan — PD scorecard

> Run `20261019T055332Z_full_bd0bb86ae168` · generated 2026-10-19 · LLM mode: dry-run · portfolio project on public Home Credit data, not a production model. Every number in this document is checked by the grounding check (`grounding_report.md`).

## 1. Objectives

Monitoring must detect loss of discrimination, calibration drift, population shift and failure of model inputs, in particular the external scores, before they affect lending decisions or provisions.

## 2. Metrics and thresholds

Thresholds are the same as those used at validation, so monitoring results are directly comparable with the validated baseline (OOT Gini 0.5602).

| Metric | Amber | Red | Frequency |
|---|---|---|---|
| Score PSI (vs development) | 0.1 | 0.25 | monthly |
| Feature CSI (each selected feature) | 0.1 | 0.25 | monthly |
| Gini (minimum) | 0.45 + margin | 0.45 | quarterly |
| Grade binomial test p-value (under-prediction) | 0.05 | 0.01 | quarterly |
| Champion vs challenger Gini gap | 0.05 | n/a | quarterly |

## 3. Frequency and triggers

Population stability and characteristic stability are reviewed monthly. Discrimination and calibration are reviewed quarterly once outcomes mature. A full revalidation is required annually, or immediately on any red result, a change in the external score provider, or a material change in product or policy.

## 4. Escalation

Amber results are reported to the model owner with an action plan. Red results are escalated to the model risk committee; use of the champion is restricted or an overlay applied until the issue is resolved. If the champion is suspended, for example on a red stability result, a constraint failure or loss of the external scores, decisions fall back to the benchmark challenger, a 13-variable weight-of-evidence logistic regression scorecard, which is scored on every run.

## 5. Known weaknesses to watch

The validation findings below set the monitoring priorities.

| ID | Severity | Area | Finding |
|---|---|---|---|
| V01 | info | Benchmarking | Champion outperforms the benchmark |
| V02 | medium | Stability | Stability testing is not temporal (pseudo-OOT) |
| V03 | info | Stability | External population (application_test) score distribution stable |
| V04 | medium | Calibration | PD under-predicts default in some grades (OOT) |
| V05 | low | Calibration | Hosmer-Lemeshow rejects on OOT |
| V06 | medium | Discrimination | Rank-ordering breaks across PD deciles (OOT) |
| V07 | low | Master scale | Sparsely populated grades |
| V08 | low | Segments | Segment calibration off in income_type = State servant |
| V09 | low | Conceptual soundness | Champion uses undisclosed external scores |
| V10 | info | Explainability | Monotonic constraints verified |
| V11 | info | Explainability | LightGBM explanations are stable |
| V12 | medium | Conceptual soundness | Target is not the regulatory default definition |
