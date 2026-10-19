# Ongoing Monitoring Plan — PD scorecard

> Run `20261019T041619Z_full_21f3ba7a1ecd` · generated 2026-10-19 · LLM mode: dry-run · portfolio project on public Home Credit data, not a production model. Every number in this document is checked by the grounding check (`grounding_report.md`).

## 1. Objectives

Monitoring must detect loss of discrimination, calibration drift, population shift and failure of model inputs, in particular the external scores, before they affect lending decisions or provisions.

## 2. Metrics and thresholds

Thresholds are the same as those used at validation, so monitoring results are directly comparable with the validated baseline (OOT Gini 0.5008).

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

Amber results are reported to the model owner with an action plan. Red results are escalated to the model risk committee; use of the model is restricted or an overlay applied until the issue is resolved.

## 5. Known weaknesses to watch

The validation findings below set the monitoring priorities.

| ID | Severity | Area | Finding |
|---|---|---|---|
| V01 | medium | Performance | Champion test Gini close to gate |
| V02 | medium | Benchmarking | Challenger overfits the training data |
| V03 | medium | Benchmarking | Challenger materially outperforms champion |
| V04 | medium | Stability | Stability testing is not temporal (pseudo-OOT) |
| V05 | info | Stability | External population (application_test) score distribution stable |
| V06 | low | Calibration | Amber calibration in some grades (OOT) |
| V07 | low | Master scale | Sparsely populated grades |
| V08 | low | Segments | Segment calibration off in contract_type = Revolving loans |
| V09 | low | Segments | Segment calibration off in income_type = Commercial associate |
| V10 | low | Segments | Segment calibration off in income_type = Pensioner |
| V11 | low | Segments | Segment calibration off in income_type = State servant |
| V12 | low | Segments | Segment calibration off in age_band = <30 |
| V13 | medium | Conceptual soundness | Heavy reliance on undisclosed external scores |
| V14 | medium | Conceptual soundness | Target is not the regulatory default definition |
