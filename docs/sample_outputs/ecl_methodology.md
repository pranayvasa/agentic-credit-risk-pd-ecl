# ECL Methodology Note — illustrative RBI ECL

> Run `20261019T041619Z_full_21f3ba7a1ecd` · generated 2026-10-19 · LLM mode: dry-run · portfolio project on public Home Credit data, not a production model. Every number in this document is checked by the grounding check (`grounding_report.md`).

## 1. Scope and status

This note describes an illustrative ECL simulation under the RBI Directions RBI/DOR/2026-27/398. Home Credit is application-level data, so the OOT holdout (76,878 loans, exposure 46,003.9 million) is treated as a book at a simulated reporting date with every loan at origination. Results do not represent any institution's provisions.

| Assumption | Value |
|---|---|
| LGD (unsecured backstop, para 97) | 70% |
| 12-month PD floor (para 96) | 0.03% |
| Floor Stage 1 / 2 / 3 (para 82(3)(x)) | 1% / 5% / 25% |
| Median term (months) | 33 |
| Lifetime method (primary) | constant_hazard |
| Implied monthly rate, Cash loans | 3.33% |
| Implied monthly rate, Consumer loans | 3.34% |

## 2. Staging

Stage 3 applies to borrowers more than 90 days past due on prior Home Credit loans in the last 12 months. Stage 2 applies on the rebuttable presumption of more than 30 days past due (Home Credit or bureau), or on relative deterioration against an application-only origination proxy: a downgrade of at least 3 grades, or a PD at least 2 times the proxy and above 5%. Realised default rates rise from 7.90% in Stage 1 to 12.99% in Stage 2 and 23.94% in Stage 3.

| Stage | Accounts | Realised default rate |
|---|---|---|
| Stage 1 | 74,397 | 7.90% |
| Stage 2 | 2,410 | 12.99% |
| Stage 3 | 71 | 23.94% |

| Trigger | Accounts |
|---|---|
| hc_90dpd | 71 |
| hc_30dpd | 383 |
| bureau_30dpd | 1,011 |
| grade_downgrade | 310 |
| pd_ratio | 1,167 |

## 3. 12-month and lifetime PD

The 12-month PD is the champion PD, floored at 0.03%. Lifetime PD uses a constant monthly hazard. An alternative scales a monthly DPD transition chain estimated from bureau history to the same 12-month PD; it changes total ECL from 2,041.0 million to 1,924.6 million.

| Method | Weighted ECL | Stage 2 ECL | Stage 2 mean lifetime PD |
|---|---|---|---|
| constant_hazard | 2,041.0 million | 201.6 million | 37.90% |
| markov | 1,924.6 million | 216.8 million | 43.06% |

## 4. EAD, EIR and LGD

Term and rate cannot both be derived from credit and annuity amounts, so the monthly rate per contract type is the rate implied by approved prior Home Credit loans, and each loan's term is solved from the annuity formula. Exposure follows the amortising balance and cash flows are discounted at that rate. Revolving loans use a 24-month life. LGD is the 70% backstop for unsecured exposures.

| LGD | Model ECL | Final ECL | Coverage |
|---|---|---|---|
| 45% | 1,312.1 million | 1,358.4 million | 2.95% |
| 55% | 1,603.7 million | 1,635.0 million | 3.55% |
| 70% | 2,041.0 million | 2,059.5 million | 4.48% |
| 80% | 2,332.6 million | 2,345.9 million | 5.10% |

## 5. Forward-looking scenarios

Three illustrative scenarios scale the 12-month PD; staging is held constant. The probability-weighted model ECL is 2,041.0 million.

| Scenario | Weight | PD scalar |
|---|---|---|
| base | 0.60 | 1.00 |
| upside | 0.15 | 0.85 |
| downside | 0.25 | 1.35 |

| Scenario column | Total ECL |
|---|---|
| base (w=0.60, PD x1.00) | 1,919.5 million |
| upside (w=0.15, PD x0.85) | 1,638.6 million |
| downside (w=0.25, PD x1.35) | 2,574.2 million |
| probability_weighted | 2,041.0 million |

## 6. Results

Final ECL after RBI minimum provisions is 2,059.5 million, a coverage of 4.48% of exposure. Coverage is 4.10%, 14.32% and 70.00% for Stages 1, 2 and 3. Floors add 18.4 million.

| Stage | Accounts | Exposure | Model ECL | Final ECL | Coverage | Floor binds |
|---|---|---|---|---|---|---|
| stage_1 | 74,397 | 44,526.4 million | 1,813.6 million | 1,827.3 million | 4.10% | 11.1% |
| stage_2 | 2,410 | 1,440.7 million | 201.6 million | 206.4 million | 14.32% | 22.4% |
| stage_3 | 71 | 36.8 million | 25.8 million | 25.8 million | 70.00% | 0.0% |
| total | 76,878 | 46,003.9 million | 2,041.0 million | 2,059.5 million | 4.48% | 11.4% |

## 7. Limitations

The book is simulated at origination; staging relies on prior-loan delinquency and an origination proxy; LGD is a regulatory backstop rather than an estimate; scenarios are illustrative; and rates are segment-level, not loan-level. The figures demonstrate the method and must not be read as a provision for any institution.
