# Model Development Document — PD scorecard

> Run `20261019T052034Z_full_975e6130602a` · generated 2026-10-19 · LLM mode: dry-run · portfolio project on public Home Credit data, not a production model. Every number in this document is checked by the grounding check (`grounding_report.md`).

## 1. Purpose and intended use

The model estimates a 12-month-equivalent probability of default (PD) for unsecured retail loan applicants, using the public Home Credit Default Risk data. It supports application decisioning and supplies the PD input to an illustrative RBI ECL calculation. The champion is a 13-variable weight-of-evidence (WoE) logistic regression scorecard; the challenger is a LightGBM gradient-boosting model on 112 candidate features. This is a portfolio demonstration on public data, not a production model.

## 2. Data

The development population is 307,511 applications with a default rate of 8.07%. TARGET flags payment difficulties on early instalments; it is not the regulatory 90+ DPD definition. Applicants were split on customer ID before any data-learned step: 75:25 into development and out-of-time (OOT), then development 75:25 into train and test, stratified on TARGET with seed 42. The OOT holdout is a random (pseudo-OOT) holdout because the data carries no application dates, so it cannot evidence temporal stability. A further 48,744 unlabelled applicants (application_test) are scored only for a distributional stability check.

| Table | Rows |
|---|---|
| application | 307,511 |
| application_test | 48,744 |
| bureau | 1,716,428 |
| bureau_balance | 27,299,925 |
| installments | 13,605,401 |
| pos_cash | 10,001,358 |
| credit_card | 3,840,312 |
| previous | 1,670,214 |

| Partition | Applicants | Share | Default rate |
|---|---|---|---|
| train | 172,974 | 56.25% | 8.07% |
| test | 57,659 | 18.75% | 8.07% |
| oot | 76,878 | 25.00% | 8.07% |

## 3. Methodology

Features come from four legs aggregated to one row per customer: application, bureau, behavioural (prior Home Credit repayments) and previous applications. Each continuous or categorical feature is grouped into monotonic optimal bins fitted on train only, and replaced by its weight of evidence. The champion is a logistic regression on WoE features with points scaled at 20 points to double the odds and a base score of 600 at 50:1 odds. The challenger is LightGBM with depth 5, 15 leaves and at least 1,000 observations per leaf, fixed in configuration after offline train-only cross-validation. Calibration was assessed on out-of-fold train predictions: not needed: OOF ECE 0.0028 (max 0.005), slope 0.998 (tolerance 0.1). PDs map to a 10-grade master scale. An application-only scorecard serves as an origination-PD proxy for ECL staging. No language model fits, tunes or alters any model.

## 4. Feature selection

Of 114 engineered features, 112 were candidates (gender and marital status are protected characteristics kept for fairness analysis only). Screens on missingness, binning, information value (minimum 0.02, leakage flag above 0.5) and train-vs-test stability left 56. Correlation (above 0.7), VIF (above 5), a cap of 15 variables and backward elimination on coefficient sign and p-value (above 0.05) gave the final 13 variables.

| Stage | Features excluded |
|---|---|
| role | 2 |
| binning | 13 |
| iv | 43 |
| corr | 10 |
| cap | 31 |
| model | 2 |

| Feature | Leg | IV |
|---|---|---|
| APP_EXT_SOURCE_MIN | application | 0.4672 |
| APP_EXT_SOURCE_3 | application | 0.3350 |
| APP_EXT_SOURCE_2 | application | 0.3198 |
| APP_EXT_SOURCE_1 | application | 0.1494 |
| APP_EMPLOYED_YEARS | application | 0.1184 |
| BUR_ACTIVE_SHARE | bureau | 0.0909 |
| BUR_MONTHS_SINCE_LAST | bureau | 0.0879 |
| BEH_INS_DPD_MAX_12M | behavioural | 0.0795 |
| APP_OCCUPATION_TYPE | application | 0.0788 |
| APP_ORGANIZATION_TYPE | application | 0.0730 |
| APP_CREDIT_TO_GOODS | application | 0.0724 |
| BEH_CC_UTIL_MEAN_3M | behavioural | 0.0719 |
| PRV_REFUSED_SHARE | previous | 0.0677 |

## 5. Performance

The champion achieves OOT Gini 0.5008 (AUC 0.7504) against 0.5041 on train, so there is no material overfitting. OOT calibration is close: decile ECE 0.0023 and calibration slope 1.003. The challenger reaches OOT Gini 0.5602 (AUC 0.7801), 0.0594 higher than the champion.

| Model | Partition | Gini | AUC | KS | Mean PD | Default rate |
|---|---|---|---|---|---|---|
| champion | train | 0.5041 | 0.7521 | 0.3782 | 8.07% | 8.07% |
| champion | test | 0.4896 | 0.7448 | 0.3676 | 8.06% | 8.07% |
| champion | oot | 0.5008 | 0.7504 | 0.3769 | 8.04% | 8.07% |
| challenger | train | 0.6120 | 0.8060 | 0.4623 | 8.07% | 8.07% |
| challenger | test | 0.5520 | 0.7760 | 0.4115 | 8.07% | 8.07% |
| challenger | oot | 0.5602 | 0.7801 | 0.4263 | 8.08% | 8.07% |

| Model (OOT) | Decile ECE | Slope | HL p-value | Mean PD / DR |
|---|---|---|---|---|
| champion | 0.0023 | 1.003 | 0.3162 | 0.996 |
| challenger | 0.0035 | 0.957 | 0.0059 | 1.001 |

## 6. Explainability of the challenger

The challenger is constrained so that 53 features can only raise PD as they rise and 38 can only lower it, following their train-only WoE trends; 21 categorical or flat features are unconstrained. Every OOT prediction is decomposed with exact TreeSHAP (76,878 applicants; attributions reconcile to the model score within 0.0e+00 in log-odds), and each applicant receives the 4 factors that raised their PD most. A sweep test found 0 violations of the constraints. The top drivers are stable: test and OOT share 100% of the top 10. External scores carry 22% of mean absolute SHAP. SHAP describes attribution, not causation, and correlated features can share credit, so importance is also shown by leg.

| Feature | Leg | Share of mean abs SHAP | Constraint |
|---|---|---|---|
| APP_EXT_SOURCE_MEAN | application | 11.0% | PD can only fall with it |
| APP_ANNUITY_TO_INCOME | application | 3.7% | PD can only rise with it |
| APP_CREDIT_TO_GOODS | application | 3.5% | PD can only rise with it |
| BEH_INS_COUNT | behavioural | 3.4% | PD can only fall with it |
| APP_ORGANIZATION_TYPE | application | 3.2% | unconstrained |
| APP_EXT_SOURCE_MIN | application | 3.1% | PD can only fall with it |
| APP_OCCUPATION_TYPE | application | 2.8% | unconstrained |
| BEH_POS_REMAINING_INSTALMENTS | behavioural | 2.7% | PD can only rise with it |
| APP_EXT_SOURCE_3 | application | 2.7% | PD can only fall with it |
| PRV_CNT_PAYMENT_MEAN | previous | 2.7% | PD can only rise with it |

| Leg | Share of mean abs SHAP |
|---|---|
| application | 53.7% |
| behavioural | 20.9% |
| previous | 13.6% |
| bureau | 11.8% |

| Check | Result |
|---|---|
| Applicants explained (OOT) | 76,878 |
| SHAP additivity, max error (log-odds) | 0.0e+00 |
| Monotonic constraints: features tested / violating rows | 91 / 0 |
| Top-10 driver overlap / rank correlation, oot half vs half | 100% / 1.000 |
| Top-10 driver overlap / rank correlation, test vs oot | 100% / 1.000 |
| Top-10 driver overlap / rank correlation, train vs oot | 100% / 1.000 |
| Challenger mean PD / default rate, gender F | 1.064 |
| Challenger mean PD / default rate, gender M | 0.918 |

## 7. Assumptions and limitations

Key limitations: (1) the OOT holdout is random, so temporal stability is untested; (2) the target is payment difficulty, not 90+ DPD; (3) undisclosed external scores carry 52% of the scorecard's points range, and without them OOT Gini falls below the gate (see the external-score sensitivity); (4) the data has no origination-date PD, so ECL staging uses a proxy; (5) the ECL calculation is an illustrative simulation on a pseudo-book, not a portfolio provision.

## 8. Governance and reproducibility

Run 20261019T052034Z_full_975e6130602a used seed 42 and configuration hash 975e6130602a; the same inputs reproduce the same numbers. Every fitting step is guarded to accept train rows only. Each graph node logs its inputs, outputs, gate decisions and timings to an audit trail. The independent validator works on saved outputs only and verifies model artefact checksums. LLM agents critique and document; they cannot change model parameters.
