# Sensitivity: models without the external scores (EXT_SOURCE)

**Question (validation finding V13).** EXT_SOURCE_1/2/3 are described by Home Credit only as a
"normalized score from external data source". The provider and construction are undisclosed,
yet these scores carry about 52% of the champion scorecard's points range. How much does
performance depend on them?

**Method.** Two full-population runs with identical code, data, split and seed (42). The only
difference is `features.exclude_features`. The variant removes APP_EXT_SOURCE_1,
APP_EXT_SOURCE_2, APP_EXT_SOURCE_3, APP_EXT_SOURCE_MEAN, APP_EXT_SOURCE_MIN and
APP_EXT_SOURCE_MISSING_COUNT from both the scorecard and the challenger. Selection, modelling
and validation then rerun unchanged: train-only fitting, the same thresholds, the same gates.

| Run | run_id | Command |
|---|---|---|
| Baseline | `20261019T034102Z_full_e20fc11ff85c` | `acr run` |
| No EXT_SOURCE | `20261019T034344Z_full_5544798d5000` | `acr run --exclude-features APP_EXT_SOURCE_1,…,APP_EXT_SOURCE_MISSING_COUNT` |

## Results

| Model | Run | Train Gini (AUC) | Test Gini (AUC) | OOT Gini (AUC) | OOT ECE / slope |
|---|---|---|---|---|---|
| Champion scorecard | Baseline (13 variables) | 0.5041 (0.7521) | 0.4896 (0.7448) | 0.5008 (0.7504) | 0.0023 / 1.003 |
| | **No EXT_SOURCE** (15 variables) | 0.4008 (0.7004) | **0.3882 (0.6941)** | **0.3961 (0.6980)** | 0.0018 / 0.987 |
| LightGBM challenger | Baseline | 0.6791 (0.8396) | 0.5704 (0.7852) | 0.5761 (0.7881) | 0.0026 / 1.004 |
| | **No EXT_SOURCE** | 0.6417 (0.8208) | 0.5258 (0.7629) | **0.5268 (0.7634)** | 0.0021 / 1.017 |

| Change on OOT | Champion | Challenger |
|---|---|---|
| Gini | −0.1047 (−20.9%) | −0.0493 (−8.6%) |
| AUC | −0.0524 | −0.0247 |
| Challenger minus champion OOT Gini | 0.0753 → **0.1307** (95% CI 0.1205 to 0.1422) | |

**Gate outcome.** Without the external scores the champion falls below the 0.45 minimum on
both test (0.3882) and OOT (0.3961). The validator raises two red findings and the run
**halts** before ECL, which is exactly what the gate is for.

**What the scorecard selects instead.** The cap of 15 now binds: there are more, weaker
features. All nine non-EXT features from the baseline are kept, and six are added:
APP_AGE_YEARS, BUR_MONTHS_SINCE_FIRST, BEH_INS_LATE_SHARE, BUR_ACTIVE_DEBT_TO_CREDIT,
BEH_CC_UTIL_MEAN and PRV_MONTHS_SINCE_REFUSAL. The strongest replacement has IV 0.118,
against 0.467 for EXT_SOURCE_MIN in the baseline.

**Other effects.**
* Calibration is unaffected (OOF ECE about 0.002 for both runs). The loss is purely in
  discrimination.
* New medium findings in the variant: weak discrimination (OOT Gini below 0.30) for revolving
  loans, pensioners and applicants aged 60+, and segment calibration for men moving outside
  tolerance (mean PD / default rate 0.886). The external scores appear to carry information
  that otherwise has to be proxied through age and income type.

## Interpretation

1. **The dependency is material, not cosmetic.** The external scores account for about a
   fifth of the scorecard's discriminatory power. Without them the champion is not fit for
   purpose under this project's own gate.
2. **The non-linear challenger is more resilient** (−8.6% vs −20.9%). It recovers part of the
   lost signal through interactions among the remaining features (organisation type,
   credit-to-annuity, employment). The cost of interpretability nearly doubles: the Gini gap
   grows from 0.075 to 0.131.
3. **Model-risk view.** Relying on undisclosed third-party scores is a third-party / vendor
   model risk. SR 11-7 expects vendor models to be understood and validated by the user. The
   RBI draft model risk management guidance (June 2026, not yet final) states that
   outsourcing does not transfer accountability. Both point to the same remediation.

## Remediation options (for the validation report)

* Obtain documentation of the external scores (provider, inputs, development data,
  performance, monitoring), and validate them as vendor model inputs.
* Monitor EXT_SOURCE availability and distribution as a key risk indicator. EXT_SOURCE_1 is
  already the least stable selected feature on the external population (CSI 0.085 against
  application_test).
* Keep a documented fallback: if the external scores become unavailable, the
  no-EXT_SOURCE scorecard (OOT Gini about 0.40) is the realistic floor, below the current
  gate. A business decision would be needed before relying on it.

This sensitivity does **not** change the production configuration: `exclude_features`
defaults to empty, so the champion keeps the external scores.
