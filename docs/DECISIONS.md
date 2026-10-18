# Decisions log

Decisions agreed with the author that refine or deviate from `docs/PROJECT_BRIEF.md`. Where this file and the brief conflict, this file wins.

Regulatory source for ECL parameters: Reserve Bank of India (Commercial Banks – Asset Classification, Provisioning and Income Recognition) Directions, 2026, RBI/DOR/2026-27/398 (DOR.STR.REC.No.6/21.06.011/2026-27), dated 27 April 2026, effective 1 April 2027. Primary text: https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx?id=13400 (checked 2026-10-18).

| ID | Date | Decision | Why |
|---|---|---|---|
| D1 | 2026-10-18 | The base-case LGD is **70%**, the para 97 unsecured backstop. Sensitivity is run at 45 / 55 / 70 / 80%, and 45% is labelled a Basel F-IRB-style comparison. | The brief suggested 45%. RBI requires the backstop where internal LGD estimates are insufficient, and we have no recovery data. |
| D2 | 2026-10-18 | Unsecured retail floors (para 82(3)(x)): Stage 1 1%, Stage 2 5%, Stage 3 25% (0–1 yr) then 100%. The 12-month PD floor is 0.03% (para 96). ECL = max(model ECL, floor × EAD), and model and floored ECL are reported separately. | Verified against the primary text. |
| D3 | 2026-10-18 | TARGET ("payment difficulties") is kept as the modelling target, and the PD is labelled "12-month-equivalent" for ECL. | The gap from the 90+ DPD / NPA default definition is documented as a key limitation. |
| D4 | 2026-10-18 | The Markov lifetime PD uses an empirical **DPD-bucket transition matrix from `bureau_balance` monthly STATUS**, not grade-to-grade transitions. | The current loan has no time dimension, so grade migrations cannot be estimated. |
| D5 | 2026-10-18 | Stage 3 = 90+ DPD on recent Home Credit prior loans (borrower level). Stage 2 = 30+ DPD presumption (para 33) on recent Home Credit or bureau history, or the relative SICR test (D6). The OOT is a simulated reporting-date book. | Delinquency data describes prior loans, not the current one. |
| D6 | 2026-10-18 | The origination-PD proxy is an application-leg-only scorecard. SICR = grade downgrade ≥ N notches, or PD ratio ≥ X together with an absolute PD floor (config). | Para 26 needs a comparison with initial recognition, and no origination PD exists. This is disclosed as an approximation. |
| D7 | 2026-10-18 | PSI is reported honestly (≈0 by construction on the pseudo-OOT), with a stress-PSI demonstration added. | A random split cannot show drift. |
| D8 | 2026-10-18 | Test is used for feature-stability selection. OOT is the only fully clean holdout, and gates check both test and OOT. | This follows the brief's design, with the consequence stated. |
| D9 | 2026-10-18 | "Agent" means a graph node with a declared contract. Only `critic` and `document` call an LLM, and the README says so. | Honesty about what is and is not AI-driven. |
| D10 | 2026-10-18 | Documents map to SR 11-7 and ECL Directions Ch. V (final), plus the RBI draft MRM guidance (24 Jun 2026; comments closed 24 Jul 2026), labelled as a draft. | The MRM guidance is not final. |
| D11 | 2026-10-18 | Gemini test model `gemini-3.5-flash-lite`, final model `gemini-3.6-flash` (both stable). Rates are in `config/llm.yaml`; 3.6 Flash pricing doubles from 1 Jan 2027. | The author already uses these models elsewhere. The brief suggested a Pro-tier final model; the author chose Flash. |
| D12 | 2026-10-18 | `--sample` runs use warn-only gates with lower thresholds. Full runs enforce the brief's thresholds. | Sample metrics are noisier. |
| D13 | 2026-10-18 | Critic bound: at most **2 rounds** of extra analyses, so at most 3 critic reviews. Requests after the bound are dropped and logged. | This is our reading of "at most 2 iterations". |
| D14 | 2026-10-18 | Graph state holds artefact paths and summaries, never DataFrames. Each node gets a read-only view of only the keys it declared. | This makes the auditability, read-only-validator and summaries-only-to-LLM rules enforceable in code. |
