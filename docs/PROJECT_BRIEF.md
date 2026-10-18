# Project Brief — Agentic Credit Risk: PD, ECL and Model-Risk Documentation

> Source of truth for scope and requirements. Saved verbatim from the author's original brief (2026-10-18).
> Decisions and deviations agreed later are recorded in `docs/DECISIONS.md`, not by editing this file.

I want to build a GitHub portfolio project with you: an agent-orchestrated credit risk pipeline that develops a Probability of Default (PD) model on the public Home Credit Default Risk dataset, extends it to RBI ECL staging and lifetime PD, and auto-generates model-risk documentation. Before writing any code, save this entire brief as docs/PROJECT_BRIEF.md and create a CLAUDE.md that points to it, so it persists across sessions. Then enter plan mode, propose the implementation plan, and wait for my approval.

## About me and the purpose
- I'm a Chartered Accountant with 7 years in BFSI credit risk and analytics. At a previous bank I built a multi-leg PD model (application, behavioural, consumer bureau, commercial bureau, transaction data).
- This repo is a portfolio piece for credit risk, model risk and credit decisioning roles. The audience is hiring managers and model validators, so the repo must read as regulatory-grade, not as a Kaggle notebook.
- Hard constraint: public data only. No proprietary or employer data, names or code anywhere.
- My GitHub username is pranayvasa. Suggested repo name: agentic-credit-risk-pd-ecl (suggest better names if you have them).

## Core design principle (non-negotiable)
LLM agents never fit, tune or alter the model. The modelling core is deterministic and reproducible: fixed seeds, versioned config, same inputs produce the same numbers. LLM agents sit around that core to critique and document, mirroring first-line development versus second-line independent validation. Make this principle explicit in the README.

## Stack
- Python 3.11+, LangGraph for orchestration (StateGraph with typed state), pandas, scikit-learn, LightGBM, optbinning or an equivalent WoE/IV binning library, pydantic for state and config schemas, pytest, ruff.
- Config in YAML. Secrets in .env (GEMINI_API_KEY), and .env must be gitignored.
- data/ must be gitignored. Include clear instructions for downloading the Kaggle data. Never commit raw data.
- The dataset is large, so downcast dtypes and be memory-conscious. Also support a --sample flag (for example 10% of customers, stratified) for fast development runs.

## Data split (my specification)
1. Split the full application_train population 75:25, stratified on TARGET with a fixed seed. The 25% is the out-of-time (OOT) holdout.
2. Split the remaining 75% into 75:25 for train and test, stratified on TARGET.
3. Final proportions: train 56.25%, test 18.75%, OOT 25%.
4. Home Credit has no application dates, so this OOT is a pseudo-OOT holdout, not a true temporal split. Document this honestly as a known limitation in the README and the validation report, and explain its implications for PSI and stability testing. Do not claim temporal validation.
5. Split on SK_ID_CURR before any feature engineering that learns from data. Binning, WoE, feature selection and calibration are fitted on train only. Leakage prevention must be explicit and tested.

## Agents and orchestration order (LangGraph)
Build these as graph nodes with typed state. Each node declares which state keys it reads and writes.

0. **Orchestrator / run setup.** Load config, create run_id, initialise state, start a JSONL audit trail logging every node's start, end, inputs summary, outputs summary, gate decisions and timings.
1. **Ingestion and data quality agent.** Load tables, run schema, missingness and duplicate checks, and confirm the target rate. Gate: halt the run on critical data failures.
2. **Split agent.** Performs the 75:25 then 75:25 split above and persists the ID lists.
3. **Leg agents (parallel fan-out, then join).** One agent per data leg, mirroring a multi-leg PD design:
   - Application leg: application_train
   - Bureau leg: bureau and bureau_balance
   - Behavioural leg: installments_payments, POS_CASH_balance, credit_card_balance
   - Previous applications leg: previous_application
   Each aggregates to SK_ID_CURR, runs its own quality checks and writes a feature table plus a feature dictionary (name, definition, source table, leg).
4. **Feature assembly and selection agent.** Merge the legs and select features on train only using IV thresholds, correlation and VIF pruning, and stability across train and test. Log every exclusion with a reason.
5. **Modelling agent.** Champion: a WoE logistic regression scorecard, with the scorecard points table exported. Challenger: LightGBM. Add probability calibration if needed, and map PD to a master scale of rating grades.
6. **Validation agent (independent, read-only).** It cannot modify model artefacts. Compute Gini/AUC, KS, PSI and CSI on train, test and OOT; a calibration test (for example Hosmer-Lemeshow or a binomial test by grade); rank-ordering; and segment performance. Compare champion and challenger. Issue structured findings with severity levels. Gates (thresholds in config): for example, test Gini at least 0.45 and PSI above 0.25 counts as red. A red result routes to a halt node with a clear report.
7. **ECL agent.** See the ECL section below.
8. **Critic agent (LLM).** Reviews validation and ECL outputs like a skeptical model validator and raises challenges. It may request additional deterministic analyses only from a whitelisted menu (for example extra segment cuts, sensitivity on the LGD assumption, or alternative binning stability checks). The orchestrator runs those and the critic re-reviews. The loop is bounded to at most 2 iterations, and every request is logged in the audit trail. It can never change model parameters.
9. **Documentation agent (LLM).** Drafts the documents listed in the Documentation section, section by section.
10. **Grounding check node (deterministic).** Extract every number from the generated documents and verify that it exists in the run's metrics JSON within rounding. Flag unverified numbers. This is an anti-hallucination control and should be highlighted in the README.
11. **Report assembly.** Write all outputs to outputs/<run_id>/ and a summary to the console.

Use conditional edges for the gates and the critic loop. Generate the Mermaid graph diagram programmatically from the compiled LangGraph and embed it in the README.

## ECL module (RBI ECL framework)
Context: RBI issued the Reserve Bank of India (Commercial Banks - Asset Classification, Provisioning and Income Recognition) Directions, 2026 (RBI/DOR/2026-27/398, dated 27 April 2026), with ECL provisioning effective 1 April 2027. Verify every regulatory parameter against the actual text before hardcoding. Put all parameters in config with a source note.

Home Credit is an application-level dataset, not a live portfolio. Build the ECL layer as a clearly labelled illustrative simulation on the OOT population treated as a reporting-date book.
- **Staging.**
  - Stage 3: credit-impaired, 90+ DPD. Use delinquency derived from installments_payments and bureau history, or define a documented proxy.
  - Stage 2: significant increase in credit risk. Use the rebuttable 30+ DPD presumption from installment payment delays, plus a relative PD deterioration or rating-grade downgrade criterion. Document how the deterioration criterion is approximated without an origination-date PD.
  - Stage 1: everything else.
- **12-month PD** comes from the champion model, with a regulatory PD floor if one is specified in the final directions (verify the value).
- **Lifetime PD term structure.** Baseline is constant-hazard conversion. Also implement a rating-grade Markov transition matrix approach. Remaining term is derived from AMT_CREDIT and AMT_ANNUITY. Compare both methods.
- **Forward-looking adjustment.** Base, upside and downside macro scenarios with probability weights and PD scalars, all in config, labelled illustrative.
- **LGD.** A configurable assumption, for example 45% for unsecured. Run sensitivity on it.
- **EAD.** An amortising balance schedule from credit amount, annuity and term.
- **Discounting.** Use an effective interest rate approximated from the annuity formula.
- **Minimum provisions.** Apply any stage-wise minimum provisioning floors in the final directions (verify the values; for example, I understand Stage 2 has a minimum floor).
- **Output.** ECL by stage, rating grade and scenario, and a stage migration summary.

## LLM layer
- Provider abstraction with three modes, selected by config or CLI flag:
  1. **dry-run** (default): canned mock responses, no API calls, zero cost. Use this for all development and tests.
  2. **gemini**: Google Gemini via the official google-genai SDK, key from GEMINI_API_KEY. Model names in config: a cheap model (Flash-Lite or Flash) for test runs, a Pro-tier model for the final run. Verify current model IDs from Google's docs.
  3. **ollama**: a local model as the no-cost fallback.
- Prompts live in versioned files under prompts/, not inline strings.
- Pass only summarised metrics and tables to the LLM, never raw loan-level data.
- Track input and output tokens per call and print an estimated cost per run, using rates in config.
- If no key or local model is available, the pipeline completes the deterministic nodes and skips the LLM nodes with a clear message.

## Documentation outputs (generated per run)
1. Model Development Document: purpose and use, data, methodology, feature selection, performance, limitations, assumptions.
2. Independent Validation Report: conceptual soundness, outcomes analysis, benchmarking against the challenger, findings with severity and remediation.
3. Ongoing Monitoring Plan: metrics, thresholds, frequency, triggers, escalation.
4. ECL Methodology Note: staging, lifetime PD, scenarios, LGD and EAD, results, limitations.
Structure them in line with SR 11-7 and RBI's model risk management guidance (verify the current status of RBI's MRM guidance; I believe a draft was issued in 2026). Output Markdown, plus PDF if straightforward.

## README (hosted on GitHub; this is the most important deliverable)
It needs a very detailed explanation of how the agents orchestrate. Include:
- A one-paragraph summary and the core design principle.
- The auto-generated LangGraph Mermaid diagram.
- A step-by-step execution order. For each node: its role, inputs, outputs, state keys read and written, gates, and what happens on failure.
- A data-flow diagram from the raw tables, through the legs, features, model, validation, ECL and critic, to the documents.
- The typed state schema explained in plain English.
- How the parallel fan-out and join work for the leg agents.
- How the critic loop works, why it is bounded, and why it cannot touch the model.
- The grounding check and other model-risk controls (audit trail, seeds, read-only validator, train-only fitting).
- Results summary tables: Gini/KS/PSI across train, test and OOT, champion versus challenger, and ECL by stage. Fill these from the final run.
- The regulatory mapping: which SR 11-7 and RBI ECL elements each component addresses.
- How to run it: setup, data download, dry-run, Gemini, Ollama, and --sample.
- The cost note, with token tracking and typical cost per run.
- An honest limitations section: pseudo-OOT, illustrative ECL, LGD assumption, no true origination PD.
- Project structure.
- An About the Author section with links to my LinkedIn and GitHub.

## How I want us to work
- Build in phases and stop for my review after each one:
  1. Skeleton, state schema, graph wiring and a dry-run on --sample
  2. Data quality, split and leg agents
  3. Feature selection and modelling
  4. Validation and gates
  5. ECL
  6. LLM critic and documentation, plus the grounding check
  7. README and final polish
- Write pytest tests for the splits (proportions, stratification, no ID overlap), leakage prevention, gate logic and the grounding check.
- Commit after each phase with clear messages. Do not push to GitHub until I say so; when I do, help me create the repo and push it.
- Use dry-run mode for all development. Only call Gemini when I explicitly ask, and tell me the estimated cost first.
- If anything in this brief is technically unsound or conflicts with the regulations, flag it before building rather than silently changing it.
