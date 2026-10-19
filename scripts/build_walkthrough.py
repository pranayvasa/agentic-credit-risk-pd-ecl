"""Build (and optionally execute) notebooks/pipeline_walkthrough.ipynb.

The notebook walks through the pipeline in execution order. For each stage it explains what
the stage does, displays the real source code from the ``acr`` package (so the notebook can
never drift from the code), and runs the stage on a 10% stratified sample.

Only aggregates are displayed, never loan-level rows: the Kaggle competition rules do not
allow redistributing the data, and this notebook is committed.

Usage:
    .venv/bin/python scripts/build_walkthrough.py            # build and execute
    .venv/bin/python scripts/build_walkthrough.py --no-run   # build only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "pipeline_walkthrough.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------------------
md("""
# Pipeline walkthrough: agentic credit risk (PD → validation)

This notebook walks through every piece of code in the pipeline **in the order it runs**, and
executes each step on a **10% stratified sample** of the public Home Credit data.

**How to read it**

* Each section starts with a short explanation of *what* the stage does and *why*.
* A `show(...)` cell then prints the **actual source code** from the `acr` package. The code
  lives in `src/acr/`, and the notebook displays it rather than copying it, so the two can
  never disagree.
* A run cell (with comments) then executes the stage and shows what it produced.
* Only **aggregates** are displayed (counts, rates, IVs, metrics), never individual loans.

**Core design principle.** LLM agents never fit, tune or alter the model. Everything shown in
this notebook is deterministic: fixed seed, versioned config, the same inputs give the same
numbers. The LLM critic and documentation agents (Phase 6) sit *around* this core.

**Status.** All phases are built (the LLM nodes run in dry-run mode). This walkthrough covers
the deterministic core; Phases 1–5 are built: skeleton, data quality, split, legs, feature selection,
modelling (plus offline tuning), independent validation and the illustrative RBI ECL layer.
The LLM critic, documentation and the grounding check are still stubs; they come in Phase 6.

| § | Stage | Graph node(s) | Main source files |
|---|---|---|---|
| 1 | Configuration | (input to setup) | `config.py`, `config/*.yaml` |
| 2 | State and node contract | (all nodes) | `state.py`, `audit.py` |
| 3 | Graph wiring | (orchestration) | `graph.py` |
| 4 | Run setup | `setup` | `nodes/setup.py` |
| 5 | Ingestion and data quality | `ingest_dq` | `core/io.py`, `core/dq.py` |
| 6 | Split | `split` | `core/split.py` |
| 7 | Four leg agents | `leg_*` | `core/legs/*.py` |
| 8 | Feature selection | `select` | `core/binning.py`, `core/selection.py` |
| 9 | Modelling | `model` | `core/scorecard.py`, `core/challenger.py`, `core/calibration.py` |
| 10 | Independent validation | `validate` | `core/validation.py`, `core/findings.py` |
| 11 | Illustrative RBI ECL | `ecl` | `core/ecl/*.py`, `nodes/ecl.py` |
| 11b | Gates and the rest of the graph | routing, `halt` | `graph.py`, `nodes/halt.py` |
| 12 | Offline tuning | `acr tune` (not in the graph) | `tuning.py` |
| 13 | Tests | | `tests/` |
""")

md(
    "## 0. Setup\n\nImports, a `show()` helper to display source code, and a `step()` helper to run one graph node and merge its output into the running state."
)
code('''
# --- Imports --------------------------------------------------------------------------
import inspect
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Code, display

# Make the repo importable when the notebook is opened from notebooks/
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)


def show(obj):
    """Display the real source code of a module, class or function from the acr package."""
    display(Code(inspect.getsource(obj), language="python"))


def show_file(rel_path):
    """Display a non-Python file (e.g. YAML config) from the repo."""
    p = ROOT / rel_path
    display(Code(p.read_text(), language=p.suffix.lstrip(".") or "text"))


def step(node_fn, state):
    """Run one graph node the way LangGraph would, and merge its output into the state.

    Nodes return only the keys they declared in `writes`. The `legs` key uses a merge
    reducer in the real graph, which we mimic here with a dict update.
    """
    out = node_fn(state)
    for k, v in out.items():
        if k in ("legs",):
            state[k] = {**state.get(k, {}), **v}
        elif k in ("gate_log", "llm_calls", "critic_reviews", "extra_analyses"):
            state[k] = state.get(k, []) + v
        else:
            state[k] = v
    print(f"{node_fn.spec.name}: wrote {sorted(out)}")
    return out
''')

# ---------------------------------------------------------------------------------------
md("""
## 1. Configuration

All settings live in versioned YAML files under `config/`, validated by **pydantic** models
in `config.py`. Unknown keys are rejected, and limits are enforced in the schema itself
(for example at most **15** scorecard variables and LightGBM `max_depth` between 1 and 10).
The merged config is hashed, and the hash goes into the `run_id`, so every output can be
traced to the exact settings that produced it.

Files: `base.yaml` (data, split, features, model), `gates.yaml` (gate and finding
thresholds), `llm.yaml` (LLM provider, default dry-run), `ecl_rbi.yaml` (RBI ECL parameters
with paragraph citations), `tuning.yaml` (offline tuning grids).
""")
code("""
# The modelling and data configuration
show_file("config/base.yaml")
""")
code("""
# Gates (what halts a run) and the validator's finding thresholds
show_file("config/gates.yaml")
""")
code("""
# The pydantic schema that validates every YAML file
import acr.config
show(acr.config)
""")
code("""
# Load the config exactly as the CLI does. The default LLM mode is dry-run (no API calls).
from acr.config import load_config

cfg = load_config(llm_mode="dry-run")
print("config hash:", cfg.hash())
print("seed:", cfg.base.seed)
print("scorecard feature cap:", cfg.base.features.max_features)
print("LightGBM params:", cfg.base.model.lightgbm)
""")

# ---------------------------------------------------------------------------------------
md("""
## 2. Typed state and the node contract

**State.** `PipelineState` is the single typed object that flows through the LangGraph
graph. It carries *artefact paths, metrics and findings*, never DataFrames. Tables are saved
as parquet under `outputs/<run_id>/artefacts/`. This keeps the state auditable, and it means
anything later handed to an LLM is a summary by construction.

**Node contract.** Every node is wrapped by the `@node(...)` decorator in `audit.py`, which:
1. gives the node a **read-only view** holding only the state keys it declared in `reads`;
2. **rejects** any returned key not declared in `writes`;
3. writes start/end/error events, input and output summaries and timings to `audit.jsonl`.
""")
code("""
import acr.state
show(acr.state)
""")
code("""
import acr.audit
show(acr.audit)
""")

# ---------------------------------------------------------------------------------------
md("""
## 3. Graph wiring

`graph.py` assembles the LangGraph `StateGraph`:

* **Conditional edges** implement the gates: after data quality (`ingest_dq` → `split` or
  `halt`) and after validation (`validate` → `ecl` or `halt`).
* **Fan-out and join**: `split` fans out to the four leg nodes, which run in parallel, and
  `select` waits for all four.
* **Bounded critic loop**: `critic` → `run_extra_analyses` → `critic`, at most two rounds.

The Mermaid diagram below is generated from the compiled graph, not drawn by hand.
""")
code("""
import acr.graph
show(acr.graph)
""")
code("""
# Mermaid source generated from the compiled LangGraph (dotted arrows = conditional edges)
from acr.graph import build_graph, mermaid

graph = build_graph()
print(mermaid())
""")
code("""
# Every node's declared contract, straight from the decorator registry
from acr.audit import NODE_REGISTRY

pd.DataFrame([
    {"node": s.name, "uses_llm": s.uses_llm, "reads": ", ".join(s.reads), "writes": ", ".join(s.writes)}
    for s in NODE_REGISTRY.values()
])
""")

# ---------------------------------------------------------------------------------------
md("""
## 4. Run setup (node `setup`)

Validates the config, builds a `run_id` (timestamp, scope, config hash), creates
`outputs/<run_id>/`, snapshots the config and starts the audit trail.

Below we run the nodes **one at a time** on a 10% sample, the same way the graph would,
so we can inspect each step. (`acr run --sample` does all of this in one go.)
""")
code("""
import acr.nodes.setup
show(acr.nodes.setup)
""")
code("""
from acr.nodes.setup import setup

# Initial input to the graph: validated config, sample fraction, LLM mode
state = {"config": cfg.model_dump(mode="json"), "sample_frac": 0.10, "llm_mode": "dry-run"}
step(setup, state)
print("run_id:", state["run_id"])
""")

# ---------------------------------------------------------------------------------------
md("""
## 5. Ingestion and data quality (node `ingest_dq`)

**Memory-conscious loading (`core/io.py`).** The raw CSVs (2.5 GB) are converted once into a
downcast, zstd-compressed parquet cache (about 513 MB): floats to float32 (amounts used in
ECL stay float64), integer-valued columns to the smallest integer type, strings to
dictionary-encoded categoricals. Later reads select only the needed columns and filter rows
by ID at read time.

**Data-quality checks (`core/dq.py`).** Each check has a severity that applies if it fails.
A failed **critical** check (missing table or column, duplicate IDs, non-binary target)
sets `status = "halt"`, and the graph's conditional edge routes to the `halt` node.
DQ always runs on the full population, even on a sample run.
""")
code("""
import acr.core.schema
show(acr.core.schema)
""")
code("""
import acr.core.io
show(acr.core.io)
""")
code("""
import acr.core.dq
show(acr.core.dq)
""")
code("""
import acr.nodes.ingest_dq
show(acr.nodes.ingest_dq)
""")
code("""
from acr.nodes.ingest_dq import ingest_dq

step(ingest_dq, state)

# Tables loaded into the cache
display(pd.DataFrame(state["dq"]["tables"]).T[["rows", "cols"]])

# Every check: failed ones first. Severity applies only when a check fails.
checks = pd.DataFrame(state["dq"]["checks"])[["table", "name", "severity", "passed", "detail"]]
display(checks.sort_values("passed"))
print("DQ status:", state["dq"]["status"], "| target rate:", state["dq"]["target_rate"])
""")

# ---------------------------------------------------------------------------------------
md("""
## 6. Split (node `split`)

Exactly as specified: the full population (or the stratified sample) is split **75:25** into
development and OOT, then development is split **75:25** into train and test, both
stratified on `TARGET`, with a fixed seed. Final shares: **56.25 / 18.75 / 25**.

The split happens on `SK_ID_CURR` **before anything learns from the data**. Home Credit has
no application dates, so the OOT is a **pseudo-OOT** (random holdout), not a temporal one.

**`application_test.csv`** is Kaggle's scoring set and has **no TARGET**, so it can't be a
validation partition. It is kept separate as an **external, unlabelled** population. Legs
build its features and the model scores it, but nothing is fitted on it. It feeds a
score-PSI and CSI check in validation, and full runs export Kaggle submission files.
""")
code("""
import acr.core.split
show(acr.core.split)
""")
code("""
from acr.nodes.split import split

step(split, state)
s = state["split"]
display(pd.DataFrame({"count": s["counts"], "share": s["proportions"], "default rate": s["target_rate"]}))
print("assignment hash (same seed + same population = same hash):", s["assignment_hash"])
""")

# ---------------------------------------------------------------------------------------
md("""
## 7. The four leg agents (nodes `leg_application`, `leg_bureau`, `leg_behavioural`, `leg_previous`)

One agent per data leg, mirroring a multi-leg PD design. In the graph they run **in
parallel** after `split`. Each aggregates its tables to one row per customer (`SK_ID_CURR`),
runs leg-level quality checks, and writes a feature table plus a **feature dictionary**
(name, definition, source table, leg, role).

**Leakage.** Leg features are per-customer aggregations of that customer's *own* records.
Leg builders are never given `TARGET`, so they can run on all partitions. Tests confirm that
features are unchanged when targets are shuffled.

**Protected characteristics.** `CODE_GENDER` and `NAME_FAMILY_STATUS` have role
`segment_only`: they're used for fairness analysis and never offered to the model.
""")
code("""
import acr.core.legs.common
show(acr.core.legs.common)
""")
code("""
import acr.core.legs.application
show(acr.core.legs.application)
""")
code("""
import acr.core.legs.bureau
show(acr.core.legs.bureau)
""")
code("""
import acr.core.legs.behavioural
show(acr.core.legs.behavioural)
""")
code("""
import acr.core.legs.previous
show(acr.core.legs.previous)
""")
code("""
# The thin node wrapper shared by all four legs
import acr.nodes.legs._base
show(acr.nodes.legs._base)
""")
code("""
from acr.nodes.legs import LEG_NODES

# In the graph these four run in parallel; here we run them one after another.
for fn in LEG_NODES.values():
    step(fn, state)

summary = pd.DataFrame({
    leg: {"features": v["n_features"], "coverage": v["dq"]["coverage"],
          "mean missing share": v["dq"]["missing_share_mean"], "issues": len(v["dq"]["issues"]),
          "seconds": v["seconds"]}
    for leg, v in state["legs"].items()
}).T
display(summary)
""")
code("""
# A slice of the feature dictionary (definitions only, no data)
dictionary = pd.concat(pd.read_csv(v["dictionary_path"]) for v in state["legs"].values())
display(dictionary.groupby("leg").head(4)[["leg", "name", "definition", "source_table", "role"]])
""")

# ---------------------------------------------------------------------------------------
md("""
## 8. Feature selection (node `select`): train only

The four legs are joined, and scorecard features are selected **using train only**. Test is
used for one thing: the train-vs-test stability (CSI) screen, which uses test *features*,
never test targets.

Binning uses **optbinning** (monotonic WoE bins). WoE convention: `ln(%good / %bad)`, so a
higher WoE means lower risk, and every scorecard coefficient should be negative.

The stages, each logged with a reason:

1. role (protected characteristics)
2. missingness above 95%
3. single bin
4. IV below 0.02, or above 0.5 (possible leakage)
5. CSI above 0.10
6. |correlation| above 0.7
7. VIF above 5
8. cap at 15 features
9. backward elimination: wrong coefficient sign, then p-value above 0.05
""")
code("""
import acr.core.guards
show(acr.core.guards)   # every fitter calls this: non-train rows raise LeakageError
""")
code("""
import acr.core.binning
show(acr.core.binning)
""")
code("""
import acr.core.selection
show(acr.core.selection)
""")
code("""
import acr.nodes.select
show(acr.nodes.select)
""")
code("""
from acr.nodes.select import select

step(select, state)
sel = state["selection"]
print(f"{sel['n_features_total']} features -> {sel['n_candidates']} candidates -> "
      f"{sel['n_screened']} after screens -> {sel['n_selected']} selected")
print("exclusions by stage:", sel["exclusions_by_stage"])
display(pd.DataFrame(sel["selected"]))
""")
code("""
# Every exclusion with its reason (first rows of each stage)
ex = pd.read_csv(sel["paths"]["exclusions"])
display(ex.groupby("stage").head(3)[["stage", "feature", "leg", "reason", "value"]])
""")

# ---------------------------------------------------------------------------------------
md("""
## 9. Modelling (node `model`): train only

* **Champion: monotone LightGBM** (`core/challenger.py`) on all candidate features, with
  **fixed** hyperparameters from config (tuned offline, §12) and **monotonic constraints** taken
  from each feature's train-only WoE trend. Early stopping uses a slice of train, never test or
  OOT. `model.champion` in config assigns the roles (decision D33).
* **Benchmark challenger and fallback: WoE logistic regression scorecard** (`core/scorecard.py`).
  It exports a coefficient table (with standard errors and p-values) and a **points table** using
  the PDO method (PDO 20, base score 600 at 50:1 odds). A higher score means lower risk.
* **Calibration** (`core/calibration.py`). Out-of-fold train predictions decide whether
  Platt recalibration is needed (decile ECE above 0.005 or slope outside 1 ± 0.1).
* **Master scale** (`core/master_scale.py`). PD maps to grades 1–10.
* **Origination proxy.** An application-leg-only model of the champion's type, used later by
  the ECL layer's relative SICR test (no true origination PD exists in this data).
* **Explainability** (node `explain`, `core/explain.py`): exact TreeSHAP for every OOT applicant,
  reason codes grouped into families, driver stability and a sweep test of the constraints.
""")
code("""
import acr.core.metrics
show(acr.core.metrics)
""")
code("""
import acr.core.scorecard
show(acr.core.scorecard)
""")
code("""
import acr.core.challenger
show(acr.core.challenger)
""")
code("""
import acr.core.calibration
show(acr.core.calibration)
""")
code("""
import acr.core.master_scale
show(acr.core.master_scale)
""")
code("""
import acr.nodes.model
show(acr.nodes.model)
""")
code("""
from acr.nodes.model import model

step(model, state)
m = state["model"]
display(pd.DataFrame({k: {kk: m[k].get(kk) for kk in ("type", "n_features", "train_auc", "train_gini", "oof_auc", "oof_gini", "calibration")}
                      for k in ("champion", "challenger")}).T)
print("roles:", m["roles"])
print("origination proxy features:", m["origination_proxy"]["features"])
""")
code("""
# Scorecard (benchmark) coefficient table: every WoE coefficient should be negative
display(pd.read_csv(m["paths"]["coefficients"]).round(4))
""")
code("""
# Scorecard points for one characteristic (bins, WoE, points). Higher points = lower risk.
pts = pd.read_csv(m["paths"]["points"])
first = pts["feature"].iloc[0]
display(pts[pts["feature"] == first][["feature", "bin", "count", "event_rate", "woe", "points_rounded"]])
""")

code("""
import acr.core.explain
show(acr.core.explain)
""")
code("""
from acr.nodes.explain import explain

step(explain, state)
X = state["explain"]
print("applicants explained:", X["applicants_explained"], "| constraint violations:", X["monotone"]["violating_rows"])
display(pd.Series(X["family_shares"]).rename("share of SHAP weight").to_frame().style.format("{:.1%}"))
""")

# ---------------------------------------------------------------------------------------
md("""
## 10. Independent validation (node `validate`): read-only

The second-line validator works only on **scored outputs**. It imports no fitting code, and it
checks every model artefact's SHA-256 against the checksums the model node recorded,
**before and after** it runs. A mismatch is a red finding.

What it computes:
* Gini/AUC, KS and Brier by partition
* score PSI and per-feature CSI, plus a stress-PSI ladder
* calibration: HL, ECE, slope, and a one-sided binomial traffic light by grade
* rank-ordering
* OOT segments, including gender as a fairness check
* champion vs challenger, with a bootstrap 95% CI
* an external score-PSI and CSI check on the unlabelled `application_test` population

Every Gini is reported with its **AUC** (Gini = 2 × AUC − 1), the metric Kaggle uses.

`core/findings.py` turns these into **structured findings** with severities (info → red),
each stating its threshold. Only **red** findings drive the gate: they halt full runs and
only warn on `--sample`.
""")
code("""
import acr.core.validation
show(acr.core.validation)
""")
code("""
import acr.core.findings
show(acr.core.findings)
""")
code("""
import acr.nodes.validate
show(acr.nodes.validate)
""")
code("""
from acr.nodes.validate import validate

step(validate, state)
v = state["validation"]
perf = {f"{model_} {part}": x for model_ in ("champion", "challenger")
        for part, x in v["metrics"][model_]["performance"].items()}
display(pd.DataFrame(perf).T[["n", "default_rate", "mean_pd", "auc", "gini", "ks", "brier"]].round(4))
print("gate decision:", v["gate_decision"], "| findings by severity:", v["findings_by_severity"])
""")
code("""
# Structured findings (sample run: the gates are warn-only)
display(pd.DataFrame(v["findings"])[["id", "severity", "area", "title", "detail"]])
""")
code("""
# Grade-level calibration on OOT with binomial traffic lights
display(pd.read_csv(v["paths"]["grade_table_oot"]).round(4))
""")
code("""
# Segment performance on OOT (gender is a fairness check; it is not a model input)
display(pd.read_csv(v["paths"]["segments_oot"]).round(3))
""")

# ---------------------------------------------------------------------------------------
md("""
## 11. Illustrative RBI ECL (node `ecl`)

**Illustrative simulation.** Home Credit is an application-level dataset, not a live book.
The pseudo-OOT partition is treated as a book at a simulated reporting date, with every loan
at origination. Regulatory parameters are in `config/ecl_rbi.yaml` with paragraph
citations to RBI/DOR/2026-27/398 (Directions dated 27 April 2026, effective 1 April 2027).

| Component | Approach | Source |
|---|---|---|
| Stage 3 | more than 90 DPD in the last 12 months on prior Home Credit loans (borrower level) | para 25, NPA definition |
| Stage 2 | more than 30 DPD (Home Credit or bureau), or grade down ≥ 3 notches vs the origination proxy, or PD ≥ 2× proxy PD and ≥ 5% | para 33 presumption; para 26 relative test (proxy, D6) |
| 12-month PD | champion PD, floored at 0.03% | para 96 |
| Lifetime PD | constant hazard (primary) and a bureau DPD-bucket Markov chain scaled to each 12-month PD | D4 |
| Term and EIR | monthly rate implied by prior approved Home Credit loans (train customers); term solved from the annuity formula | D26 |
| EAD | amortising balance; revolving: limit × CCF over a 24-month life | para 46(2)(iii) |
| LGD | 70% unsecured backstop, with a sensitivity ladder | para 97 |
| Scenarios | base / upside / downside, weights and PD scalars (illustrative) | para 48 |
| Floors | Stage 1 1%, Stage 2 5%, Stage 3 25% in the first year | para 82(3)(x) |

The engine works at LGD = 1, so the LGD sensitivity is exact (ECL is linear in LGD).
Implied rates and the transition matrix are learned from **train customers only**.
""")
code("""
import acr.core.ecl.rates
show(acr.core.ecl.rates)
""")
code("""
import acr.core.ecl.staging
show(acr.core.ecl.staging)
""")
code("""
import acr.core.ecl.lifetime
show(acr.core.ecl.lifetime)
""")
code("""
import acr.core.ecl.engine
show(acr.core.ecl.engine)
""")
code("""
import acr.nodes.ecl
show(acr.nodes.ecl)
""")
code("""
from acr.nodes.ecl import ecl

step(ecl, state)
E = state["ecl"]
print("book:", E["book"])
print("implied monthly rates:", E["assumptions"]["implied_monthly_rates"])
print("stage counts:", E["stage_counts"], "| triggers:", E["stage_triggers"])
print("QC checks:", E["checks"])
""")
code("""
# ECL by stage: model ECL, after RBI floors (final), coverage and realised default rate
display(pd.read_csv(E["paths"]["by_stage"], index_col=0).round(4))
""")
code("""
# Scenarios (staging is fixed; PD scalars move ECL), lifetime-method comparison, LGD sensitivity
display(pd.read_csv(E["paths"]["by_scenario"], index_col=0).round(0))
display(pd.read_csv(E["paths"]["methods"], index_col=0).round(4))
display(pd.read_csv(E["paths"]["lgd"]).round(4))
""")
code("""
# Monthly DPD-bucket transition matrix estimated from bureau_balance (train customers)
display(pd.read_csv(E["paths"]["transition_matrix"], index_col=0).round(4))
""")

# ---------------------------------------------------------------------------------------
md("""
## 11b. Gates and the rest of the graph

The routing functions in `graph.py` (shown in §3) read the status each gate node wrote and
pick the next node; each routing decision is logged to the audit trail. If a gate says
`halt`, the `halt` node writes `halt_report.md` listing the red findings.

Still stubs, built in Phase 6: `critic`, `run_extra_analyses`, `document` and
`grounding_check`. The critic loop already runs in dry-run mode
with canned responses, to prove the bounded loop and the whitelist work.
""")
code("""
import acr.nodes.halt
show(acr.nodes.halt)
""")
code("""
# The audit trail written so far: one line per node start/end, gate and route
from acr.audit import read_events

events = pd.DataFrame(read_events(state["run_dir"]))
display(events[events["event"].isin(["node_end", "run_start"])][["ts", "event", "node", "seconds"]])
""")

# ---------------------------------------------------------------------------------------
md("""
## 12. Offline tuning (`acr tune`): not part of the graph

Hyperparameters were tuned **offline**, by deterministic 3-fold cross-validation **inside the
train partition only**. Test and OOT are never read. The winner is the **simplest
configuration within one standard error** of the best CV Gini. The tuner writes a report;
the winners were reviewed and then frozen into `base.yaml` by hand (decision D22).

Not re-run here (it takes about 35 minutes on the full data). The code:
""")
code("""
import acr.tuning
show(acr.tuning)
""")

# ---------------------------------------------------------------------------------------
md("""
## 13. Tests

The test suite runs on a **synthetic** Home Credit-shaped dataset (`tests/synthetic.py`),
never the Kaggle files. What each file guards:

| File | Guards |
|---|---|
| `test_split.py` | 56.25/18.75/25 proportions, stratification, no ID overlap, determinism, sampling |
| `test_dq.py` | DQ checks and the DQ halt gate |
| `test_io.py` | downcasting preserves values; cache reuse and rebuild |
| `test_legs.py` | leg shape and dictionaries; legs never touch TARGET; hand-checked DPD maths |
| `test_modelling.py` | **leakage**: shuffling test/OOT targets changes nothing learned; reproducibility; points table reproduces scores |
| `test_tuning.py` | tuning ignores holdout targets, is deterministic, respects the 15-variable and depth-10 limits |
| `test_validation.py` | gate logic; a tampered artefact halts the run; the validator imports no fitting code |
| `test_contract.py` | the node read/write contract is enforced |
| `test_gates.py` | halt routing; critic loop bounded at 2; off-menu critic requests rejected; LLM nodes skip cleanly |
| `test_graph.py`, `test_config.py` | graph wiring and fan-out/join; config validation and limits |

Run them with `.venv/bin/pytest`.
""")


def build(run: bool) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    OUT.parent.mkdir(exist_ok=True)
    if run:
        from nbclient import NotebookClient

        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(OUT.parent)}}).execute()  # fmt: skip
    nbf.write(nb, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(cells)} cells, executed={run})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    build(not ap.parse_args().no_run)
