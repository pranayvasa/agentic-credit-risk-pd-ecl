"""Build the beginner notebook: notebooks/credit_risk_explained.ipynb.

Plain Python, top to bottom: pandas, scikit-learn, optbinning, statsmodels and LightGBM used
directly (no `acr` package calls). Each step rebuilds what the pipeline did and checks its own
numbers against the pipeline's saved results for one full run.

Outputs:
  notebooks/local/credit_risk_explained.ipynb   executed, with all outputs (gitignored: it
                                                shows raw Kaggle rows, which must not be published)
  notebooks/local/credit_risk_explained.html    the same, as a web page
  notebooks/credit_risk_explained.ipynb         the same notebook with outputs cleared (committed)

Usage:  .venv/bin/python scripts/build_explained.py [--no-run]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "notebooks" / "credit_risk_explained.ipynb"
LOCAL = ROOT / "notebooks" / "local" / "credit_risk_explained.ipynb"
cells: list = []


def md(t):
    cells.append(nbf.v4.new_markdown_cell(t.strip()))


def code(t):
    cells.append(nbf.v4.new_code_cell(t.strip()))


# =========================================================================================
md("""
# Credit risk, explained step by step

This notebook rebuilds the whole project **in plain Python**, one small step at a time, so you can see
exactly what happens to the data. It's written for someone new to Python.

**How each step is laid out**

1. **What and why**: a short explanation in plain English.
2. **The code**: short cells using pandas, scikit-learn and friends directly. Comments starting with `#` explain each line.
3. **The output**: tables (`.head()` shows the first rows), counts and charts.
4. **The check**: a line like `✓ matches the pipeline`. It compares the number we just
   built by hand with the number the real pipeline produced, so you know this notebook is a faithful
   explanation and not a different model.

**Before running.** The pipeline must have been run once on the full data (`acr run`), because the
checks compare against that run's saved results in `outputs/`.

**Python basics you'll see**

* `df.head()` shows the first 5 rows of a table (a *DataFrame*). `df.shape` is `(rows, columns)`.
* `df["COL"]` picks one column. `df[df["COL"] > 0]` keeps only the rows where the condition is true.
* `df.groupby("KEY").agg(...)` means "for each value of KEY, summarise the other columns".
* `round(x, 4)` rounds to 4 decimals; `f"{x:.2%}"` prints `0.0807` as `8.07%`.

| Part | What you'll learn |
|---|---|
| 1 | Look at the data |
| 2 | Data-quality checks |
| 3 | Splitting into train, test and OOT |
| 4 | Feature engineering, by example |
| 5 | Weight of evidence (WoE) and information value (IV) |
| 6 | Feature selection |
| 7 | The benchmark scorecard and scoring one applicant by hand |
| 8 | The champion: monotone LightGBM, SHAP and reason codes |
| 9 | Validation: is the champion any good? |
| 10 | Expected credit loss (ECL) for one loan, then the whole book |
| 11 | The "agents" around all of this |
| 12 | Glossary |
""")

md("""
## 0. Setup

We import the libraries, point at the data folder and at one completed pipeline run, and set a
simple chart style. `check()` is a tiny helper that prints ✓ when our number equals the
pipeline's (to a stated number of decimals).
""")
code('''
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 30)   # show up to 30 columns when printing a table
pd.set_option("display.width", 160)

# Where things are. Run this notebook from the repo root or from the notebooks/ folder.
ROOT = Path.cwd().parent if Path.cwd().name in ("notebooks", "local") else Path.cwd()
ROOT = ROOT.parent if ROOT.name == "notebooks" else ROOT
RAW = ROOT / "data" / "raw"          # the Kaggle CSV files
CACHE = ROOT / "data" / "interim"    # the same tables saved as parquet (much faster to read)

# The most recent full pipeline run with documents: we compare our numbers with it.
runs = sorted(p for p in (ROOT / "outputs").glob("*_full_*") if (p / "documents").exists())
RUN = runs[-1]
ART = RUN / "artefacts"
summary = json.loads((RUN / "run_summary.json").read_text())
print("Comparing against pipeline run:", RUN.name)

SEED = 42   # the fixed random seed used everywhere, so results are reproducible

# Chart style: light surface, recessive grid, two series colours (blue, orange).
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "figure.figsize": (7, 3.6),
})


def check(label, ours, pipeline, decimals=4):
    """Print a tick if our hand-built number equals the pipeline's at `decimals` decimals."""
    ok = round(float(ours), decimals) == round(float(pipeline), decimals)
    mark = "✓ matches the pipeline" if ok else "✗ DIFFERENT from the pipeline"
    print(f"{label}: ours {float(ours):.{decimals}f} | pipeline {float(pipeline):.{decimals}f}  {mark}")
''')

# =========================================================================================
md("""
## 1. Look at the data

Home Credit is a consumer lender. Kaggle published seven tables about **307,511 loan applicants**.
The main table, `application_train.csv`, has one row per applicant. The `TARGET` column is **1 if the
customer had payment difficulties** on their early instalments, and **0** otherwise.

Everything links through **`SK_ID_CURR`**, the applicant's ID.
""")
code("""
# Read the main table straight from the Kaggle CSV file.
app = pd.read_csv(RAW / "application_train.csv")

print("rows, columns:", app.shape)
app.head()
""")
md("""
**What to notice.** There are 122 columns. Some to know:

* `TARGET`: what we predict (1 = payment difficulties).
* `AMT_CREDIT` (loan amount), `AMT_ANNUITY` (monthly instalment), `AMT_INCOME_TOTAL` (income).
* `DAYS_BIRTH`, `DAYS_EMPLOYED`: counted **backwards** from the application date, so they're negative.
  `-12005` days is about 33 years.
* `EXT_SOURCE_1/2/3`: scores from undisclosed external data sources (more on these later).
""")
code("""
# How many applicants had payment difficulties?
print(app["TARGET"].value_counts())
print("default rate:", f"{app['TARGET'].mean():.2%}")
check("default rate", app["TARGET"].mean(), summary["dq"]["target_rate"])
""")
md("""
The other six tables have **many rows per applicant**: one per previous loan, or one per
month of history. Here's the shape of each, and the first few rows.
""")
code("""
tables = {
    "bureau": "credit with OTHER lenders (from the credit bureau), one row per credit",
    "bureau_balance": "monthly status of each bureau credit (0 = on time, 1-5 = days past due buckets)",
    "installments": "every repayment on PREVIOUS Home Credit loans",
    "pos_cash": "monthly snapshots of previous point-of-sale / cash loans",
    "credit_card": "monthly snapshots of previous Home Credit credit cards",
    "previous": "previous applications to Home Credit (approved, refused, cancelled)",
}
for name, meaning in tables.items():
    t = pd.read_parquet(CACHE / f"{name}.parquet")
    print(f"{name:15s} {t.shape[0]:>11,} rows x {t.shape[1]:>2} cols  : {meaning}")
""")
code("""
# One applicant across the tables: how many rows does customer 100002 have in each?
cust = 100002
bureau = pd.read_parquet(CACHE / "bureau.parquet")
inst = pd.read_parquet(CACHE / "installments.parquet")
prev = pd.read_parquet(CACHE / "previous.parquet")
print("bureau rows:", (bureau["SK_ID_CURR"] == cust).sum())
print("installment rows:", (inst["SK_ID_CURR"] == cust).sum())
print("previous applications:", (prev["SK_ID_CURR"] == cust).sum())
bureau[bureau["SK_ID_CURR"] == cust].head()
""")
md("""
**Why this matters.** A model needs **one row per applicant**. So for every child table we must
*summarise* the many rows into a few numbers per customer, for example "the worst days past due
in the last 12 months". That's **feature engineering** (Part 4).
""")

# =========================================================================================
md("""
## 2. Data-quality checks

Before modelling, we check that the data is usable: IDs are unique, the target is only 0 or 1,
we know which columns are mostly empty, and we look for odd values.
""")
code("""
print("duplicate applicant IDs:", app["SK_ID_CURR"].duplicated().sum())
print("TARGET values:", sorted(app["TARGET"].unique()))

# Share of missing values per column, largest first
missing = app.isna().mean().sort_values(ascending=False)
print("columns more than 50% empty:", (missing > 0.5).sum())
missing.head(8).to_frame("share missing")
""")
code("""
# An odd value: 365243 days (about 1,000 years) in DAYS_EMPLOYED is a placeholder,
# mostly for pensioners. The pipeline treats it as "missing" and adds a flag for it.
odd = (app["DAYS_EMPLOYED"] == 365243)
print("rows with DAYS_EMPLOYED = 365243:", odd.sum())
print("default rate with the placeholder:", f"{app.loc[odd, 'TARGET'].mean():.2%}",
      "| without:", f"{app.loc[~odd, 'TARGET'].mean():.2%}")
""")

# =========================================================================================
md("""
## 3. Splitting into train, test and OOT

We hold data back so we can test the model on applicants it has never seen.

1. Split all applicants **75 : 25** into *development* and *OOT* ("out-of-time" holdout).
2. Split development **75 : 25** into *train* and *test*.

That gives **56.25% / 18.75% / 25%**. *Stratified* means each part keeps the same default rate.
We split on the applicant ID **before** anything learns from the data, so nothing about the test or
OOT applicants can leak into the model.

**Honest caveat.** The data has no application dates, so the "OOT" is really a random holdout
(a *pseudo*-OOT). It can't tell us whether the model stays stable over time.
""")
code("""
from sklearn.model_selection import train_test_split

ids = app[["SK_ID_CURR", "TARGET"]].sort_values("SK_ID_CURR").reset_index(drop=True)

dev, oot = train_test_split(ids, test_size=0.25, stratify=ids["TARGET"], random_state=SEED)
train, test = train_test_split(dev, test_size=0.25, stratify=dev["TARGET"], random_state=SEED)

split_table = pd.DataFrame({
    "applicants": [len(train), len(test), len(oot)],
    "share": [len(train) / len(ids), len(test) / len(ids), len(oot) / len(ids)],
    "default rate": [train["TARGET"].mean(), test["TARGET"].mean(), oot["TARGET"].mean()],
}, index=["train", "test", "oot"])
split_table.style.format({"applicants": "{:,}", "share": "{:.2%}", "default rate": "{:.2%}"})
""")
code("""
# Same split as the pipeline? Compare the exact list of train IDs.
pipe = pd.read_parquet(ART / "split" / "assignment.parquet")
same = set(train["SK_ID_CURR"]) == set(pipe.loc[pipe["partition"] == "train", "SK_ID_CURR"])
print("train IDs identical to the pipeline:", "✓" if same else "✗")
print("no applicant in two parts:",
      "✓" if not (set(train.SK_ID_CURR) & set(test.SK_ID_CURR)) and not (set(train.SK_ID_CURR) & set(oot.SK_ID_CURR)) else "✗")
""")

# =========================================================================================
md("""
## 4. Feature engineering, by example

A **feature** is one number (or category) per applicant that might help predict default.
The pipeline builds **114 features** in four groups ("legs"): application, bureau, behavioural
and previous applications. We build a few of each by hand and check them against the pipeline.

### 4a. Application leg: ratios and ages
""")
code("""
f = app.set_index("SK_ID_CURR")
my = pd.DataFrame(index=f.index)

# Ratios: how big is the loan or instalment relative to income?
my["APP_CREDIT_TO_INCOME"] = f["AMT_CREDIT"] / f["AMT_INCOME_TOTAL"]
my["APP_ANNUITY_TO_INCOME"] = f["AMT_ANNUITY"] / f["AMT_INCOME_TOTAL"]   # debt-service burden

# Age in years (DAYS_BIRTH is negative days)
my["APP_AGE_YEARS"] = -f["DAYS_BIRTH"] / 365.25

# Years employed, treating the 365243 placeholder as missing
days_emp = f["DAYS_EMPLOYED"].where(f["DAYS_EMPLOYED"] != 365243)
my["APP_EMPLOYED_YEARS"] = -days_emp / 365.25

# Average and minimum of the three external scores (ignoring missing ones)
ext = f[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
my["APP_EXT_SOURCE_MEAN"] = ext.mean(axis=1)
my["APP_EXT_SOURCE_MIN"] = ext.min(axis=1)

my.head()
""")
code("""
# Load the pipeline's full feature table (114 features for every applicant) and compare.
features = pd.read_parquet(ART / "selection" / "features_all.parquet")
print("pipeline feature table:", features.shape)
for col in my.columns:
    a, b = my[col], features.loc[my.index, col]
    ok = np.allclose(a.fillna(-1), b.astype(float).fillna(-1), rtol=1e-5, atol=1e-6)
    print(f"{col:24s} {'✓ matches' if ok else '✗ differs'}")
""")
md("""
### 4b. Behavioural leg: days past due on previous Home Credit loans

Each row of `installments_payments` is one payment. A single instalment can be paid in several
parts, so we first collapse to **one row per scheduled instalment**: its due date and the date of its
*last* payment. Days past due (DPD) = last payment day − due day, floored at zero.

Let's do it for one customer who has paid late.
""")
code("""
cust = int(features.index[features["BEH_INS_DPD_MAX_12M"] > 30][0])   # first customer with 30+ DPD
raw_rows = inst[inst["SK_ID_CURR"] == cust].sort_values(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])
print("customer", cust, "has", len(raw_rows), "payment rows")
raw_rows.head(8)
""")
code("""
key = ["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]
per_instalment = raw_rows.groupby(key).agg(
    due=("DAYS_INSTALMENT", "min"),          # when it was due
    last_paid=("DAYS_ENTRY_PAYMENT", "max"),  # when the final part was paid
    amount_due=("AMT_INSTALMENT", "max"),
    amount_paid=("AMT_PAYMENT", "sum"),
)
per_instalment["dpd"] = (per_instalment["last_paid"] - per_instalment["due"]).clip(lower=0)
per_instalment.sort_values("dpd", ascending=False).head()
""")
code("""
last_12m = per_instalment[per_instalment["due"] >= -365]       # instalments due in the last year
ours = last_12m["dpd"].max()
check(f"customer {cust}: worst DPD in last 12 months", ours, features.loc[cust, "BEH_INS_DPD_MAX_12M"], 0)
check(f"customer {cust}: share of instalments paid late", (per_instalment["dpd"] > 0).mean(),
      features.loc[cust, "BEH_INS_LATE_SHARE"])
""")
md("""
### 4c. Bureau leg: credit with other lenders

`bureau_balance` has a monthly status for each credit with another lender: `0` on time, `1` = 1-30 days
late, `2` = 31-60, up to `5` = 120+ or written off; `C` = closed, `X` = unknown.
""")
code("""
bb = pd.read_parquet(CACHE / "bureau_balance.parquet")
cust_b = int(features.index[features["BUR_BB_WORST_BUCKET_12M"] >= 2][0])   # someone with 31+ DPD
credits = bureau.loc[bureau["SK_ID_CURR"] == cust_b, "SK_ID_BUREAU"]
history = bb[bb["SK_ID_BUREAU"].isin(credits)].sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"])
history.tail(8)
""")
code("""
bucket = history["STATUS"].astype(str).map({"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5})  # C/X -> missing
worst_12m = bucket[history["MONTHS_BALANCE"] >= -12].max()
check(f"customer {cust_b}: worst bureau bucket, last 12 months", worst_12m,
      features.loc[cust_b, "BUR_BB_WORST_BUCKET_12M"], 0)

b = bureau[bureau["SK_ID_CURR"] == cust_b]
check(f"customer {cust_b}: share of bureau credits still active",
      (b["CREDIT_ACTIVE"].astype(str) == "Active").mean(), features.loc[cust_b, "BUR_ACTIVE_SHARE"])
""")
md("""
### 4d. Previous-applications leg: refusals
""")
code("""
p = prev[prev["SK_ID_CURR"] == cust]
print(p["NAME_CONTRACT_STATUS"].astype(str).value_counts())
check(f"customer {cust}: share of previous applications refused",
      (p["NAME_CONTRACT_STATUS"].astype(str) == "Refused").mean(), features.loc[cust, "PRV_REFUSED_SHARE"])
""")
md("""
### 4e. All 114 features, and the feature dictionary

The pipeline builds the rest the same way. Every feature has a written **definition**, its source
table and its leg. Gender and marital status are marked **segment_only**: they're used only to check
fairness, never as model inputs.
""")
code("""
dictionary = pd.read_csv(ART / "selection" / "feature_dictionary.csv")
print(dictionary.groupby("leg").size().rename("features"))
dictionary.sample(8, random_state=1)[["name", "leg", "definition", "role"]]
""")

# =========================================================================================
md("""
## 5. Weight of evidence (WoE) and information value (IV)

A scorecard groups each feature into a few **bins** and gives each bin a number, its **weight of
evidence**:

$$\\text{WoE} = \\ln\\left(\\frac{\\%\\text{ of all goods in the bin}}{\\%\\text{ of all bads in the bin}}\\right)$$

A **high WoE means low risk**. The **information value** adds up how different goods and bads look
across the bins. Rule of thumb: below 0.02 is useless, 0.1 to 0.3 is medium, above 0.5 is
suspiciously strong.

We use the `optbinning` library on **train only**, with bins that move in one direction
(*monotonic*), so the result makes intuitive sense. The same binning later tells LightGBM which direction each feature may move PD.
""")
code("""
from optbinning import OptimalBinning

train_ids = pipe.loc[pipe["partition"] == "train", "SK_ID_CURR"]
x = features.loc[train_ids, "APP_EXT_SOURCE_3"].astype(float)
y = pipe.set_index("SK_ID_CURR").loc[train_ids, "TARGET"]

ob = OptimalBinning(name="APP_EXT_SOURCE_3", min_bin_size=0.05, max_n_bins=6,
                    monotonic_trend="auto_asc_desc")
ob.fit(x.to_numpy(), y.to_numpy())
table = ob.binning_table.build()
table[["Bin", "Count", "Count (%)", "Event rate", "WoE", "IV"]]
""")
code("""
# Work out one WoE by hand: the first bin.
row = table.iloc[0]
total_good = (y == 0).sum(); total_bad = (y == 1).sum()
pct_good = row["Non-event"] / total_good
pct_bad = row["Event"] / total_bad
print(f"bin {row['Bin']}: {pct_good:.2%} of goods, {pct_bad:.2%} of bads")
print("WoE by hand:", round(np.log(pct_good / pct_bad), 4), "| optbinning:", round(row["WoE"], 4))
print("IV of the whole feature:", round(table.loc["Totals", "IV"], 4))
""")
code("""
bins = table.drop(index="Totals")
bins = bins[bins["Count"] > 0]
fig, ax = plt.subplots()
ax.bar(range(len(bins)), bins["Event rate"], color=BLUE, width=0.6)
ax.set_xticks(range(len(bins)), [str(b) for b in bins["Bin"]], rotation=20)
ax.set_ylabel("default rate")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("APP_EXT_SOURCE_3: higher external score, lower default rate", loc="left", color=INK)
plt.tight_layout()
""")

# =========================================================================================
md("""
## 6. Feature selection: from 114 features to 13

The pipeline filters features **on train only**, in stages, and logs *every* exclusion with a reason:

1. protected characteristics (gender, marital status)
2. more than 95% missing
3. only one bin
4. IV below 0.02, or above 0.5 (too good to be true)
5. unstable between train and test (CSI)
6. highly correlated with a stronger feature
7. VIF above 5
8. more than 15 features
9. wrong sign or not significant
""")
code("""
excl = pd.read_csv(ART / "selection" / "exclusions.csv")
print(excl.groupby("stage").size().rename("features excluded"))
excl.groupby("stage").head(2)[["stage", "feature", "reason", "value"]]
""")
code("""
meta = json.loads((ART / "selection" / "selection.json").read_text())
selected = meta["selected"]
iv = pd.read_csv(ART / "selection" / "iv_table.csv").set_index("feature")["iv"]
pd.DataFrame({"IV": iv.loc[selected].round(4),
              "leg": dictionary.set_index("name").loc[selected, "leg"]})
""")

# =========================================================================================
md("""
## 7. The benchmark: a scorecard (logistic regression on WoE)

The project has two models. The **champion**, LightGBM (Part 8), makes the decisions. The
**scorecard** is the *benchmark and fallback*: simpler and fully transparent, kept to show what
the champion's extra complexity buys and to take over if the champion is ever suspended.

We replace each of the 13 features by its bin's WoE, then fit a **logistic regression**. That's a
formula which turns a weighted sum of the WoE values into a probability of default (PD) between 0 and 1.
Because high WoE means low risk, every coefficient should be **negative**.
""")
code('''
from optbinning import BinningProcess
import statsmodels.api as sm

roles = summary["model"]["roles"]                       # which model is champion, which is challenger
metrics = summary["validation"]["metrics"]
role_of = {model_name: role for role, model_name in roles.items()}   # e.g. {"lightgbm": "champion"}
print("roles in the pipeline run:", roles)

categorical = [c for c in selected if c in meta["categorical"]]


def prepare(X):
    """optbinning wants plain text for categories and decimals for numbers."""
    X = X.copy()
    for c in X.columns:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan) if c in categorical else X[c].astype(float)
    return X


bp = BinningProcess(selected, categorical_variables=categorical, min_bin_size=0.05, max_n_bins=6,
                    binning_fit_params={c: {"monotonic_trend": "auto_asc_desc"}
                                        for c in selected if c not in categorical})
bp.fit(prepare(features.loc[train_ids, selected]), y.to_numpy())

# WoE for every applicant (train, test and OOT) using bins learned on TRAIN only.
lab_ids = pipe["SK_ID_CURR"]
W = bp.transform(prepare(features.loc[lab_ids, selected]), metric="woe",
                 metric_missing="empirical", metric_special="empirical")
W.index = lab_ids
W.head()
''')
code("""
logit = sm.Logit(y.to_numpy(), sm.add_constant(W.loc[train_ids].to_numpy())).fit(disp=0)
coefs = pd.Series(logit.params, index=["intercept"] + selected)
pd.DataFrame({"coefficient": coefs.round(4), "p-value": pd.Series(logit.pvalues, index=coefs.index).round(4)})
""")
code("""
from sklearn.metrics import roc_auc_score

oot_ids = pipe.loc[pipe["partition"] == "oot", "SK_ID_CURR"]
y_all = pipe.set_index("SK_ID_CURR")["TARGET"]
pd_sc_oot = 1 / (1 + np.exp(-(coefs["intercept"] + W.loc[oot_ids, selected] @ coefs[selected])))

auc = roc_auc_score(y_all.loc[oot_ids], pd_sc_oot)
gini = 2 * auc - 1           # Gini is just a rescaled AUC
print(f"scorecard OOT AUC {auc:.4f}  ->  Gini {gini:.4f}")
check("scorecard OOT Gini", gini, metrics[role_of["scorecard"]]["performance"]["oot"]["gini"])
""")
md("""
### Points: why scorecards are easy to explain

Banks show a scorecard as **points**: add up the points for each characteristic, and a higher
score means lower risk. The scaling here is "20 points to double the odds" (PDO 20), with 600
points at 50:1 odds:

* `factor = 20 / ln(2)` and `offset = 600 − factor × ln(50)`
* `points for a bin = −factor × coefficient × WoE + (offset − factor × intercept) / 13`
""")
code("""
factor = 20 / np.log(2)
offset = 600 - factor * np.log(50)
n = len(selected)

# Score ONE applicant by hand: look up their WoE for each characteristic and convert it to points.
applicant = int(oot_ids.iloc[0])
w = W.loc[applicant, selected]
points = -factor * coefs[selected] * w + (offset - factor * coefs["intercept"]) / n
card = pd.DataFrame({"value": features.loc[applicant, selected], "WoE": w.round(3), "points": points.round(1)})
print("applicant", applicant)
card
""")
code("""
score = points.sum()
pd_applicant = 1 / (1 + np.exp(-(coefs["intercept"] + (w * coefs[selected]).sum())))
scored = pd.read_parquet(ART / "model" / "scored.parquet")
print(f"total score {score:.1f}  ->  scorecard PD {pd_applicant:.2%}")
check("this applicant's score", score, scored.loc[applicant, "score"], 2)
check("this applicant's scorecard PD", pd_applicant, scored.loc[applicant, "pd_scorecard"])
""")

# =========================================================================================
md("""
## 8. The champion: monotone LightGBM, explained with SHAP

LightGBM builds hundreds of small decision trees, each one fixing the errors of the previous ones.
It uses all 112 candidate features, raw rather than WoE, and finds interactions a scorecard can't,
so it ranks applicants better. Two things make it acceptable as the champion:

1. **Monotonic constraints.** Each feature may push PD in one direction only, so the model can never
   contradict credit logic (for example "more days past due lowers PD").
2. **SHAP explanations.** Every individual prediction is split exactly into "how much each feature
   pushed this applicant's PD up or down". That gives reason codes, the equivalent of the scorecard's points.

### 8a. Which direction may each feature move PD?

We read the direction from each feature's train-only binning: if the default rate rises bin by bin,
the feature may only **raise** PD (+1); if it falls, only **lower** it (−1); categories are left free (0).
""")
code("""
import lightgbm as lgb
from pickle import load

with open(ART / "selection" / "binning_process.pkl", "rb") as fh:
    bp_all = load(fh)                              # the pipeline's train-only binning of every candidate
live = list(bp_all.variable_names)
cands = [c for c in meta["candidates"] if c in live]
cat_all = [c for c in live if c in meta["categorical"]]


def direction(feature):
    if feature in cat_all:
        return 0
    t = bp_all.get_binned_variable(feature).binning_table.build().drop(index="Totals")
    t = t[(t["Count"] > 0) & ~t["Bin"].astype(str).isin(["Special", "Missing"])]
    rates = t["Event rate"].to_numpy()
    if len(rates) < 2:
        return 0
    if np.all(np.diff(rates) >= 0):
        return 1
    if np.all(np.diff(rates) <= 0):
        return -1
    return 0


signs = {f: direction(f) for f in cands}
print(pd.Series(signs).value_counts().rename({1: "may only raise PD", -1: "may only lower PD", 0: "free"}))
pd.Series({f: signs[f] for f in ["BEH_INS_DPD_MAX_12M", "APP_EXT_SOURCE_MEAN", "APP_AGE_YEARS",
                                 "APP_ANNUITY_TO_INCOME", "PRV_REFUSED_SHARE"]}).rename("direction").to_frame()
""")
md("""
### 8b. Fit the constrained model

The settings were tuned **offline, on train only**, and kept lean: depth 5, 15 leaves, at least 1,000
applicants per leaf. Early stopping uses a 10% slice of train to choose the number of trees.
""")
code("""
def prepare_lgb(X):
    X = X.copy()
    for c in X.columns:
        X[c] = X[c].astype("category") if c in cat_all else X[c].astype("float32")
    return X


params = dict(max_depth=5, num_leaves=15, min_child_samples=1000, n_estimators=1000,
              learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
              reg_lambda=10.0, random_state=SEED, deterministic=True, force_row_wise=True,
              n_jobs=4, verbose=-1,
              monotone_constraints=[signs[c] for c in cands],     # the directions from 8a
              monotone_constraints_method="intermediate")
Xtr = prepare_lgb(features.loc[train_ids, cands])
Xa, Xv, ya, yv = train_test_split(Xtr, y, test_size=0.10, stratify=y, random_state=SEED)
probe = lgb.LGBMClassifier(**params).fit(Xa, ya, eval_X=Xv, eval_y=yv, eval_metric="auc",
                                         callbacks=[lgb.early_stopping(50, verbose=False)])
gbm = lgb.LGBMClassifier(**{**params, "n_estimators": probe.best_iteration_}).fit(Xtr, y)
print("trees:", probe.best_iteration_)

X_oot = prepare_lgb(features.loc[oot_ids, cands])
pd_gbm = pd.Series(gbm.predict_proba(X_oot)[:, 1], index=oot_ids)
gini_gbm = 2 * roc_auc_score(y_all.loc[oot_ids], pd_gbm) - 1
print(f"champion OOT Gini {gini_gbm:.4f} vs scorecard {gini:.4f}")
check("champion (LightGBM) OOT Gini", gini_gbm, metrics[role_of["lightgbm"]]["performance"]["oot"]["gini"])
""")
code("""
# What the extra Gini is worth: bad rate among approved applicants at the same approval rate.
rows = []
for approve in (0.6, 0.7, 0.8, 0.9):
    r = {"approval rate": approve}
    for name, p in (("scorecard", pd_sc_oot), ("LightGBM", pd_gbm)):
        approved = p <= p.quantile(approve)
        r[f"bad rate, {name}"] = y_all.loc[oot_ids][approved].mean()
    rows.append(r)
pd.DataFrame(rows).set_index("approval rate").style.format("{:.2%}")
""")
md("""
### 8c. The master scale: from PD to a rating grade

The champion's PDs are grouped into 10 grades (grade 1 = best). If the model works, the
observed default rate should rise grade by grade.
""")
code("""
bounds = [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.25, 1.0]   # upper PD of each grade
grade = np.searchsorted(bounds, pd_gbm, side="left") + 1
by_grade = pd.DataFrame({"grade": grade, "pd": pd_gbm.to_numpy(), "default": y_all.loc[oot_ids].to_numpy()}) \\
             .groupby("grade").agg(accounts=("pd", "size"), mean_pd=("pd", "mean"), default_rate=("default", "mean"))
by_grade.style.format({"accounts": "{:,}", "mean_pd": "{:.2%}", "default_rate": "{:.2%}"})
""")
code("""
fig, ax = plt.subplots()
xs = np.arange(len(by_grade))
ax.bar(xs - 0.18, by_grade["mean_pd"], width=0.34, color=BLUE, label="predicted PD")
ax.bar(xs + 0.18, by_grade["default_rate"], width=0.34, color=ORANGE, label="observed default rate")
ax.set_xticks(xs, by_grade.index)
ax.set_xlabel("grade (OOT, champion)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(frameon=False)
ax.set_title("Predicted and observed default rate rise together", loc="left", color=INK)
plt.tight_layout()
""")
md("""
### 8d. SHAP: explaining one decision exactly

For one applicant, SHAP gives a number per feature: how much it pushed the prediction up (+) or
down (−), in *log-odds*. The baseline ("bias") plus all the contributions adds up **exactly** to
the model's score for that applicant.
""")
code("""
risky = int(pd_gbm.sort_values().index[-50])   # one of the riskiest OOT applicants
contrib = gbm.booster_.predict(X_oot.loc[[risky]], pred_contrib=True)[0]     # LightGBM's built-in exact TreeSHAP
shap_one = pd.Series(contrib[:-1], index=cands)
bias = contrib[-1]
log_odds = bias + shap_one.sum()
print(f"applicant {risky}: PD {pd_gbm[risky]:.1%}")
print(f"bias {bias:.3f} + sum of contributions {shap_one.sum():.3f} = {log_odds:.3f} log-odds"
      f" -> PD {1 / (1 + np.exp(-log_odds)):.1%}")
shap_one.sort_values(ascending=False).head(8).rename("pushes PD up by (log-odds)").to_frame().round(3)
""")
md("""
### 8e. Reason codes, grouped into families

Several features measure the same thing; for example the three external scores and their mean and
minimum. Listing all of them as separate reasons would be confusing on a decline letter, so the
pipeline adds contributions up within **reason families** and reports the top families.
""")
code("""
import re

families = json.loads((RUN / "config_snapshot.json").read_text())["base"]["explain"]["families"]


def family_of(feature):
    for fam in families:
        if any(re.search(p, feature) for p in fam["patterns"]):
            return fam["label"]


by_family = shap_one.groupby(shap_one.index.map(family_of)).sum().sort_values(ascending=False)
my_reasons = by_family[by_family > 0].head(4)
my_reasons.rename("pushes PD up by (log-odds)").to_frame().round(3)
""")
code("""
pipeline_reasons = pd.read_parquet(ART / "explain" / "reason_families_oot.parquet")
theirs = pipeline_reasons[pipeline_reasons["SK_ID_CURR"] == risky].sort_values("rank")
print("pipeline's reason families for this applicant:", list(theirs["reason"]))
print("same reasons in the same order:", "✓" if list(theirs["reason"]) == list(my_reasons.index) else "✗")
""")
code("""
# Across all OOT applicants: which families carry the most weight?
fam_share = pd.Series(summary["explain"]["family_shares"])
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.barh(fam_share.index[::-1], fam_share.values[::-1], color=BLUE, height=0.6)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
ax.set_title("Share of the champion's explanatory weight, by reason family", loc="left", color=INK)
plt.tight_layout()
""")

# =========================================================================================
md("""
## 9. Validation: is the champion any good?

A second line of defence checks the model independently. The key measures:

* **AUC / Gini** (*discrimination*): does the model rank bad applicants above good ones? AUC 0.5 is
  random and 1.0 is perfect. Gini = 2 × AUC − 1.
* **KS**: the biggest gap between the cumulative score distributions of goods and bads.
* **Calibration**: are the PDs the right *size*? Within each group, average PD should match the
  observed default rate.
* **PSI** (*stability*): has the population changed? Below 0.10 is stable; 0.25 and above is a red flag.
""")
code("""
from sklearn.metrics import roc_curve

rows = []
for part in ("train", "test", "oot"):
    ids_p = pipe.loc[pipe["partition"] == part, "SK_ID_CURR"]
    p = scored.loc[ids_p, "pd_champion"]
    fpr, tpr, _ = roc_curve(y_all.loc[ids_p], p)
    auc_p = roc_auc_score(y_all.loc[ids_p], p)
    rows.append({"partition": part, "AUC": auc_p, "Gini": 2 * auc_p - 1, "KS": (tpr - fpr).max(),
                 "mean PD": p.mean(), "default rate": y_all.loc[ids_p].mean()})
perf = pd.DataFrame(rows).set_index("partition")
perf.style.format({"AUC": "{:.4f}", "Gini": "{:.4f}", "KS": "{:.4f}", "mean PD": "{:.2%}", "default rate": "{:.2%}"})
""")
md("""
**What to notice.** Train Gini is somewhat higher than OOT: a tree model learns its training data a
little too well, but the gap is within the tolerance the validator allows. The mean PD matches the
default rate in every partition.
""")
code("""
# Calibration by PD decile on OOT: sort applicants by PD into 10 equal groups.
oot_df = pd.DataFrame({"pd": scored.loc[oot_ids, "pd_champion"], "y": y_all.loc[oot_ids]})
oot_df["decile"] = pd.qcut(oot_df["pd"].rank(method="first"), 10, labels=False) + 1
cal = oot_df.groupby("decile").agg(mean_pd=("pd", "mean"), default_rate=("y", "mean"))

fig, ax = plt.subplots(figsize=(4.6, 4.2))
lim = cal.max().max() * 1.1
ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
ax.plot(cal["mean_pd"], cal["default_rate"], marker="o", markersize=7, linewidth=2, color=BLUE,
        label="champion (OOT deciles)")
ax.set_xlabel("average predicted PD"); ax.set_ylabel("observed default rate")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(frameon=False, loc="upper left")
ax.set_title("Calibration: dots close to the diagonal", loc="left", color=INK)
plt.tight_layout()
""")
code("""
# PSI by hand: compare the champion's PD distribution in train and OOT over 10 bands from train.
train_pd = scored.loc[train_ids, "pd_champion"]
edges = np.unique(np.quantile(train_pd, np.linspace(0, 1, 11))); edges[0], edges[-1] = -np.inf, np.inf
share_train = pd.cut(train_pd, edges, labels=False).value_counts(normalize=True, sort=False)
share_oot = pd.cut(scored.loc[oot_ids, "pd_champion"], edges, labels=False).value_counts(normalize=True, sort=False)
psi = ((share_oot - share_train) * np.log(share_oot / share_train)).sum()
print(f"PSI train vs OOT: {psi:.4f}  (near zero, because OOT is a random holdout)")
check("PSI", psi, metrics["stability"]["score_psi"]["train_vs_oot"])
""")
code("""
# Grade-level calibration test (binomial): if a grade's true PD were its average PD, how surprising
# would the observed number of defaults be? A small p-value means the model under-predicts risk.
from scipy import stats

for g in (3, 8):
    row = by_grade.loc[g]
    defaults = round(row["default_rate"] * row["accounts"])
    p_value = stats.binom.sf(defaults - 1, int(row["accounts"]), row["mean_pd"])
    verdict = "red" if p_value <= 0.01 else "amber" if p_value <= 0.05 else "green"
    print(f"grade {g}: {int(row['accounts']):,} accounts, {defaults} defaults vs mean PD {row['mean_pd']:.2%}"
          f" -> p-value {p_value:.4f} ({verdict})")
""")
md("""
**What to notice.** The champion's calibration is good overall but not perfect: in some grades it
slightly *under*-predicts risk. The validator records this as a medium finding to monitor, and a
grade-level recalibration is a possible fix. This is the honest cost of the more complex model.

### The validator's findings

The pipeline's validator turns these measures into **findings** with a severity. Any **red**
finding would stop the run. Here there are none; the medium ones are limitations to document and monitor.
""")
code("""
findings = pd.DataFrame(json.loads((ART / "validation" / "findings.json").read_text()))
findings[["id", "severity", "area", "title"]]
""")

# =========================================================================================
md("""
## 10. Expected credit loss (ECL)

Under the RBI ECL Directions (RBI/DOR/2026-27/398, effective April 2027), banks provide for losses
they **expect**, not only those already incurred:

$$\\text{ECL} = \\sum_{\\text{months}} \\text{PD in that month} \\times \\text{LGD} \\times \\text{EAD in that month} \\times \\text{discount factor}$$

* **Stage 1** (performing): 12 months of expected loss.
* **Stage 2** (*significant increase in credit risk*, for example more than 30 days past due):
  expected loss over the **whole life** of the loan.
* **Stage 3** (credit-impaired, more than 90 days past due): PD = 100%.

This is an **illustrative simulation**: Home Credit isn't a live loan book, so the OOT applicants are
treated as a book of new loans.

### 10a. One loan, worked by hand

Loan **100257** is in **Stage 2**, because the borrower was more than 30 days late on a previous Home Credit loan.
""")
code("""
acct = pd.read_parquet(ART / "ecl" / "ecl_accounts.parquet")
loan_id = 100257
loan = acct.loc[loan_id]
P = float(app.set_index("SK_ID_CURR").loc[loan_id, "AMT_CREDIT"])   # loan amount
A = float(app.set_index("SK_ID_CURR").loc[loan_id, "AMT_ANNUITY"])  # monthly instalment
print(f"loan amount {P:,.0f} | monthly instalment {A:,.0f} | stage {loan['stage']} ({loan['triggers']})")
print(f"12-month PD from the champion (LightGBM): {loan['pd12']:.2%}")
""")
md("""
**Step 1: the interest rate.** We know the loan amount and the instalment, but not the term or the rate:
one equation, two unknowns. So we take the rate implied by previous Home Credit loans, where amount,
instalment *and* term are all known. The annuity formula is `A = P × r / (1 − (1 + r)^−n)`.
We solve it for `r` on each previous approved loan of a **train** customer, then take the median.
""")
code('''
def annuity(P, r, n):
    return P * r / (1 - (1 + r) ** -n)


def solve_rate(P, A, n, lo=0.0, hi=0.5):
    """Find the monthly rate r with annuity(P, r, n) == A, by halving the search interval."""
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if annuity(P, mid, n) < A else (lo, mid)
    return (lo + hi) / 2


prev_train = prev[prev["SK_ID_CURR"].isin(train_ids)]
ok = ((prev_train["NAME_CONTRACT_STATUS"].astype(str) == "Approved") & (prev_train["CNT_PAYMENT"] > 0)
      & (prev_train["AMT_ANNUITY"] > 0) & (prev_train["AMT_CREDIT"] > 0)
      & (prev_train["NAME_CONTRACT_TYPE"].astype(str) == "Cash loans"))
cash = prev_train[ok]
# Vectorised version of the same halving search, for all cash loans at once
lo, hi = np.zeros(len(cash)), np.full(len(cash), 0.5)
Pv, Av, nv = cash["AMT_CREDIT"].to_numpy(float), cash["AMT_ANNUITY"].to_numpy(float), cash["CNT_PAYMENT"].to_numpy(float)
for _ in range(80):
    mid = (lo + hi) / 2
    low = annuity(Pv, mid, nv) < Av
    lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
rates = np.where(Av * nv <= Pv, 0.0, (lo + hi) / 2)
r = float(np.median(rates))
print(f"median monthly rate on {len(cash):,} prior cash loans: {r:.2%} (about {(1 + r) ** 12 - 1:.0%} a year)")
check("monthly rate", r, loan["rate_monthly"], 6)
''')
code("""
# Step 2: the term, from the annuity formula solved for n.
n_months = int(np.ceil(-np.log(1 - r * P / A) / np.log(1 + r)))
check("term in months", n_months, loan["term_months"], 0)

# Step 3: the repayment schedule. Balance at the START of each month = exposure if default happens then.
balance, rows = P, []
for month in range(1, n_months + 1):
    rows.append({"month": month, "exposure (EAD)": balance})
    balance = max(balance * (1 + r) - A, 0)     # add a month's interest, subtract the instalment
schedule = pd.DataFrame(rows).set_index("month")
pd.concat([schedule.head(4), schedule.tail(3)]).style.format("{:,.0f}")
""")
code("""
# Step 4: PD month by month. A constant monthly hazard h gives the same 12-month PD: 1-(1-h)^12 = PD12.
pd12 = max(loan["pd12"], 0.0003)                    # RBI floor on the 12-month PD: 0.03%
h = 1 - (1 - pd12) ** (1 / 12)
survive = (1 - h) ** (schedule.index - 1)           # chance of reaching month t without defaulting
schedule["PD this month"] = survive * h
schedule["discount factor"] = (1 + r) ** -schedule.index
print(f"monthly hazard {h:.3%}; lifetime PD over {n_months} months: {schedule['PD this month'].sum():.2%}")
schedule.head(4)
""")
code("""
# Step 5: ECL. Stage 2 = lifetime. LGD = 70% (RBI backstop for unsecured loans).
# Three illustrative scenarios scale the PD; the result is their probability-weighted average.
LGD = 0.70
scenarios = {"base": (0.60, 1.00), "upside": (0.15, 0.85), "downside": (0.25, 1.35)}   # (weight, PD scalar)
ecl = {}
for name, (weight, scalar) in scenarios.items():
    p12 = min(max(loan["pd12"] * scalar, 0.0003), 0.9999)
    hz = 1 - (1 - p12) ** (1 / 12)
    pd_t = (1 - hz) ** (schedule.index - 1) * hz
    ecl[name] = (pd_t * LGD * schedule["exposure (EAD)"] * schedule["discount factor"]).sum()
weighted = sum(scenarios[k][0] * v for k, v in ecl.items())
print({k: round(v) for k, v in ecl.items()}, "-> weighted", round(weighted))
check("model ECL for this loan", weighted, loan["ecl_model"], 2)
""")
code("""
# Step 6: the RBI minimum provision. For unsecured retail in Stage 2 it's 5% of exposure.
floor = 0.05 * P
final = max(weighted, floor)
print(f"model ECL {weighted:,.0f} vs floor {floor:,.0f} -> final ECL {final:,.0f}"
      f" ({'floor binds' if floor > weighted else 'model ECL used'})")
check("final ECL for this loan", final, loan["ecl_final"], 2)
""")
md("""
### 10b. The whole book

Stage rules applied to all 76,878 OOT loans, by hand with pandas conditions:
""")
code("""
book = features.loc[oot_ids].join(scored.loc[oot_ids])
hc_dpd = book[["BEH_INS_DPD_MAX_12M", "BEH_POS_DPD_MAX_12M"]].max(axis=1).fillna(0)
stage3 = hc_dpd > 90
sicr = ((hc_dpd > 30)
        | (book["BUR_BB_WORST_BUCKET_12M"].fillna(0) >= 2) | (book["BUR_DAYS_OVERDUE_MAX"].fillna(0) > 30)
        | ((book["grade_champion"] - book["grade_origination_proxy"]) >= 3)
        | ((book["pd_champion"] >= 2 * book["pd_origination_proxy"]) & (book["pd_champion"] >= 0.05)))
stage = np.where(stage3, 3, np.where(sicr, 2, 1))
ours = pd.Series(stage).value_counts().sort_index()
print(ours.rename("loans"))
print("same stage counts as the pipeline:",
      "✓" if ours.to_dict() == {int(k): v for k, v in summary["ecl"]["stage_counts"].items()} else "✗")
""")
code("""
by_stage = pd.read_csv(ART / "ecl" / "ecl_by_stage.csv", index_col=0)
by_stage[["accounts", "ead", "ecl_final", "coverage_final", "realised_default_rate"]].style.format(
    {"accounts": "{:,.0f}", "ead": "{:,.0f}", "ecl_final": "{:,.0f}", "coverage_final": "{:.2%}",
     "realised_default_rate": "{:.2%}"})
""")
code("""
s = by_stage.drop(index="total")
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.bar(s.index.str.replace("_", " "), s["coverage_final"], color=BLUE, width=0.5)
for i, v in enumerate(s["coverage_final"]):
    ax.text(i, v, f"{v:.1%}", ha="center", va="bottom", color=INK)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylabel("ECL as % of exposure")
ax.set_title("Coverage rises with stage", loc="left", color=INK)
plt.tight_layout()
""")

# =========================================================================================
md("""
## 11. The "agents" around all of this

Everything above is **deterministic**: same data and settings, same numbers. The project wraps it in a
**LangGraph** workflow, a flowchart that a program walks through step by step. Each box is a "node"
(we call them agents), and each node declares exactly which information it reads and writes.

* Four **leg** nodes build features *in parallel*.
* An **explain** node computes the SHAP reason codes for the champion.
* **Gates** stop the run if data quality or validation fails.
* Only **two** nodes use an LLM (a large language model such as Gemini):
  * the **critic**, which challenges the results like a sceptical validator and may only request
    pre-approved calculations;
  * the **documentation** agent, which writes the reports.
* A **grounding check** then verifies every number in those reports.

**LLMs never fit, tune or change the model.**

Here's the order the nodes ran in, from the run's audit trail:
""")
code("""
audit = pd.DataFrame([json.loads(line) for line in (RUN / "audit.jsonl").read_text().splitlines()])
audit[audit["event"] == "node_end"][["node", "seconds"]].reset_index(drop=True)
""")
code("""
# What the critic said, and the extra analyses it asked for
for review in summary["critic_reviews"]:
    print(f"--- review {review['iteration']} ---")
    for c in review.get("challenges", []):
        print(f"[{c['severity']}] {c['title']}: {c['detail']}")
    print("conclusion:", review["conclusion"], "\\n")
pd.DataFrame(summary["extra_analyses"])[["iteration", "kind", "params", "rationale"]]
""")
code("""
# The grounding check: every number in the generated documents was matched to a real result.
print((RUN / "documents" / "grounding_report.md").read_text())
""")
code("""
# The first lines of one generated document
print("\\n".join((RUN / "documents" / "independent_validation.md").read_text().splitlines()[:16]))
""")

# =========================================================================================
md("""
## 12. Glossary

| Term | Meaning |
|---|---|
| **PD** | Probability of default: the chance a borrower fails to pay, here over about 12 months. |
| **AUC** | Area under the ROC curve: the probability the model ranks a random bad above a random good. 0.5 is random, 1 is perfect. |
| **Gini** | 2 × AUC − 1. The usual credit-risk way to quote AUC. |
| **KS** | Kolmogorov-Smirnov: the largest gap between goods' and bads' cumulative score distributions. |
| **WoE** | Weight of evidence of a bin: ln(% of goods ÷ % of bads). High = low risk. |
| **IV** | Information value: how much a feature separates goods from bads. |
| **PSI / CSI** | Population / characteristic stability index: how much a distribution has shifted. |
| **Calibration** | Whether PDs are the right size: predicted PD vs observed default rate. |
| **Champion / challenger** | The model in use, and the alternative it's benchmarked against. |
| **OOT** | Out-of-time holdout. Here it's a random holdout ("pseudo-OOT"), because the data has no dates. |
| **ECL** | Expected credit loss: PD × LGD × EAD, discounted, summed over the horizon. |
| **LGD** | Loss given default: share of exposure lost if the borrower defaults (70% backstop here). |
| **EAD** | Exposure at default: what's owed at the time of default. |
| **EIR** | Effective interest rate, used to discount future losses. |
| **SICR** | Significant increase in credit risk: moves a loan to Stage 2 (lifetime ECL). |
| **DPD** | Days past due. |
| **Leakage** | Accidentally letting information from the test data into model building. Prevented by fitting on train only. |
""")


def build(run: bool) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if run:
        from nbclient import NotebookClient

        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(LOCAL.parent)}}).execute()  # fmt: skip
        nbf.write(nb, LOCAL)
    clean = copy.deepcopy(nb)
    for c in clean.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    nbf.write(clean, CLEAN)
    print(
        f"wrote {CLEAN.relative_to(ROOT)} (outputs cleared)"
        + (f" and {LOCAL.relative_to(ROOT)} (executed)" if run else "")
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true")
    build(not ap.parse_args().no_run)
