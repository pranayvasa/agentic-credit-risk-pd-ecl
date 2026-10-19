"""Phase 6: facts payload guard, whitelisted analyses, documentation and the grounding check."""

from __future__ import annotations

from pathlib import Path

import pytest

from acr.core import extras as X
from acr.core.docs import fill, load_spec
from acr.core.facts import assert_summary_only
from acr.core.grounding import check
from acr.graph import build_graph
from acr.nodes.grounding import grounding_check

# --- grounding unit tests -----------------------------------------------------------------

REF = [0.500847, 0.750423, 0.044822, 2_059_450_000.0, 76878, 56.25]


@pytest.mark.parametrize("text", [
    "OOT Gini 0.5008", "AUC 0.7504", "AUC 0.75", "coverage 4.48%", "coverage 4.5%",
    "ECL 2,059.4 million", "ECL 2.06 billion", "76,878 loans", "train share 56.25%",
])  # fmt: skip
def test_true_numbers_verify(text):
    rep = check(text, REF)
    assert rep.counts["unverified"] == 0 and rep.counts["verified"] == 1


@pytest.mark.parametrize(
    "text", ["Gini 0.6123", "coverage 7.31%", "ECL 3,100.0 million", "77,001 loans"]
)
def test_invented_numbers_are_flagged(text):
    assert check(text, REF).counts["unverified"] == 1


def test_structural_numbers_are_not_data():
    text = ("Under SR 11-7 and RBI/DOR/2026-27/398, para 82(3)(x) and paragraph 97, dated "
            "27 April 2026 and effective 1 April 2027, with a 95% CI, Stage 2 and grade 10.\n"
            "1. First point\n## 3. Heading")  # fmt: skip
    rep = check(text, [])
    assert rep.counts["unverified"] == 0


def test_identifiers_with_digits_are_ignored():
    rep = check("Run 20261019T034344Z_full_5544798d5000, hash e20fc11ff85c, finding V13, D22.", [])
    assert rep.counts["unverified"] == 0


# --- payload guard ------------------------------------------------------------------------


def test_payload_guard_blocks_loan_level_data():
    with pytest.raises(ValueError, match="identifier"):
        assert_summary_only({"rows": [{"SK_ID_CURR": 1, "pd": 0.1}]})
    with pytest.raises(ValueError, match="summaries only"):
        assert_summary_only({"pds": list(range(500))})
    assert_summary_only({"gini": 0.5, "grades": list(range(10))})


# --- documentation spec -------------------------------------------------------------------


def test_template_fill_and_missing_fact():
    facts = {"a": {"b": 0.123456, "n": 76878, "m": 2_059_460_000.0}}
    assert fill("{a.b:.4f} {a.n:,} {a.m:m1}", facts) == "0.1235 76,878 2,059.5 million"
    with pytest.raises(KeyError, match="fact not found"):
        fill("{a.zzz}", facts)


def test_spec_has_four_documents_with_sections():
    spec = load_spec()["documents"]
    assert set(spec) == {"model_development", "independent_validation", "ongoing_monitoring",
                         "ecl_methodology"}  # fmt: skip
    for d in spec.values():
        assert d["sections"] and all(
            {"id", "title", "guidance", "template"} <= set(s) for s in d["sections"]
        )


# --- end to end ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from acr.config import load_config
    from synthetic import make_tables, write_raw

    tmp = tmp_path_factory.mktemp("docs")
    raw = write_raw(tmp / "raw", make_tables(n=4000))
    cfg = load_config().model_dump(mode="json")
    cfg["base"]["paths"].update(raw_dir=str(raw), interim_dir=str(tmp / "interim"),
                                outputs_dir=str(tmp / "out"))  # fmt: skip
    return build_graph().invoke({"config": cfg, "sample_frac": 0.5, "llm_mode": "dry-run"},
                                {"recursion_limit": 60})  # fmt: skip


def test_four_documents_written_and_grounded(run):
    docs = {k: v for k, v in run["documents"].items() if not k.startswith("_")}
    assert set(docs) == {"model_development", "independent_validation", "ongoing_monitoring",
                         "ecl_methodology"}  # fmt: skip
    for p in docs.values():
        text = Path(p).read_text()
        assert text.startswith("# ") and "## 1." in text
    assert run["grounding"]["status"] == "pass", run["grounding"]["unverified"]
    assert run["grounding"]["totals"]["verified"] > 100


def test_grounding_flags_a_number_injected_into_a_document(run, tmp_path):
    import shutil

    docs = {k: v for k, v in run["documents"].items() if not k.startswith("_")}
    run_dir = tmp_path / "run"
    (run_dir / "documents").mkdir(parents=True)
    copied = {}
    for name, p in docs.items():
        copied[name] = str(shutil.copy(p, run_dir / "documents" / Path(p).name))
    with open(copied["model_development"], "a") as f:
        f.write("\nThe champion's OOT Gini is 0.9137 and coverage is 12.84%.\n")
    state = {**run, "run_dir": str(run_dir), "documents": {**run["documents"], **copied}}
    out = grounding_check(state)["grounding"]
    assert out["status"] == "flagged"
    assert {u["number"] for u in out["unverified"]} == {"0.9137", "12.84%"}


def test_critic_loop_ran_whitelisted_analyses(run):
    kinds = [e["kind"] for e in run["extra_analyses"]]
    assert kinds and set(kinds) <= set(X.ANALYSES)
    assert all("result" in e for e in run["extra_analyses"])
    assert len(run["critic_reviews"]) == 2  # review, then re-review of the results
    assert len([c for c in run["llm_calls"] if c["node"] == "document"]) == sum(
        len(d["sections"]) for d in load_spec()["documents"].values()
    )


def test_analysis_parameters_are_validated(run):
    import json

    import pandas as pd

    meta = json.loads(Path(run["selection"]["paths"]["meta"]).read_text())
    x = X.Inputs(scored=pd.read_parquet(run["model"]["paths"]["scored"]),
                 features=pd.read_parquet(run["selection"]["paths"]["features_all"]),
                 ecl_accounts=pd.read_parquet(run["ecl"]["paths"]["accounts"]),
                 selected=meta["selected"], floors={}, pd_floor=0.0003, lgd_base=0.70)  # fmt: skip
    with pytest.raises(X.BadRequest, match="unknown parameters"):
        X.run("segment_cut", {"segment": "age_band", "refit": True}, x)
    with pytest.raises(X.BadRequest):
        X.run("lgd_sensitivity", {"values": "1.5"}, x)
    with pytest.raises(X.BadRequest):
        X.run("segment_cut", {"segment": "postcode"}, x)
    t = X.run("lgd_sensitivity", {"values": "0.35,0.70"}, x)["table"]
    assert t[0]["ecl_model"] == pytest.approx(t[1]["ecl_model"] / 2)  # linear in LGD


def test_dry_run_is_deterministic(init_state):
    runs = [
        build_graph().invoke(init_state(sample_frac=0.5), {"recursion_limit": 60}) for _ in range(2)
    ]
    assert [r["conclusion"] for r in runs[0]["critic_reviews"]] == [
        r["conclusion"] for r in runs[1]["critic_reviews"]]  # fmt: skip
    assert all(r["grounding"]["status"] == "pass" for r in runs)
