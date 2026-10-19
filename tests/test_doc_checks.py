"""Hardened document checks: label-aware numbers, contradictions against findings, v2 prompts."""

from __future__ import annotations

import pytest

from acr.core import consistency
from acr.core.grounding import check_labelled

LABELLED = [
    ("stability.external.challenger_pd_psi", 0.00175),
    ("stability.external.score_psi", 0.007244),
    ("stability.stress_psi.under30_x3", 0.0152),
    ("champion.performance.oot.gini", 0.560243),
    ("champion.performance.oot.auc", 0.780121),
    ("challenger.performance.oot.gini", 0.500847),
    ("comparison.champion_lead", 0.059396),
]


def _status(text):
    return [t.status for t in check_labelled(text, LABELLED).tokens]


def test_number_attached_to_wrong_label_is_misattributed():
    # 0.015 exists (as the stress PSI 0.0152) but the sentence calls it the challenger PD PSI.
    assert _status("Score PSI 0.0072, with a challenger PD PSI of 0.015.") == [
        "verified",
        "misattributed",
    ]


def test_correct_label_passes_and_unlabelled_numbers_fall_back_to_existence():
    assert _status("The challenger PD PSI is 0.0018.") == ["verified"]
    assert _status("Stress PSI values are 0.0152, then larger.") == ["verified"]
    assert _status("The champion leads by 0.0594 Gini.") == ["verified"]


def test_multiple_measures_and_asides_do_not_cause_false_alarms():
    assert _status("Gini and AUC (on OOT) are 0.5602 (AUC 0.7801) for the champion.") == [
        "verified",
        "verified",
    ]
    assert _status("Champion OOT Gini 0.5602, and Gini of 0.5008 for the benchmark.") == [
        "verified",
        "verified",
    ]


def test_invented_number_is_unverified_and_tables_skip_label_check():
    assert _status("The champion lead in Gini is 0.59396.") == ["unverified"]
    assert _status("| challenger PD PSI | 0.0152 |") == ["verified"]


FACTS = {"findings": {"list": [{"title": "Rank-ordering breaks across PD deciles (OOT)"},
                               {"title": "Stability testing is not temporal (pseudo-OOT)"}]}}  # fmt: skip


@pytest.mark.parametrize("sentence, n", [
    ("Rank-ordering performance is strong across the score distribution.", 1),
    ("Rank-ordering is consistent apart from one break across deciles.", 0),
    ("The model is temporally stable.", 1),
    ("Temporal stability remains untested.", 0),
])  # fmt: skip
def test_contradictions_against_findings(sentence, n):
    assert len(consistency.check(sentence, FACTS, [], [])) == n


def test_rules_only_apply_when_their_finding_is_present():
    assert consistency.check("Rank-ordering is strong.", {"findings": {"list": []}}, [], []) == []


def test_critic_analyses_claim_needs_analyses():
    s = "The critic challenged the framework, prompting specific analytical reviews."
    ran = [{"kind": "segment_cut"}]
    critic = [{"challenges": [{"title": "x"}]}]
    assert len(consistency.check(s, FACTS, [], critic)) == 1
    assert consistency.check(s, FACTS, ran, critic) == []


def test_v2_prompts_are_configured_and_fully_filled(init_state):
    from acr.config import load_config
    from acr.llm import load_prompt

    p = load_config().llm.prompts
    assert p == {"critic": "v2", "doc_section": "v2"}
    critic = load_prompt("critic", p["critic"])
    assert "restates those findings is a failed review" in critic
    assert "{{CRITIC}}" in load_prompt("doc_section", p["doc_section"])
