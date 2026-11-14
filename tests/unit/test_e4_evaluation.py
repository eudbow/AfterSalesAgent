"""Counterexamples for evidence alternatives and crash-safe paid-call limits; no network."""

import sqlite3

import pytest

from aftersalesagent.evaluation.e4.budget import GenerationBudget
from aftersalesagent.evaluation.e4.scoring import answer_checks, retrieval_metrics, summarize


def test_complete_alternative_does_not_require_all_redundant_sources():
    gold = {"support_groups": [["A"], ["B", "C"]]}
    assert retrieval_metrics(["A"], gold)["complete_support"]
    assert retrieval_metrics(["B", "C"], gold)["complete_support"]
    partial = retrieval_metrics(["B"], gold)
    assert not partial["complete_support"] and partial["recall_at_5"] == 0.5
    assert retrieval_metrics(["X", "A"], gold)["ndcg_at_5"] < 1
    assert retrieval_metrics(["A"], gold)["ndcg_at_5"] == 1


def test_no_answer_not_in_recall_denominator():
    no_answer = retrieval_metrics([], {"support_groups": []})
    assert no_answer["recall_at_5"] is None
    report = summarize(
        [{"metrics": no_answer}, {"metrics": retrieval_metrics(["A"], {"support_groups": [["A"]]})}]
    )
    assert report["answerable_denominator"] == 1 and report["recall_at_5"] == 1


def test_keywords_and_valid_citation_are_not_semantic_truth():
    result = {
        "answer": {"answer": "十年", "status": "answered", "citations": ["A"]},
        "citation_ids_valid": True,
        "citation_versions_valid": True,
        "error": None,
    }
    gold = {"expected_status": "answered", "support_groups": [["A"]], "required_patterns": ["十年"]}
    score = answer_checks(result, gold)
    assert score["automatic_pass"] and score["semantic_verdict"] == "needs_review"


def test_budget_reservation_survives_crash_and_prevents_automatic_retry(tmp_path):
    path = tmp_path / "budget.sqlite"
    first = GenerationBudget(path)
    first.reserve("run1-case1-L0", "run1")
    first.close()
    second = GenerationBudget(path)
    with pytest.raises(ValueError, match="Unresolved"):
        second.reserve("run2-case1-L0", "run2")
    assert second.report()["reserved_calls"] == 1
    second.finish("run1-case1-L0", input_tokens=None, output_tokens=None)
    with pytest.raises(ValueError, match="Unresolved"):
        second.reserve("run2-case2-L0", "run2")
    second.close()


def test_budget_global_limit_and_duplicate_key(tmp_path):
    budget = GenerationBudget(tmp_path / "budget.sqlite", limit=2)
    budget.reserve("one", "run1")
    budget.finish("one", input_tokens=1, output_tokens=2)
    with pytest.raises(sqlite3.IntegrityError):
        budget.reserve("one", "run1")
    budget.reserve("two", "run2")
    budget.finish("two", input_tokens=1, output_tokens=2)
    with pytest.raises(ValueError, match="exhausted"):
        budget.reserve("three", "run3")
    assert budget.report()["states"] == {"known": 2}
    budget.close()
