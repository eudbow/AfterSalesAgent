"""Alternative sufficient evidence groups, with explicit limits on automatic answer checks."""

import math
import re


def retrieval_metrics(ids: list[str], gold: dict) -> dict:
    groups = [set(group) for group in gold["support_groups"]]
    if not groups:
        return {
            "recall_at_5": None,
            "recall_at_10": None,
            "complete_support": None,
            "ndcg_at_5": None,
            "mrr": None,
            "answerable": False,
        }

    def recall(k):
        found = set(ids[:k])
        return max(len(group & found) / len(group) for group in groups)

    def ndcg(group):
        gains = [int(cid in group) for cid in ids[:5]]
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(5, len(group))))
        return dcg / ideal

    relevant = set.union(*groups)
    return {
        "recall_at_5": recall(5),
        "recall_at_10": recall(10),
        "complete_support": any(group <= set(ids[:5]) for group in groups),
        "ndcg_at_5": max(ndcg(group) for group in groups),
        "mrr": next((1 / rank for rank, cid in enumerate(ids, 1) if cid in relevant), 0.0),
        "answerable": True,
    }


def answer_checks(result: dict, gold: dict) -> dict:
    answer = result.get("answer") or {}
    text = answer.get("answer", "")
    citations = set(answer.get("citations", []))
    groups = [set(group) for group in gold["support_groups"]]
    checks = {
        "no_technical_error": result.get("error") is None,
        "status_correct": answer.get("status") == gold["expected_status"],
        "citation_ids_valid": bool(result.get("citation_ids_valid")),
        "citation_versions_valid": bool(result.get("citation_versions_valid")),
        "support_group_cited": not groups or any(group <= citations for group in groups),
        "required_expression_checks": all(
            re.search(pattern, text, re.I) for pattern in gold.get("required_patterns", [])
        ),
        "forbidden_expression_checks": not any(
            re.search(pattern, text, re.I) for pattern in gold.get("forbidden_patterns", [])
        ),
    }
    return {
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "semantic_verdict": "needs_review",
        "reviewer_kind": None,
        "note": (
            "Regex and citation membership do not prove semantic entailment; "
            "explicit blinded review required"
        ),
    }


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["metrics"]["answerable"]]
    metrics = {}
    for key in ("recall_at_5", "recall_at_10", "complete_support", "ndcg_at_5", "mrr"):
        metrics[key] = (
            sum(float(r["metrics"][key]) for r in answerable) / len(answerable)
            if answerable
            else None
        )
    return {
        "tasks": len(rows),
        "answerable_denominator": len(answerable),
        "no_answer_denominator": len(rows) - len(answerable),
        **metrics,
    }
