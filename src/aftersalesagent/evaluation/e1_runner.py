"""固定证据/事实/版本规则评分；不向模型提供私有criteria。"""

import re


def verify(result, criteria):
    answer = result["answer"]
    citations = set(answer["citations"]) if answer else set()
    gold = set(criteria["required_ids"])
    checks = {
        "valid_model_output": answer is not None and result["error"] is None,
        "status_correct": bool(answer and answer["status"] == criteria["status"]),
        "key_facts": bool(
            answer and all(re.search(p, answer["answer"], re.I) for p in criteria["fact_patterns"])
        ),
        "citation_ids_grounded": result["citation_ids_valid"],
        "required_citations_present": gold <= citations,
        "cited_versions_consistent": result["citation_versions_valid"],
        "version_selection_correct": bool(
            answer and set(answer["policy_versions"]) == set(criteria["versions"])
        ),
    }
    found = set(result["context_ids"])
    return {
        "success": all(checks.values()),
        "checks": checks,
        "failure_labels": [k for k, v in checks.items() if not v],
        "evidence_recall": len(gold & found) / len(gold) if gold else None,
        "strict_reference_set_exact": citations == gold,
        "required_reference_precision": len(gold & citations) / len(citations)
        if citations
        else (1.0 if not gold else 0.0),
        "gold_evidence_present": gold <= found,
    }


def summarize(rows):
    summary = {}
    for mode in ("full", "rag"):
        selected = [r for r in rows if r["mode"] == mode]
        if not selected:
            continue
        n = len(selected)
        known = [r["result"]["usage"]["reported_tokens"] for r in selected]
        recall = [
            r["verification"]["evidence_recall"]
            for r in selected
            if r["verification"]["evidence_recall"] is not None
        ]
        summary[mode] = {
            "cases": n,
            "passed": sum(r["verification"]["success"] for r in selected),
            "rule_based_accuracy": sum(r["verification"]["success"] for r in selected) / n,
            "evidence_recall": sum(recall) / len(recall) if recall else None,
            "recall_denominator": len(recall),
            "key_fact_passed": sum(r["verification"]["checks"]["key_facts"] for r in selected),
            "citation_passed": sum(
                r["verification"]["checks"]["required_citations_present"] for r in selected
            ),
            "version_passed": sum(
                r["verification"]["checks"]["version_selection_correct"] for r in selected
            ),
            "model_calls": n,
            "reported_tokens": sum(t or 0 for t in known),
            "input_tokens": sum(r["result"]["usage"]["input_tokens"] or 0 for r in selected),
            "output_tokens": sum(r["result"]["usage"]["output_tokens"] or 0 for r in selected),
            "unknown_usage_calls": sum(
                r["result"]["usage"]["unknown_usage_calls"] for r in selected
            ),
            "mean_latency_seconds": sum(r["result"]["latency_seconds"] for r in selected) / n,
            "max_context_characters": max(r["result"]["context_characters"] for r in selected),
            "cost_estimate": None,
        }
    return summary
