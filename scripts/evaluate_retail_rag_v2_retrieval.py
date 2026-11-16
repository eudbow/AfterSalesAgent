"""Offline retrieval metrics over recorded v2 rankings and explicit post-hoc qrels."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/results/retail_agent_rag_v2"
OUT = BASE / "retrieval_metrics"
KS = (1, 3, 5, 10)


def rank_metrics(ranking, relevant, k):
    if not relevant:
        raise ValueError("A judged query must have at least one relevant document")
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate ranked document")
    top = ranking[:k]
    first = next((i for i, cid in enumerate(top, 1) if cid in relevant), None)
    dcg = sum(1 / math.log2(i + 1) for i, cid in enumerate(top, 1) if cid in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return {
        f"recall@{k}": len(set(top) & relevant) / len(relevant),
        f"mrr@{k}": 0 if first is None else 1 / first,
        f"ndcg@{k}": dcg / ideal,
    }


def main():
    raw = (BASE / "retrieval_diagnostics.json").read_bytes()
    diagnostics = json.loads(raw)
    qrels = json.loads((OUT / "qrels_v1.json").read_text("utf-8"))
    if hashlib.sha256(raw).hexdigest() != qrels["diagnostics_sha256"]:
        raise ValueError("Diagnostics changed after labeling")
    labels = {r["qid"]: r for r in qrels["queries"]}
    expected = {q["case"] + ":" + q["call_id"] for q in diagnostics}
    if set(labels) != expected or len(labels) != len(diagnostics):
        raise ValueError("Query/qrels coverage mismatch")
    corpus = set(qrels["corpus_ids"])
    rows, contexts = [], []
    for query in diagnostics:
        qid = query["case"] + ":" + query["call_id"]
        label = labels[qid]
        if label["query"] != query["query"] or set(label["relevance"]) != corpus:
            raise ValueError("Judgment query/corpus mismatch")
        relevant = {cid for cid, score in label["relevance"].items() if score == 1}
        if relevant != set(label["relevant_ids"]):
            raise ValueError("Judgment relevance mismatch")
        rankings = {
            "bm25": query["bm25"]["rankings"]["lexical"],
            "dense": query["hybrid"]["rankings"]["dense"],
            "hybrid_rrf": query["hybrid"]["rankings"]["fused"],
        }
        for method, ranking in rankings.items():
            if not set(ranking) <= corpus:
                raise ValueError("Unknown document in ranking")
            row = {
                "qid": qid,
                "case": query["case"],
                "method": method,
                "relevant_count": len(relevant),
                "top10": ranking[:10],
            }
            for k in KS:
                row.update(rank_metrics(ranking, relevant, k))
            first = next((i for i, cid in enumerate(ranking, 1) if cid in relevant), None)
            row["mrr_observed_candidates"] = 0 if first is None else 1 / first
            rows.append(row)
        run = json.loads(
            (BASE / "iteration_04" / f"{query['case']}-tool_rag.json").read_text("utf-8")
        )
        matches = [r for r in run["tool_results"] if r["call_id"] == query["call_id"]]
        if len(matches) != 1 or matches[0]["status"] != "ok":
            raise ValueError("Missing actual successful tool result")
        final_ids = {e["chunk_id"] for e in matches[0]["data"]["evidence"]}
        contexts.append(
            {
                "qid": qid,
                "case": query["case"],
                "final_evidence_ids": sorted(final_ids),
                "recall_all_final_context": len(final_ids & relevant) / len(relevant),
                "all_relevant_covered": relevant <= final_ids,
                "missing_relevant_ids": sorted(relevant - final_ids),
            }
        )
    metric_names = [key for key in rows[0] if "@" in key or key == "mrr_observed_candidates"]
    query_macro, case_macro = {}, {}
    for method in ("bm25", "dense", "hybrid_rrf"):
        selected = [row for row in rows if row["method"] == method]
        query_macro[method] = {key: mean(row[key] for row in selected) for key in metric_names}
        cases = defaultdict(list)
        for row in selected:
            cases[row["case"]].append(row)
        case_macro[method] = {
            key: mean(mean(row[key] for row in group) for group in cases.values())
            for key in metric_names
        }
    summary = {
        "status": qrels["status"],
        "n_queries": len(diagnostics),
        "n_cases_with_search": len({q["case"] for q in diagnostics}),
        "excluded_no_search_cases": ["R02", "R11", "R12", "R14"],
        "main_unit": "query macro; actual retrieval calls including repair queries",
        "binary_relevance_policy": qrels["policy"],
        "limitations": qrels["limitations"],
        "ranking_stage": "raw lexical/dense/RRF parent chunk ranking BEFORE dependencies",
        "query_macro": query_macro,
        "case_macro": case_macro,
        "final_context": {
            "mean_recall_all_final_context": mean(r["recall_all_final_context"] for r in contexts),
            "all_relevant_covered_queries": sum(r["all_relevant_covered"] for r in contexts),
            "n_queries": len(contexts),
            "note": (
                "Actual v2 tool context after relation expansion; "
                "unordered, no MRR/nDCG assigned"
            ),
        },
    }
    for name, obj in (
        ("summary.json", summary),
        ("per_query.json", rows),
        ("final_context.json", contexts),
    ):
        (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", "utf-8")
    with (OUT / "per_query.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
