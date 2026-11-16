"""Post-hoc structural evidence-ID coverage; never a semantic support score."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/results/retail_agent_rag_v1/comparison_v1"
FROZEN = ROOT / "data/results/retail_agent_rag_v1/scenarios_v1.json"


def read(path):
    return json.loads(path.read_text("utf-8"))


def main():
    config = read(BASE / "evidence_groups_posthoc_v1.json")
    cases = read(FROZEN)["cases"]
    ids = {case["id"] for case in cases}
    if set(config["groups"]) | set(config["not_applicable"]) != ids:
        raise ValueError("Evidence audit does not cover all frozen cases")
    rows = []
    for case in cases:
        for arm in ("full", "tool_rag"):
            run = read(BASE / f"{case['id']}-{arm}.json")
            citation_ids = {source["chunk_id"] for source in run["first_sources"]}
            card_ids = {
                basis["rule_id"] for basis in (run["action"] or {}).get("policy_basis", [])
            }
            available = citation_ids | card_ids
            groups = config["groups"].get(case["id"])
            missing = (
                []
                if groups is None
                else [group for group in groups if not available.intersection(group)]
            )
            rows.append(
                {
                    "id": case["id"],
                    "split": case["split"],
                    "arm": arm,
                    "status": "not_applicable"
                    if groups is None
                    else "covered"
                    if not missing
                    else "missing",
                    "answer_citations": sorted(citation_ids),
                    "backend_card_basis": sorted(card_ids),
                    "missing_groups": missing,
                }
            )
    totals = {
        arm: {
            "covered": sum(r["status"] == "covered" for r in rows if r["arm"] == arm),
            "applicable": sum(r["status"] != "not_applicable" for r in rows if r["arm"] == arm),
            "missing_cases": [
                r["id"] for r in rows if r["arm"] == arm and r["status"] == "missing"
            ],
        }
        for arm in ("full", "tool_rag")
    }
    result = {
        "status": config["status"],
        "definition": config["definition"],
        "totals": totals,
        "rows": rows,
    }
    (BASE / "evidence_coverage_posthoc_v1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
