"""Offline source integrity and retrieval smoke audit; not model evaluation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prepare_retail_corpus import BASE, ROOT, sha

from aftersalesagent.knowledge.corpus import eligible, load_corpus
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex


def audit():
    chunks, manifest = load_corpus(ROOT, "retail_sim_v1")
    rules = [json.loads(s) for s in (BASE / "rules.jsonl").read_text("utf-8").splitlines()]
    by_id = {r["rule_id"]: r for r in rules}
    assert len(by_id) == len(rules)
    for r in rules:
        text = (ROOT / r["source_text_path"]).read_text("utf-8")
        assert sha(text[r["source_start"] : r["source_end"]].encode()) == r["source_excerpt_sha256"]
    coverage = json.loads((BASE / "coverage.json").read_text("utf-8"))
    for cid, link in coverage["chunks"].items():
        assert cid in {c.chunk_id for c in chunks}
        assert link["derived_from"] and set(link["derived_from"]) <= by_id.keys()
    assert all(c.effective_from is None and c.effective_until is None for c in chunks)
    assert not eligible(chunks, {"scope": "ikea_cn"})
    for filters in [{"scope": "missing"}, {"scope": "retail_sim", "unexpected": "x"}]:
        try:
            eligible(chunks, filters)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid filter accepted")
    legacy, _ = load_corpus(ROOT, "e1_v1")
    load_corpus(ROOT, "e4_v1")
    index = SQLiteKnowledgeIndex(chunks + legacy)
    cases = [
        ("订单已经处理还能取消吗", "03_cancellation"),
        ("原卡不用了可以退另一张新银行卡吗", "06_refund_methods"),
        ("只退两件商品需要什么信息", "05_returns"),
        ("退货退款多少天到账", "06_refund_methods"),
        ("商品破损能赔多少钱", "08_exceptions_and_handoff"),
        ("修改收货地址需要邮编吗", "04_address_change"),
        ("能把衣服换成鞋吗", "07_products_and_statuses"),
        ("知道订单号可以帮朋友退货吗", "02_identity_and_orders"),
    ]
    observations = []
    for query, expected in cases:
        found = index.retrieve(query, {"scope": "retail_sim"}, 5)
        assert all(c.scope == "retail_sim" for c in found)
        observations.append(
            dict(
                query=query,
                expected=expected,
                returned=[c.chunk_id for c in found],
                hit=expected in {c.chunk_id for c in found},
            )
        )
    assert not index.retrieve("zxqvnonexistent", {"scope": "retail_sim"}, 5)
    index.close()
    result = dict(
        integrity="passed",
        legacy_load="e1_v1/e4_v1 passed",
        section_rule_records=len(rules),
        documents=len(coverage["chunks"]),
        chunks=len(chunks),
        retrieval_smoke=observations,
        hits=sum(o["hit"] for o in observations),
        total=len(observations),
        limitations=("Authored development smoke only; no model answers, "
                      "no independent accuracy or refusal evaluation"),
    )
    out = ROOT / "data/results/retail_sim_v1_offline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    audit()
