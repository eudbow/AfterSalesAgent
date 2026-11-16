"""Deterministic, offline Retail corpus build. No model or private task inputs."""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/knowledge/retail_sim_v1"
POLICY = "data/upstream/tau2-bench/data/tau2/domains/retail/policy.md"
APP = "src/aftersalesagent/policies/application.py"
SERVICE = "src/aftersalesagent/business/service.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build():
    BASE.mkdir(parents=True, exist_ok=True)
    rules, chunks, links = [], [], {}
    sources = [POLICY, APP, SERVICE]
    for source in sources:
        text = (ROOT / source).read_text("utf-8")
        if source == POLICY:
            spans = list(re.finditer(r"(?m)^#{1,3} .+$", text))
            sections = [
                (
                    m.group().lstrip("# "),
                    m.start(),
                    spans[i + 1].start() if i + 1 < len(spans) else len(text),
                )
                for i, m in enumerate(spans)
            ]
        else:
            sections = [
                (
                    "Application protocol" if source == APP else "Implemented capabilities",
                    0,
                    len(text),
                )
            ]
        for i, (title, start, end) in enumerate(sections):
            rid = f"{'POL' if source == POLICY else 'APP' if source == APP else 'CODE'}-{i + 1:02}"
            rule = dict(
                rule_id=rid,
                section=title,
                source_text_path=source,
                source_start=start,
                source_end=end,
                source_excerpt_sha256=sha(text[start:end].encode()),
                origin="upstream_policy" if source == POLICY else "application_contract",
            )
            rules.append(rule)
            if source == POLICY:
                chunks.append(
                    dict(
                        chunk_id=rid,
                        document_id="retail-policy",
                        section=title,
                        text=text[start:end],
                        version="retail_sim_v1",
                        source=source,
                        scope="retail_sim",
                        **{k: v for k, v in rule.items() if k not in {"rule_id", "section"}},
                    )
                )
    by_title = {r["section"]: r["rule_id"] for r in rules}
    for path in sorted((BASE / "documents").glob("*.md")):
        text = path.read_text("utf-8")
        meta = json.loads(text.splitlines()[0].removeprefix("<!-- ").removesuffix(" -->"))
        refs = [by_title[t] for t in meta["sources"]]
        cid = path.stem
        chunks.append(
            dict(
                chunk_id=cid,
                document_id=cid,
                section=text.splitlines()[1].lstrip("# "),
                text=text,
                version="retail_sim_v1",
                source=path.relative_to(ROOT).as_posix(),
                scope="retail_sim",
                origin="derived_guidance",
                source_text_path=path.relative_to(ROOT).as_posix(),
                source_start=0,
                source_end=len(text),
                source_excerpt_sha256=sha(text.encode()),
            )
        )
        links[cid] = dict(
            rule_ids=refs, derived_from=refs, applies_to=meta["applies_to"], related_rule_ids=[]
        )
    for filename, rows in [("rules.jsonl", rules), ("chunks.jsonl", chunks)]:
        (BASE / filename).write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8",
        )
    write_json(
        BASE / "coverage.json",
        dict(
            chunks=links,
            unknowns=["退货到账时间", "运费", "手续费", "包装要求", "品类禁退", "赔偿", "邮件送达"],
            unsupported=["换货", "修改支付方式", "修改商品规格", "修改默认用户地址"],
            review="Assistant source review; not independent human annotation",
        ),
    )
    files = sources + [
        p.relative_to(ROOT).as_posix()
        for p in sorted(BASE.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    ]
    write_json(
        BASE / "manifest.json",
        dict(
            corpus_id="retail_sim_v1",
            upstream_commit="fc0055dc4e0a316c3f83133267fbd6faaa770992",
            section_rule_records=len(rules),
            independent_atomic_rule_count=None,
            chunk_count=len(chunks),
            document_count=len(links),
            files={p: sha((ROOT / p).read_bytes()) for p in files},
        ),
    )


if __name__ == "__main__":
    build()
