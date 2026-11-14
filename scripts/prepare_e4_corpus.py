"Structure-based E4 HTML text spans; preserve exact source offsets and audit exclusions."

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from aftersalesagent.knowledge.corpus import KnowledgeChunk, dump

ROOT = Path(__file__).resolve().parents[1]
TITLES = {
    "mattress": "床垫品质保证",
    "wardrobe": "PAX 帕克思 / KOMPLEMENT 康普蒙 衣柜与内部配件品质保证",
    "sliding": "SKYTTA 斯基塔 滑门 品质保证详情",
    "bathroom": "浴室产品的品质保证",
    "seating": "客厅产品的品质保证",
    "kitchen": "METOD 米多 厨房系列 享有25年品质保证",
    "knives": "宜家厨房刀具品质保证",
    "cookware": "宜家炊具品质保证",
    "office": "办公家具的品质保证",
    "chairs": "办公椅品质保证",
}


def heading(text):
    return (
        len(text) <= 65
        and "。" not in text
        and not text.startswith("•")
        and (
            text.endswith(("？", "?", "：", ":"))
            or text.endswith("说明")
            or ("品质保证" in text and not text.startswith("本品质保证"))
            or text
            in {
                "清洗",
                "磨刀",
                "安装",
                "使用",
                "家电",
                "售后服务",
                "认证",
                "有效范围",
                "第一次使用前",
                "护理和清洁",
                "如何使用",
            }
        )
    )


def main(rebuild=False):
    directory = ROOT / "data/knowledge/e4_v1"
    if (directory / "manifest.json").exists():
        if not rebuild or (ROOT / "data/manifests/e4_suite_v1.json").exists():
            raise SystemExit("E4 corpus exists or suite is frozen; use a new version")
        old_hash = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        backup = ROOT / ".local/e4_artifacts/material_drafts" / old_hash[:12]
        backup.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "chunks.jsonl", "review.json"):
            shutil.copy2(directory / name, backup / name)
    sources = json.loads((directory / "sources/manifest.json").read_text("utf-8"))["documents"]
    chunks, audit, sections = [], [], []
    for source in sources:
        name = source["category"]
        raw = (ROOT / source["text_path"]).read_text("utf-8")
        title = TITLES[name]
        start = raw.find(title)
        if start < 0:
            raise ValueError("Expected policy title absent: " + name)
        end = len(raw)
        for footer in (
            "\n查看其他品质保证",
            "\n返回卧室的产品品质保证",
            "\n返回办公家具的品质保证",
            "\n返回产品品质保证",
        ):
            if (position := raw.find(footer, start)) >= 0:
                end = min(end, position)
        if name == "mattress":
            start = raw.index("床垫质保内容")
            end = raw.index("\n床垫使用注意事项", start)
        audit.append(
            {
                "document_id": source["document_id"],
                "title": title,
                "body_span": [start, end],
                "reviewer_kind": "assistant",
                "status": "body_reviewed",
                "notes": (
                    "Retain source text; exclude navigation, marketing and footer from chunks"
                ),
            }
        )
        lines = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\n]+", raw[start:end])]
        groups = []
        active = []
        section = title
        major = title
        for left, right, line in lines:
            left, right = left + start, right + start
            if "美国地区" in line or "针对美国" in line:
                if active:
                    groups.append((section, active))
                    active = []
                audit.append(
                    {
                        "document_id": source["document_id"],
                        "excluded_span": [left, right],
                        "reason": "Explicitly US-only clause; not mainland-China evidence",
                    }
                )
                continue
            if line.startswith("对于在一个欧盟国家购买") or line.startswith(
                "这些限制不适用于由专业技工"
            ):
                if active:
                    groups.append((section, active))
                    active = []
                audit.append(
                    {
                        "document_id": source["document_id"],
                        "excluded_span": [left, right],
                        "reason": "Explicitly EU cross-border clause; quarantined for scope review",
                    }
                )
                continue
            if heading(line) and active:
                # Consecutive headings (e.g. '5-year guarantee' + 'covered components')
                # must stay together rather than producing evidence with no substantive body.
                if any(not heading(raw[a:b]) for a, b in active):
                    groups.append((section, active))
                    active = []
                if "保养说明" in line or line == "家电" or re.search(r"\d+年品质保证", line):
                    major = line
                section = major + " / " + line
            active.append((left, right))
        if active:
            groups.append((section, active))
        sequence = 0
        for section, spans in groups:
            # Preserve headings with paragraphs/lists. Split only at full stops or line boundaries.
            left, right = spans[0][0], spans[-1][1]
            pieces = []
            cursor = left
            for boundary in re.finditer(r"[。！？]\s*|\n", raw[left:right]):
                stop = left + boundary.end()
                if stop - cursor >= 200:
                    pieces.append((cursor, stop))
                    cursor = stop
            if cursor < right:
                if pieces and right - cursor < 60 and right - pieces[-1][0] <= 420:
                    pieces[-1] = (pieces[-1][0], right)
                else:
                    pieces.append((cursor, right))
            for left, right in pieces:
                # A skipped regional paragraph must never reappear via a surrounding span.
                text = raw[left:right]
                if (
                    "美国地区" in text
                    or "对于在一个欧盟国家购买" in text
                    or "这些限制不适用于由专业技工" in text
                ):
                    audit.append(
                        {
                            "document_id": source["document_id"],
                            "excluded_span": [left, right],
                            "reason": "Span bridges quarantined regional clause",
                        }
                    )
                    continue
                if not text.strip():
                    continue
                sequence += 1
                chunk = KnowledgeChunk(
                    chunk_id=f"E4-{name}-{sequence:03}",
                    document_id=source["document_id"],
                    section=title + " / " + section,
                    text=text,
                    version=source["version"],
                    source=source["final_url"],
                    scope="ikea_cn",
                    origin="official",
                    category=name,
                    source_text_path=source["text_path"],
                    source_start=left,
                    source_end=right,
                    source_excerpt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                )
                chunks.append(chunk)
                sections.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "section": section,
                        "condition_context": (
                            "See same-document/section chunks; "
                            "evaluation requires complete evidence groups"
                        ),
                    }
                )
    # Deliberate project simulations for temporal/conflict checks, never IKEA rules.
    simulations = [
        (
            "E4-SIM-D1",
            "demo_store",
            "e4_accessory",
            "[项目模拟，不是宜家政策] E4演示商店配件补寄：运费由顾客承担。",
            "2026-01-01",
            "2026-06-30",
        ),
        (
            "E4-SIM-D2",
            "demo_store",
            "e4_accessory",
            "[项目模拟，不是宜家政策] E4演示商店配件补寄：运费由商家承担。",
            "2026-07-01",
            None,
        ),
        (
            "E4-SIM-C1",
            "demo_conflict",
            "e4_return",
            (
                "[项目模拟，不是宜家政策] E4冲突商店退货期限条款A：包装"
                "破损商品应在购买后7天内提出退货申请。"
            ),
            "2026-01-01",
            None,
        ),
        (
            "E4-SIM-C2",
            "demo_conflict",
            "e4_return",
            (
                "[项目模拟，不是宜家政策] E4冲突商店退货期限条款B：包装"
                "破损商品应在购买后14天内提出退货申请。"
            ),
            "2026-01-01",
            None,
        ),
        (
            "E4-SIM-C3",
            "demo_conflict",
            "e4_shipping",
            ("[项目模拟，不是宜家政策] E4冲突商店运费条款A：未拆封商品退回运费由顾客承担。"),
            "2026-01-01",
            None,
        ),
        (
            "E4-SIM-C4",
            "demo_conflict",
            "e4_shipping",
            ("[项目模拟，不是宜家政策] E4冲突商店运费条款B：未拆封商品退回运费由商家承担。"),
            "2026-01-01",
            None,
        ),
    ]
    sim_path = directory / "sources/e4_simulations.txt"
    sim_text = "\n".join(row[3] for row in simulations) + "\n"
    sim_path.write_text(sim_text, "utf-8")
    cursor = 0
    for cid, scope, category, text, since, until in simulations:
        chunks.append(
            KnowledgeChunk(
                chunk_id=cid,
                document_id=cid,
                section="E4项目模拟 / " + category,
                text=text,
                version="e4-sim-v1-" + cid,
                source="project-simulation:e4",
                scope=scope,
                origin="simulated",
                category=category,
                effective_from=since,
                effective_until=until,
                source_text_path=sim_path.relative_to(ROOT).as_posix(),
                source_start=cursor,
                source_end=cursor + len(text),
                source_excerpt_sha256=hashlib.sha256(text.encode()).hexdigest(),
            )
        )
        cursor += len(text) + 1
    (directory / "chunks.jsonl").write_text(
        "".join(json.dumps(c.model_dump(), ensure_ascii=False) + "\n" for c in chunks), "utf-8"
    )
    (directory / "review.json").write_bytes(
        dump({"reviewer_kind": "assistant", "sources": audit, "sections": sections})
    )
    files = {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p != directory / "manifest.json"
    }
    (directory / "manifest.json").write_bytes(
        dump(
            {
                "version": "e4_v1",
                "files": files,
                "official_documents": len(sources),
                "chunks": len(chunks),
                "coordinate_system": "unicode codepoint offsets in saved UTF-8 extracted text",
            }
        )
    )
    print(
        json.dumps(
            {
                "sources": len(sources),
                "chunks": len(chunks),
                "max_characters": max(len(c.text) for c in chunks),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    main(parser.parse_args().rebuild)
