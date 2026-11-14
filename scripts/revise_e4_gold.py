"""Create a new gold version for overlooked legal evidence paths; never alter frozen v1 files."""

import itertools
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.evaluation.e4.execution import sha
from aftersalesagent.knowledge.corpus import dump, load_corpus

ROOT = Path(__file__).resolve().parents[1]


def main():
    public = ROOT / "data/tasks/e4_v2"
    private = ROOT / "data/tasks/private/e4_v2"
    if public.exists() or private.exists():
        raise ValueError("Gold v2 already exists; no overwrite")
    chunks, _ = load_corpus(ROOT, "e4_v1")
    by_id = {c.chunk_id: c for c in chunks}
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    changes = {}
    for split in ("dev", "validation", "test"):
        shutil.copy2(ROOT / f"data/tasks/e4_v1/{split}.jsonl", public / (split + ".jsonl"))
        rows = []
        for line in (
            (ROOT / f"data/tasks/private/e4_v1/{split}.jsonl").read_text("utf-8").splitlines()
        ):
            row = json.loads(line)
            original = {tuple(sorted(group)) for group in row["support_groups"]}
            groups = set(original)
            fid = row["family_id"]
            # Exact repeated paragraphs within one policy are alternative source locations.
            for group in original:
                alternatives = []
                for cid in group:
                    source = by_id[cid]
                    normalized = re.sub(r"\s+", "", source.text)
                    alternatives.append(
                        [
                            c.chunk_id
                            for c in chunks
                            if c.document_id == source.document_id
                            and re.sub(r"\s+", "", c.text) == normalized
                        ]
                    )
                groups.update(
                    tuple(sorted(set(combo))) for combo in itertools.product(*alternatives)
                )
            if fid == "D11":
                metod = [
                    c.chunk_id
                    for c in chunks
                    if c.category == "kitchen"
                    and (
                        c.section.split(" / ")[1] == "METOD 米多 厨房系列 享有25年品质保证"
                        or ("25年品质保证" in c.text and "本品质保证涵盖METOD" in c.text)
                    )
                ]
                knox = [
                    c.chunk_id
                    for c in chunks
                    if c.category == "kitchen" and "KNOXHULT" in c.text and "5年品质保证" in c.text
                ]
                groups.update(tuple(sorted(set(pair))) for pair in itertools.product(metod, knox))
            if fid == "D13":
                normal = [
                    c.chunk_id
                    for c in chunks
                    if c.category == "kitchen"
                    and (
                        "购买家电之日起五(5)年" in c.text
                        or "免费五(5)年质保服务适用于宜家所有家用厨房电器" in c.text
                    )
                ]
                exception = [
                    c.chunk_id
                    for c in chunks
                    if c.category == "kitchen"
                    and "TILLREDA" in c.text
                    and "LAGAN" in c.text
                    and ("两(2)年" in c.text or "2年内有效" in c.text)
                ]
                groups.update(
                    tuple(sorted(set(pair))) for pair in itertools.product(normal, exception)
                )
            if fid == "V04":
                basin = [
                    c.chunk_id
                    for c in chunks
                    if c.category == "bathroom"
                    and (
                        "浴室洗脸池，混合水龙头和恒温淋浴水混合器10年品质保证" in c.text
                        or ("我们提供10年免费质保" in c.text and "所有洗脸池" in c.text)
                    )
                ]
                shower = [
                    c.chunk_id
                    for c in chunks
                    if c.category == "bathroom"
                    and (
                        "适用于混合型水龙头的ÅBÄCKEN" in c.text
                        or (
                            "浴室淋浴配件3年品质保证" in c.section + c.text
                            and "手持花洒" in c.text
                            and "淋浴管" in c.text
                        )
                    )
                ]
                groups.update(tuple(sorted(set(pair))) for pair in itertools.product(basin, shower))
            if fid in {"D18", "V12"}:
                quote = row["source_review"][0]["quote"]
                # These exact clauses also occur in the official chair policy, with the same
                # named exclusions or furniture-maintenance wording. Do not generalize this
                # alias rule to unrelated products merely sharing generic boilerplate.
                groups.update(
                    (c.chunk_id,)
                    for c in chunks
                    if c.category in {"office", "chairs"} and quote in c.text
                )
            if groups != original:
                changes[row["task_id"]] = {"old": sorted(original), "new": sorted(groups)}
                row["gold_revision_note"] = (
                    "Additional source-supported path: scoped section headings "
                    "or reviewed exact duplicate clauses"
                )
            row["support_groups"] = sorted(groups)
            row["gold_version"] = "e4_v2"
            rows.append(row)
        (private / (split + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8"
        )
    review = json.loads((ROOT / "data/tasks/private/e4_v1/review.json").read_text("utf-8"))
    review.update(
        {
            "gold_version": "e4_v2",
            "reviewer_kind": "assistant",
            "revision_reason": (
                "v1 omitted valid structured-heading and duplicate-source evidence paths"
            ),
            "changed_tasks": changes,
            "questions_changed": 0,
            "corpus_changed": False,
            "retrieval_parameters_changed": False,
            "heldout_results_used": False,
            "created_at": datetime.now(UTC).isoformat(),
        }
    )
    (private / "review.json").write_bytes(dump(review))
    base = json.loads((ROOT / "data/manifests/e4_suite_v1.json").read_text("utf-8"))
    files = {k: v for k, v in base["files"].items() if k.startswith("data/knowledge/")}
    for folder in (public, private):
        files.update(
            {p.relative_to(ROOT).as_posix(): sha(p) for p in folder.rglob("*") if p.is_file()}
        )
    target = ROOT / "data/manifests/e4_suite_v2.json"
    target.write_bytes(
        dump(
            {
                **base,
                "suite_version": "e4_v2",
                "files": files,
                "created_at": datetime.now(UTC).isoformat(),
                "previous_manifest_sha256": sha(ROOT / "data/manifests/e4_suite_v1.json"),
                "revision_report": (private / "review.json").relative_to(ROOT).as_posix(),
            }
        )
    )
    print(
        json.dumps(
            {
                "new_suite": "e4_v2",
                "changed_tasks": len(changes),
                "changed_families": sorted({key.split("-")[1] for key in changes}),
                "questions_changed": 0,
                "heldout_results_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
