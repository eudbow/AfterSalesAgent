"""全文/检索同配置成对比较；默认不执行，留出要求冻结。"""

import argparse
import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from aftersalesagent.evaluation.e1_runner import summarize, verify
from aftersalesagent.evaluation.e1_suite import load
from aftersalesagent.knowledge.corpus import dump, load_corpus
from aftersalesagent.knowledge.factory import make_service
from aftersalesagent.knowledge.service import KnowledgeQuestion
from aftersalesagent.local_config import load_local_config, options_from_settings

ROOT = Path(__file__).resolve().parents[1]


def config(settings, manifest, tokenizer, top_k):
    _, corpus = load_corpus(ROOT)
    options = options_from_settings(settings)
    source = sorted([*ROOT.joinpath("src").rglob("*.py"), Path(__file__).resolve()])
    return {
        "options": options.model_dump(mode="json"),
        "model_fingerprint": hashlib.sha256(options.model_dump_json().encode()).hexdigest(),
        "task_hashes": manifest["files"],
        "knowledge_hashes": corpus["files"],
        "tokenizer": tokenizer,
        "top_k": top_k,
        "source_hashes": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source
        },
        "protocol": (
            "one read-only request; identical core/tools/options; only supplemental evidence varies"
        ),
        "context_char_limit": 50000,
    }


async def run(args, cases, settings, secret, snapshot):
    run_id = "e1_" + uuid.uuid4().hex[:10]
    folder = ROOT / ".local/evaluations" / run_id
    folder.mkdir(parents=True)
    (folder / "config.json").write_bytes(dump(snapshot))
    service = make_service(
        ROOT,
        settings,
        secret,
        template=ROOT / ".local/databases/retail_template.sqlite",
        database=folder / "knowledge.sqlite",
        tokenizer=args.tokenizer,
        top_k=args.top_k,
        enable_paid_api=True,
    )
    rows = []
    try:
        for index, case in enumerate(cases):
            # 成对顺序交替，减少总是全文先运行造成的缓存/时间偏差；不声称控制所有延迟因素。
            modes = ("full", "rag") if index % 2 == 0 else ("rag", "full")
            for mode in modes:
                public = case["public"]
                question = KnowledgeQuestion(
                    question=public["question"], mode=mode, **public["filters"]
                )
                result = await service.ask(question, public["task_id"] + "_" + mode)
                verdict = verify(result, case["criteria"])
                row = {
                    "task_id": public["task_id"],
                    "split": args.split,
                    "mode": mode,
                    "verification": verdict,
                    "result": result,
                }
                rows.append(row)
                (folder / (public["task_id"] + "_" + mode + ".json")).write_bytes(dump(row))
                print(
                    json.dumps(
                        {
                            "task": public["task_id"],
                            "mode": mode,
                            "passed": verdict["success"],
                            "failures": verdict["failure_labels"],
                        }
                    ),
                    flush=True,
                )
    finally:
        await service.close()
    report = {
        "run_id": run_id,
        "label": args.label,
        "split": args.split,
        "config": snapshot,
        "metrics": summarize(rows),
        "cases": rows,
        "official_tau2_score": False,
        "notes": [
            "Full means all eligible curated evidence, not website HTML/navigation",
            "Shared corpus/facts across splits; small authored task suite",
            "Rule checks are not complete semantic entailment or independent human truth",
        ],
    }
    path = folder / "summary.json"
    path.write_bytes(dump(report))
    print("Summary:", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "validation", "test"], default="dev")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--tokenizer", choices=["unicode61", "trigram", "char"], default="char")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--label", default="preview")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-paid-api", action="store_true")
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--write-freeze", type=Path)
    parser.add_argument("--freeze-report", type=Path)
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 23:
        parser.error("top-k must be 1..23")
    cases, manifest = load(ROOT, args.split)
    settings, secret = load_local_config(args.env_file)
    snapshot = config(settings, manifest, args.tokenizer, args.top_k)
    if args.freeze_report and json.loads(args.freeze_report.read_text("utf-8")) != snapshot:
        parser.error("Frozen E1 source/model/tasks/knowledge/config differ")
    if args.write_freeze:
        if args.execute or args.write_freeze.exists():
            parser.error("Freeze cannot execute or overwrite")
        args.write_freeze.parent.mkdir(parents=True, exist_ok=True)
        args.write_freeze.write_bytes(dump(snapshot))
    print(
        json.dumps(
            {
                "execute": args.execute,
                "split": args.split,
                "cases": len(cases),
                "maximum_model_calls": len(cases) * 2,
                "preview_api_calls": 0,
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.execute:
        return
    if not args.enable_paid_api or not args.smoke_report:
        parser.error("Explicit paid opt-in and smoke required")
    smoke = json.loads(args.smoke_report.read_text("utf-8"))
    if (
        not smoke.get("passed")
        or smoke.get("settings_fingerprint") != snapshot["model_fingerprint"]
    ):
        parser.error("Smoke model configuration differs")
    if args.split != "dev" and not args.freeze_report:
        parser.error("Held-out requires frozen experiment")
    if args.split == "test" and not args.final_test:
        parser.error("Test requires --final-test")
    asyncio.run(run(args, cases, settings, secret, snapshot))


if __name__ == "__main__":
    main()
