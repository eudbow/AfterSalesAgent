"""Run the offline M0 source/schema audit; does not call a model or execute upstream code."""

import json
from pathlib import Path

from aftersalesagent.business.source_audit import build_report

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    report = build_report(ROOT)
    target = ROOT / ".local/results/m0_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)
