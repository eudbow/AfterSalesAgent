"""Build immutable v2-compatible fixed workflow Skill; no model calls."""

import json
from pathlib import Path

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.knowledge.retail_v2 import RetailV2Index
from aftersalesagent.policies.application import load_retail_core_policy
from aftersalesagent.policies.retail_capabilities import (
    RETAIL_APP_CAPABILITIES,
    RETAIL_RAG_INSTRUCTION,
)
from aftersalesagent.skills.base import text_hash, tools_hash
from aftersalesagent.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


def main():
    parent = ROOT / "skills/after-sales-workflow/1.1.1"
    target = ROOT / "skills/after-sales-workflow/1.2.0"
    content = (parent / "SKILL.md").read_text("utf-8") + (
        "\n\n## 七、v2 依据与能力\n\n"
        "固定能力 APP-CAP-V2 与商家政策分别引用。Hybrid 检索后核实条件和应用边界；"
        "不要把模拟退款记录说成外部到账。不得将用户取消原因改写为允许原因。"
        "不提供资料没有支持的拒收或承运方处理流程，模拟交接不承诺真人办理。\n"
    )
    index = RetailV2Index(method="bm25")
    store = RetailStore(ROOT / ".local/databases/retail_template.sqlite")
    try:
        manifest = json.loads((parent / "manifest.json").read_text("utf-8")) | {
            "version": "1.2.0",
            "parent_version": "1.1.1",
            "content_sha256": text_hash(content),
            "policy_sha256": text_hash(
                load_retail_core_policy() + RETAIL_RAG_INSTRUCTION + RETAIL_APP_CAPABILITIES
            ),
            "tools_sha256": tools_hash(
                ToolRegistry(RetailService(store), retail_index=index).specs()
            ),
            "change_notes": "Retail v2 Hybrid 和应用能力工作流；Skill 无独立模型收益评测。",
        }
        target.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("SKILL.md", content),
            ("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)),
        ):
            path = target / name
            if path.exists() and path.read_text("utf-8") != value:
                raise ValueError("Immutable Skill version already differs")
            path.write_bytes(value.encode("utf-8"))
        print("Retail Skill 1.2.0 built; no activation or API call")
    finally:
        index.close()
        store.close()


if __name__ == "__main__":
    main()
