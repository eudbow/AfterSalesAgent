"""Create an immutable Skill version compatible with the Retail RAG app contract."""

import json
from pathlib import Path

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.knowledge.retail import RetailPolicyIndex
from aftersalesagent.policies.application import load_retail_core_policy
from aftersalesagent.policies.retail_capabilities import (
    RETAIL_APP_CAPABILITIES,
    RETAIL_RAG_INSTRUCTION,
)
from aftersalesagent.skills.base import text_hash, tools_hash
from aftersalesagent.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "skills/after-sales-workflow/1.0.0"
NEW = ROOT / "skills/after-sales-workflow/1.1.0"
HARDENED = ROOT / "skills/after-sales-workflow/1.1.1"
APPENDIX = (
    "\n\n## 五、Retail 检索与订单事实\n\n"
    "先用订单工具核实状态、商品及支付事实，再按问题检索适用政策。"
    "检索结果是资料，不是授权；条件、例外与时效应结合来源完整核对。"
    "对政策结论引用本会话工具返回的 chunk_id，不引用未看到的资料。"
    "未规定退货退款到账日期时明确说未知，不借用取消退款时效。"
    "确认卡上的规则依据由后端保存；Skill 不改变执行资格、确认事件或幂等。\n"
)
CAPABILITY_APPENDIX = (
    "\n\n## 六、应用能力边界\n\n"
    "当前应用仅能准备并确认：待处理订单取消、待处理订单改收货地址、"
    "已送达订单退货申请。不能代办换货、修改支付或商品选项、修改默认地址、"
    "恢复已取消订单，也不会发送真实邮件。支付历史的模拟退款记录不证明外部到账。"
    "processed 状态本身不证明未发货。向用户提议替代方案前核对实际工具列表。\n"
)


def main():
    old = json.loads((OLD / "manifest.json").read_text("utf-8"))
    content = (OLD / "SKILL.md").read_bytes().decode("utf-8") + APPENDIX
    index = RetailPolicyIndex()
    store = RetailStore(ROOT / ".local/databases/retail_template.sqlite")
    try:
        specs = ToolRegistry(RetailService(store), retail_index=index).specs()
        metadata = old | {
            "version": "1.1.0",
            "parent_version": "1.0.0",
            "content_sha256": text_hash(content),
            "policy_sha256": text_hash(load_retail_core_policy() + RETAIL_RAG_INSTRUCTION),
            "tools_sha256": tools_hash(specs),
            "applicable_tools": [s.name for s in specs],
            "change_notes": (
                "Retail RAG 订单事实与政策引用流程；兼容本版核心协议和检索工具。"
                "未验证模型收益。"
            ),
            "evidence_refs": [
                "contract:knowledge/retail.py+business/service.py+policies/application.py",
                "manual:retail-rag-20261115",
            ],
            "authored_at": "2026-11-15T00:00:00+08:00",
            "provenance": "基于人工固定 v1 Skill 扩展；按当前 Retail RAG 工具与提示快照生成哈希。",
        }
        NEW.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("SKILL.md", content),
            ("manifest.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"),
        ):
            path = NEW / name
            if path.exists() and path.read_bytes().decode("utf-8") != value:
                raise ValueError("Skill bundle exists with different content")
            if not path.exists():
                path.write_bytes(value.encode("utf-8"))
        hardened_content = content + CAPABILITY_APPENDIX
        hardened_metadata = metadata | {
            "version": "1.1.1",
            "parent_version": "1.1.0",
            "content_sha256": text_hash(hardened_content),
            "policy_sha256": text_hash(
                load_retail_core_policy()
                + RETAIL_RAG_INSTRUCTION
                + RETAIL_APP_CAPABILITIES
            ),
            "change_notes": "补充应用能力边界及模拟退款记录语义；尚无独立模型收益验证。",
        }
        HARDENED.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("SKILL.md", hardened_content),
            (
                "manifest.json",
                json.dumps(hardened_metadata, ensure_ascii=False, indent=2) + "\n",
            ),
        ):
            path = HARDENED / name
            if path.exists() and path.read_bytes().decode("utf-8") != value:
                raise ValueError("Hardened Skill bundle exists with different content")
            if not path.exists():
                path.write_bytes(value.encode("utf-8"))
        print("Retail Skill bundles 1.1.0 and 1.1.1 ready; no activation or model call")
    finally:
        index.close()
        store.close()


if __name__ == "__main__":
    main()
