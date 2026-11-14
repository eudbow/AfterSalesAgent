"""从本地官方快照构建已审核证据卡片；模拟版本有独立范围；已有内容不覆盖。"""

import hashlib
import json
from pathlib import Path

from aftersalesagent.knowledge.corpus import KnowledgeChunk, dump

ROOT = Path(__file__).resolve().parents[1]


def build():
    directory = ROOT / "data/knowledge/e1_v1"
    sources = json.loads((directory / "sources/manifest.json").read_text("utf-8"))
    source_map = {d["document_id"]: d for d in sources["documents"]}
    chunks = []
    # 仅把段落中已核对的事实整理成短卡片；完整原文/偏移单独保留，不冒充原文自动切块。
    cards = [
        (
            "R01",
            "ikea_returns",
            "退货期限",
            "您可在60天内退货",
            "普通顾客退货期限为60天，宜家俱乐部会员为365天；同时须满足货品及凭证条件。",
        ),
        (
            "R02",
            "ikea_returns",
            "退货材料和银行卡退款",
            "您可在60天内退货",
            "退货需完好货品、原始购物凭证和已开具的发票；银行卡退款还需原始签购单和银行卡。",
        ),
        (
            "R03",
            "ikea_returns",
            "不适用的商品",
            "很抱歉：食品",
            "食品、家用电器、购买前另签协议的特殊定制品，以及被切割、裁剪、油漆的产品不适用该退货政策。",
        ),
        (
            "R04",
            "ikea_returns",
            "发票已入账抵扣",
            "您可在60天内退货",
            "已开票且发票已入账或抵扣时，须先在电子税局完成红字信息确认单的确认流程，再办理退货。",
        ),
        (
            "R05",
            "ikea_returns",
            "货品完好条件",
            "“完好的货品”",
            "完好货品须未受损、无污渍、未变形、未经洗涤、无使用痕迹等。",
        ),
        (
            "R06",
            "ikea_returns",
            "退款处理时间",
            "宜家收到退货商品并确认无误后",
            "收到退货商品并确认无误后，宜家在7日内办理退款手续，按原始支付方式退回；不等同于付款机构即时到账。",
        ),
        (
            "R07",
            "ikea_returns",
            "线上发票中心时间",
            "对于线上购买的产品",
            "线上购买的产品，发票中心工作日09:00至17:30处理，节假日与非工作日不处理发票。",
        ),
        (
            "R08",
            "ikea_returns",
            "配送费用",
            "如果订单在送货当天",
            "订单在送货当天或配送服务完成后取消，运费不予退还。",
        ),
        (
            "R09",
            "ikea_returns",
            "混合付款退款",
            "如果购买时是以多种支付方式",
            "购买时采用多种支付方式，退货退款按各种支付方式的比例退回。",
        ),
        (
            "R10",
            "ikea_returns",
            "政策地域和渠道",
            "* 以上退货政策",
            "适用范围为中国大陆宜家商场、宜家官网、官方APP和小程序及电商商城所购货品，不能直接套用到其他地区。",
        ),
        (
            "R11",
            "ikea_returns",
            "线上退货方式",
            "宜家提供有偿上门收货",
            "可选择有偿上门收货或自行邮寄至指定商场。",
        ),
        (
            "G01",
            "ikea_guarantees",
            "质保开始和凭证",
            "宜家产品的品质保证从",
            "质保从购买之日起开始，索赔需提供原始购买凭证。",
        ),
        (
            "G02",
            "ikea_guarantees",
            "床垫质保",
            "宜家床垫皆经过严格的测试",
            "床垫概览列出10年品质保证，涵盖材料和工艺；具体保障范围仍须核对完整条款，不代表任意损坏都承保。",
        ),
        (
            "G03",
            "ikea_guarantees",
            "浴室系列与配件",
            "ENHET 安纳特",
            "页面列出的浴室系列享有10年免费质保，淋浴配件为3年；仅限页面列出范围，须核对完整条款。",
        ),
        (
            "G04",
            "ikea_guarantees",
            "厨房系列质保区别",
            "METOD 米多",
            "厨房系列METOD米多为25年、ENHET安纳特10年、KNOXHULT诺克胡5年品质保证，涵盖材料和工艺缺陷，具体范围见完整条款。",
        ),
        (
            "G05",
            "ikea_guarantees",
            "厨房刀具与锅具",
            "宜家厨房刀具可享受15年",
            "厨房刀具可享15年品质保证；无不粘涂层的锅具为15年免费质保，各以完整条款范围为准。",
        ),
        (
            "G06",
            "ikea_guarantees",
            "办公椅品质保证",
            "宜家办公椅皆经过测试",
            "办公椅概览列出10年品质保证，涵盖活动部件、框架材料、工艺及功能瑕疵，以完整文档条款为准。",
        ),
    ]
    for cid, doc_id, title, probe, text in cards:
        doc = source_map[doc_id]
        raw = (ROOT / doc["text_path"]).read_text("utf-8")
        start = raw.index(probe)
        end = raw.find("\n", start)
        if end == -1:
            end = len(raw)
        # 两条概览跨相邻段落，保留包含所归纳事实的原文跨度。
        if cid == "G03":
            end = raw.index("淋浴配件享有3年免费质保。", start) + len("淋浴配件享有3年免费质保。")
        if cid == "G05":
            end = raw.index("无不粘涂层的宜家锅具可享受15年免费质保。", start) + len(
                "无不粘涂层的宜家锅具可享受15年免费质保。"
            )
        chunks.append(
            KnowledgeChunk(
                chunk_id=cid,
                document_id=doc_id,
                section=title,
                text=text,
                version=doc["version"],
                source=doc["source_url"],
                scope="ikea_cn",
                origin="official",
                source_text_path=doc["text_path"],
                source_start=start,
                source_end=end,
                source_excerpt_sha256=hashlib.sha256(raw[start:end].encode()).hexdigest(),
            ).model_dump(mode="json")
        )
    simulation = [
        (
            "D01",
            "return-v1",
            "demo_store",
            "2026-01-01",
            "2026-06-30",
            "模拟退货期限旧版",
            "【项目模拟条款，非宜家政策】模拟商店在本版本有效期内，收货后14天内可申请退货。",
        ),
        (
            "D02",
            "return-v2",
            "demo_store",
            "2026-07-01",
            "2026-12-31",
            "模拟退货期限新版",
            "【项目模拟条款，非宜家政策】模拟商店在本版本有效期内，收货后30天内可申请退货。",
        ),
        (
            "D03",
            "materials-v1",
            "demo_store",
            "2026-01-01",
            "2026-06-30",
            "模拟售后材料旧版",
            "【项目模拟条款，非宜家政策】模拟商店售后材料：订单号、问题照片。",
        ),
        (
            "D04",
            "materials-v2",
            "demo_store",
            "2026-07-01",
            "2026-12-31",
            "模拟售后材料新版",
            "【项目模拟条款，非宜家政策】模拟商店售后材料：订单号、问题照片、商品序列号。",
        ),
        (
            "C01",
            "conflict-a",
            "demo_conflict",
            "2026-01-01",
            "2026-12-31",
            "模拟资料冲突A",
            "【项目模拟冲突资料，非宜家政策】模拟商店甲文件称：退货审核需5个工作日。",
        ),
        (
            "C02",
            "conflict-b",
            "demo_conflict",
            "2026-01-01",
            "2026-12-31",
            "模拟资料冲突B",
            "【项目模拟冲突资料，非宜家政策】模拟商店乙文件称：退货审核需10个工作日。",
        ),
    ]
    simulated_text = "\n".join(row[-1] for row in simulation) + "\n"
    sim_rel = "data/knowledge/e1_v1/sources/simulated_policy.txt"
    sim_bytes = simulated_text.encode()
    for cid, version, scope, start_date, end_date, title, text in simulation:
        start = simulated_text.index(text)
        chunks.append(
            KnowledgeChunk(
                chunk_id=cid,
                document_id="project_simulation",
                section=title,
                text=text,
                version=version,
                source="project://e1_v1/simulated-policy",
                scope=scope,
                origin="simulated",
                effective_from=start_date,
                effective_until=end_date,
                source_text_path=sim_rel,
                source_start=start,
                source_end=start + len(text),
                source_excerpt_sha256=hashlib.sha256(text.encode()).hexdigest(),
            ).model_dump(mode="json")
        )
    files = {
        sim_rel: sim_bytes,
        "data/knowledge/e1_v1/chunks.jsonl": "".join(
            json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in chunks
        ).encode(),
    }
    for doc in sources["documents"]:
        for kind in ("html", "text"):
            p = ROOT / doc[kind + "_path"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == doc[kind + "_sha256"]
            files[doc[kind + "_path"]] = p.read_bytes()
    files["data/knowledge/e1_v1/sources/manifest.json"] = (
        directory / "sources/manifest.json"
    ).read_bytes()
    manifest = {
        "corpus_id": "aftersales-knowledge-e1-v1",
        "snapshot_date": "2026-10-24",
        "official_documents": 2,
        "official_chunks": 17,
        "simulated_chunks": 6,
        "chunk_count": 23,
        "review": (
            "Assistant curated paraphrases "
            "with exact source spans; not i"
            "ndependent human annotation"
        ),
        "date_rule": (
            "Official effective dates unkno"
            "wn; simulated dates explicit; "
            "fetch date never substituted"
        ),
        "files": {p: hashlib.sha256(b).hexdigest() for p, b in files.items()},
    }
    files["data/knowledge/e1_v1/manifest.json"] = dump(manifest)
    for rel, content in files.items():
        path = ROOT / rel
        if path.exists() and path.read_bytes() != content:
            raise ValueError("Frozen corpus differs; create another version")
    for rel, content in files.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
