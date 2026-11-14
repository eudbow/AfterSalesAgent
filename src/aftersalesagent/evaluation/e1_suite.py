"""公开知识问题与私有证据/事实/版本目标分开冻结。"""

import hashlib
import json
from pathlib import Path

from aftersalesagent.knowledge.corpus import dump, load_corpus


def build(root: Path):
    chunks, corpus = load_corpus(root)
    by_id = {c.chunk_id: c for c in chunks}
    wording = [
        [
            "宜家普通顾客和俱乐部会员分别有多少天可以退货？",
            "宜家的食品、家用电器、单独协议的特殊定制产品能用这份退货政策吗？",
            "宜家用银行卡支付后退货，需要带哪些货品和凭证？",
            "宜家退货发票已入账抵扣，还需要先完成什么手续？",
            "宜家退货所说的货品完好对污渍、洗涤、使用痕迹有什么要求？",
            "宜家收到退货并确认无误后，几日内办理退款，退到哪里？",
            "宜家床垫的概览质保多少年，涵盖哪些缺陷？",
            "METOD、ENHET、KNOXHULT厨房系列分别保修多少年？",
            "能否凭这次官方快照确认2020年适用的宜家退货期限？",
        ],
        [
            "非会员与宜家俱乐部会员的退货时间上限各是多少？",
            "食品、电器以及购买前另签协议的定制品，在宜家一般退货范围内吗？",
            "银行卡买的宜家商品要退掉，原购物凭证、发票、卡片等需要哪些？",
            "宜家商品要退货，发票已抵扣应先办哪张确认单、在哪办？",
            "有污渍、已洗过或用过的货品，符合宜家页面中的完好定义吗？",
            "宜家验收退回商品没问题以后，多长时间办退款、按何种方式退？",
            "床垫页面写了几年的品质保证，材料与工艺是否在概览保障中？",
            "厨房METOD、ENHET、KNOXHULT三系列的质保年限有何区别？",
            "这份抓取资料能证明2021年生效的宜家退货规则吗？",
        ],
        [
            "宜家非俱乐部成员与俱乐部成员的退货期限有什么差别？",
            "我想按宜家一般退货政策退食品、家用电器和特殊协议定制产品，行吗？",
            "原来刷银行卡买宜家商品，现在退货，应准备货品及哪些材料？",
            "已经入账或抵扣的发票在宜家退货前要如何处理？",
            "退给宜家的商品是不是必须没污渍、没洗过且没使用痕迹？",
            "退回宜家的商品验收无误后，退款办理期限和支付渠道是什么？",
            "宜家床垫品质保证的概览年限，以及材料和工艺保障是什么？",
            "请分别列出METOD、ENHET、KNOXHULT厨房系列的品质保证年数。",
            "是否能拿当前这两页官方快照说明2019年宜家政策已经生效？",
        ],
    ]
    facts = [
        ["60", "365"],
        ["食品", "家用电器|电器", "定制"],
        ["购物凭证|购物小票|收据|购买凭证", "发票", "银行卡", "签购单"],
        ["红字信息确认单", "电子税局|电子税务局"],
        ["污渍", "洗涤|洗过|清洗", "使用痕迹"],
        ["7|七", "原始支付|原支付|原付款|原路|原来的支付"],
        ["10|十年", "材料", "工艺"],
        [
            r"(METOD|米多)[\s\S]{0,30}25|25[\s\S]{0,30}(METOD|米多)",
            r"(ENHET|安纳特)[\s\S]{0,30}10|10[\s\S]{0,30}(ENHET|安纳特)",
            r"(KNOXHULT|诺克胡)[\s\S]{0,30}5|5[\s\S]{0,30}(KNOXHULT|诺克胡)",
        ],
        ["无法|不能|不足|未知|缺乏|未标"],
    ]
    official = ["R01", "R03", "R02", "R04", "R05", "R06", "G02", "G04", None]
    visible, private = [], []
    for j, split in enumerate(("dev", "validation", "test")):
        for i in range(12):
            tid = f"e1-{split}-{i + 1:02d}"
            if i < 9:
                filters = {"scope": "ikea_cn"}
                if i == 8:
                    filters["as_of"] = ("2020-01-01", "2021-01-01", "2019-01-01")[j]
                query = wording[j][i]
                gold = [official[i]] if official[i] else []
                status = "answered" if i != 8 else "insufficient_evidence"
                patterns = facts[i]
                category = (
                    "conditions",
                    "scope",
                    "materials",
                    "materials",
                    "conditions",
                    "process",
                    "category",
                    "category",
                    "unknown_date",
                )[i]
            elif i == 9:
                day = ("2026-03-15", "2026-08-01", "2026-06-30")[j]
                filters = {"scope": "demo_store", "as_of": day}
                query = f"【仅模拟商店，非宜家】按{day}适用版本，收货后多少天内能申请退货？"
                gold = ["D02" if j == 1 else "D01"]
                patterns = ["30" if j == 1 else "14"]
                status, category = "answered", "version"
            elif i == 10:
                day = ("2026-09-01", "2026-05-01", "2026-07-01")[j]
                filters = {"scope": "demo_store", "as_of": day}
                query = f"【仅模拟商店】在{day}申请售后，要提供哪些材料？请按当时版本回答。"
                gold = ["D03" if j == 1 else "D04"]
                patterns = ["订单号", "照片"] + ([] if j == 1 else ["序列号"])
                status, category = "answered", "version"
            else:
                filters = {"scope": "demo_conflict", "as_of": "2026-10-01"}
                query = (
                    "【项目模拟冲突资料】同一天有效的甲乙文件中，"
                    "退货审核究竟需要几个工作日？如有矛盾请指出，不能自行选一个。"
                )
                gold, patterns = ["C01", "C02"], ["5|五", "10|十", "冲突|矛盾|不一致"]
                status, category = "conflict", "conflict"
            visible.append(
                {
                    "task_id": tid,
                    "question": query,
                    "filters": filters,
                    "split": split,
                    "category": category,
                }
            )
            private.append(
                {
                    "task_id": tid,
                    "required_ids": gold,
                    "status": status,
                    "fact_patterns": patterns,
                    "versions": sorted({by_id[x].version for x in gold}),
                }
            )
    files = {
        "data/tasks/e1_v1/inputs.jsonl": "".join(
            json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in visible
        ).encode(),
        "data/tasks/private/e1_v1/criteria.jsonl": "".join(
            json.dumps(v, ensure_ascii=False, sort_keys=True) + "\n" for v in private
        ).encode(),
    }
    manifest = {
        "suite_id": "aftersales-knowledge-tasks-e1-v1",
        "frozen_on": "2026-10-24",
        "task_count": 36,
        "counts_by_split": {s: 12 for s in ("dev", "validation", "test")},
        "corpus_hashes": corpus["files"],
        "files": {p: hashlib.sha256(b).hexdigest() for p, b in files.items()},
        "limitations": [
            "Evidence corpus and fact families shared across splits; held-out wording limited",
            "Simulated dates/conflicts are not official merchant policy history",
            "Assistant-authored rule-based scoring, not independent human semantic truth",
        ],
    }
    files["data/manifests/e1_v1_suite.json"] = dump(manifest)
    for p, b in files.items():
        target = root / p
        if target.exists() and target.read_bytes() != b:
            raise ValueError("Frozen E1 task differs")
    for p, b in files.items():
        target = root / p
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(b)
    return manifest


def load(root: Path, split: str):
    manifest = json.loads((root / "data/manifests/e1_v1_suite.json").read_text("utf-8"))
    _, corpus = load_corpus(root)
    if manifest["corpus_hashes"] != corpus["files"]:
        raise ValueError("Knowledge version mismatch")
    data = []
    for rel, sha in manifest["files"].items():
        raw = (root / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("E1 task hash mismatch")
        data.append([json.loads(line) for line in raw.splitlines()])
    inputs, criteria = data
    private = {c["task_id"]: c for c in criteria}
    selected = [
        {"public": c, "criteria": private[c["task_id"]]} for c in inputs if c["split"] == split
    ]
    if len(inputs) != 36 or len(criteria) != 36 or len(selected) != 12:
        raise ValueError("Task count mismatch")
    return selected, manifest
