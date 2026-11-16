"""Freeze 24 authored business cases from public simulated orders, never hidden tasks."""

import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".local/databases/retail_template.sqlite"
OUT = ROOT / "data/results/retail_agent_rag_v1/scenarios_v1.json"

# Distinct business questions; items and payment instruments are read at runtime.
PROMPTS = {
    "pending": [
        ("取消不再需要的订单", "请取消这笔订单，原因是我不再需要了。", "cancelled"),
        (
            "修改完整地址",
            "请把这笔订单送到 10 Pine Street, Apt 2, Seattle, WA 98101, USA。",
            "pending",
        ),
        ("待处理订单申请退货", "我想退这笔还没送到的订单，可以直接申请退货吗？", "pending"),
        ("不允许的取消原因", "请取消这笔订单，原因是我想得到赔偿。", "pending"),
        (
            "修改地址前核对状态",
            "现在还能修改这笔订单的收货地址吗？先解释条件，不要修改。",
            "pending",
        ),
        ("待处理订单部分退货", "只想退这笔订单里的第一件商品，可以现在提交退货申请吗？", "pending"),
    ],
    "processed": [
        ("已处理订单取消", "请取消这笔订单，原因是我不再需要了；不行请解释依据。", "processed"),
        ("已处理订单改址", "这笔订单已经处理了，能改收货地址吗？", "processed"),
        ("已处理订单退货", "还没收到商品，能现在申请退货并退款吗？", "processed"),
        ("已处理订单改支付", "这笔已处理订单还可以换一张卡付款吗？", "processed"),
        ("已处理订单商品规格", "把这笔订单中的商品换成另一种颜色，现在能办理吗？", "processed"),
        ("已处理订单转人工", "这笔订单已处理，我有运送争议，请帮我转人工。", "processed"),
    ],
    "delivered": [
        (
            "已送达部分退货",
            "这笔订单已收到，想退第一件商品，退款用原支付方式。",
            "return requested",
        ),
        ("退货新银行卡", "想退货，退款能打到一张新银行卡吗？不行请解释。", "delivered"),
        ("退货到账时间未知", "申请退货后，退款一定几天到账？请给依据，不要替我提交。", "delivered"),
        ("已送达订单取消", "商品已经收到，我仍想直接取消原订单，能办吗？", "delivered"),
        ("未实现的换货", "把买的商品换成同款另一规格，能自动帮我办理吗？", "delivered"),
        ("损坏赔偿未知", "刚收到发现商品损坏，能赔多少钱？请给依据。", "delivered"),
    ],
    "cancelled": [
        ("取消后重复取消", "这笔订单已经取消了，还能再取消一遍吗？", "cancelled"),
        ("取消退款渠道", "这笔已取消的订单，原支付方式退款政策是什么？", "cancelled"),
        ("取消退款时效边界", "这笔取消订单的退款时效是否和退货退款一样？", "cancelled"),
        ("已取消订单改址", "订单取消后，还能修改原来的收货地址吗？", "cancelled"),
        ("已取消订单退货", "已取消订单还能提交退货申请吗？", "cancelled"),
        ("已取消订单重新付款", "这笔已取消的订单能通过改支付方式恢复吗？", "cancelled"),
    ],
}


def main():
    if OUT.exists():
        raise SystemExit("Frozen task file exists; never silently overwrite")
    cases = []
    with sqlite3.connect(f"file:{TEMPLATE.as_posix()}?mode=ro", uri=True) as db:
        for status, prompts in PROMPTS.items():
            rows = db.execute(
                "select o.order_id,u.first_name,u.last_name,u.address_json "
                "from orders o join users u on o.user_id=u.user_id "
                "where o.status=? order by o.order_id limit 6",
                (status,),
            ).fetchall()
            if len(rows) != 6:
                raise ValueError("Not enough independent public orders")
            for number, ((title, request, expected), row) in enumerate(
                zip(prompts, rows, strict=True), 1
            ):
                order_id, first, last, raw_address = row
                zip_code = json.loads(raw_address)["zip"]
                split = "dev" if number <= 3 else "test"
                cases.append(
                    dict(
                        id=f"R{len(cases) + 1:02}",
                        split=split,
                        title=title,
                        order_id=order_id,
                        initial_status=status,
                        expected_order_status=expected,
                        user_message=(
                            f"我是 {first} {last}，邮编 {zip_code}。"
                            f"订单 {order_id}。{request}"
                        ),
                        requires_policy=True,
                    )
                )
    if len(cases) != 24 or {s: sum(c["split"] == s for c in cases) for s in ("dev", "test")} != {
        "dev": 12,
        "test": 12,
    }:
        raise AssertionError("Split error")
    material = dict(
        source="public tau2 retail order database; project-authored tasks",
        cases=cases,
        knowledge_version="retail_sim_v1",
        exclusions=["hidden user simulator", "private evaluation criteria"],
    )
    material["sha256"] = hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(material, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(dict(total=len(cases), dev=12, test=12, sha256=material["sha256"]))


if __name__ == "__main__":
    main()
