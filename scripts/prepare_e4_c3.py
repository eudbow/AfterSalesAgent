"""Author C3 materials before any model output; immutable old assets remain intact."""

import itertools
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from aftersalesagent.evaluation.e4.execution import sha
from aftersalesagent.knowledge.corpus import KnowledgeChunk, dump, load_corpus

ROOT = Path(__file__).resolve().parents[1]

# Every family: split, primary challenge, source support paths, checklist, two phrasings.
FAMILIES = [
    (
        "D01",
        "identifier",
        [["kitchen-013"]],
        ["ENHET厨房和UTRUSTA网篮均列在10年部分"],
        [
            "ENHET厨房与UTRUSTA网篮，在厨房条款中分别属于几年保证？",
            "安纳特厨房和乌斯塔网篮能直接套用米多的二十五年吗，原文把它们列在哪档？",
        ],
    ),
    (
        "D02",
        "identifier",
        [["bathroom-007"]],
        ["BROGRUND浴用搁架在3年淋浴配件范围，需原始收据"],
        [
            "BROGRUND浴用搁架属于三年淋浴配件保证吗，需要什么凭证？",
            "布鲁格隆德的浴用搁架能用淋浴配件保证吗，申请时要留什么票据？",
        ],
    ),
    (
        "D03",
        "identifier",
        [["wardrobe-004"]],
        ["KOMPLEMENT托盘分割件和储物盒明确不涵盖"],
        [
            "KOMPLEMENT托盘分割件和储物盒属于衣柜保证范围吗？",
            "康普蒙托盘里的分割件和储物盒，是不是买了衣柜就一起保？",
        ],
    ),
    (
        "D04",
        "paraphrase",
        [["mattress-007"]],
        ["不要折叠床垫，可能损坏弹簧和内里"],
        [
            "搬家时能把宜家床垫对折捆起来吗，保养说明担心什么损伤？",
            "床垫太大过不了门，弯折收起来会伤到里面哪些东西？",
        ],
    ),
    (
        "D05",
        "paraphrase",
        [["cookware-010"], ["cookware-012"], ["cookware-016"]],
        ["不锈钢内壁锅应水沸后放盐，冷水加盐可能盐渍腐蚀"],
        [
            "不锈钢内壁锅煮水，盐应该冷水时放还是烧开后放，为什么？",
            "不锈钢锅水还凉时先撒盐，会留下什么问题，说明建议什么时候加？",
        ],
    ),
    (
        "D06",
        "paraphrase",
        [["knives-011"]],
        ["硬食物来回划动，不左右摇晃；木制/塑料砧板，不玻璃金属陶瓷"],
        [
            "宜家厨刀切较硬食物应怎样运刀，能用玻璃砧板吗？",
            "硬东西切不动能左右掰刀吗，下面垫玻璃板和木板有区别吗？",
        ],
    ),
    (
        "D07",
        "condition",
        [["bathroom-004"]],
        ["浴室洗脸池/混合水龙头公共环境不在保证范围"],
        [
            "浴室混合水龙头装在公共场所，能照搬家庭使用的保证吗？",
            "给公共洗手间装的浴室龙头，也能直接按家用浴室条款免费保证吗？",
        ],
    ),
    (
        "D08",
        "condition",
        [["wardrobe-006"]],
        ["PAX室外或潮湿环境不在范围"],
        [
            "PAX衣柜放在室外或潮湿浴室，品质保证是否适用？",
            "帕克思摆到露天或经常潮湿的位置，损坏后还能按普通家用保证处理吗？",
        ],
    ),
    (
        "D09",
        "condition",
        [["kitchen-029"]],
        ["家电连接不合适电压、生锈/腐蚀等不涵盖，不承诺具体认定"],
        [
            "宜家家电接错电压或发生腐蚀，条款是否把这些列为排除项？",
            "厨房电器连了不适合的电压，或因腐蚀坏掉，原文有排除吗？",
        ],
    ),
    (
        "D10",
        "multi_evidence",
        [["cookware-027", "cookware-028"]],
        ["碳钢炉灶开锅依次约10分钟/10分钟/6分钟，分轮冷却薄油"],
        [
            "碳钢锅在炉灶上开锅，三轮各大约多久，中间要如何处理？",
            "用灶台给碳钢锅养第一层油膜，三遍分别加热几分钟，每遍结束怎么办？",
        ],
    ),
    (
        "D11",
        "multi_evidence",
        [["knives-011", "knives-012"]],
        ["木刀柄食用接触油、擦余油24小时再涂；不得长时间泡水"],
        [
            "木制刀柄怎样上油、多久补第二次，能长期泡水吗？",
            "木头刀把抹油后要等多长时间再抹，洗完一直浸在水里行不行？",
        ],
    ),
    (
        "D12",
        "no_answer",
        [],
        ["不能给出未提供售后工单的确定预约时间"],
        [
            "只凭这些资料，工单C3-APPOINT-01的上门预约是几月几日几点？",
            "帮我查C3-APPOINT-01师傅准确几点到，当前资料有没有这个预约记录？",
        ],
    ),
    (
        "D13",
        "no_answer",
        [],
        ["其他商家品牌不在本资料范围，不套宜家条款"],
        [
            "其他商家ALPHA品牌床垫的保证期限是多少，当前宜家资料能确认吗？",
            "我买的是别家ALPHA床垫，能从这份知识库确定它保几年吗？",
        ],
    ),
    (
        "D14",
        "date_scope",
        [["SIM-DATE-D-B"]],
        ["项目模拟，7月1日起包装留存建议60天"],
        [
            "2026-07-01，C3演示商店建议把包装保留多少天？",
            "按七月一日有效的C3模拟包装说明，外包装建议留多久？",
        ],
    ),
    (
        "D15",
        "conflict",
        [["SIM-CONFLICT-D-A", "SIM-CONFLICT-D-B"]],
        ["项目模拟，预约提前2天和5天矛盾，不自行选定"],
        [
            "C3冲突商店上门预约应提前2天还是5天？",
            "C3模拟资料对上门预约提前量有两种说法，应如何说明？",
        ],
    ),
    (
        "V01",
        "identifier",
        [["kitchen-004"]],
        ["MAXIMERA抽屉与UTRUSTA搁板列在METOD25年部分"],
        [
            "MAXIMERA全拉伸抽屉和UTRUSTA搁板属于METOD的25年部件吗？",
            "米多厨房的马斯麦抽屉、乌斯塔搁板，原文是否列入二十五年清单？",
        ],
    ),
    (
        "V02",
        "identifier",
        [["office-003"], ["chairs-001", "chairs-002"]],
        ["FLINTAN和SMÖRKULL都列明办公椅"],
        [
            "FLINTAN与SMÖRKULL是否都在办公椅保证清单中？",
            "福通椅和斯莫库勒椅有没有被这份办公家具保证点名列出？",
        ],
    ),
    (
        "V03",
        "identifier",
        [["kitchen-013"]],
        ["VADHOLMA与TORNVIKEN厨柜家具在10年部分"],
        [
            "VADHOLMA和TORNVIKEN厨柜及家具在厨房条款中保几年？",
            "瓦德侯玛、托恩维肯厨柜是按厨房文件的哪档年限保证？",
        ],
    ),
    (
        "V04",
        "paraphrase",
        [["cookware-014"]],
        ["铜绿由铜与水氧化学反应，不等同磨损，可去除"],
        [
            "铜质锅表面发暗出现铜绿，说明认为一定是磨损吗？",
            "铜锅用久颜色暗了、起铜绿，就能认定铜已经磨坏了吗？",
        ],
    ),
    (
        "V05",
        "paraphrase",
        [["cookware-025"]],
        ["碳钢锅柄已涂漆无需上油，内侧开锅外侧偶尔薄油"],
        [
            "碳钢煎锅开锅时外侧和手柄都要像内侧一样涂油吗？",
            "给碳钢锅养油膜，锅外面和已经上漆的把手也得每次涂吗？",
        ],
    ),
    (
        "V06",
        "paraphrase",
        [["kitchen-042"]],
        ["吸油烟机条码不要撕便于维修核实"],
        [
            "吸油烟机上的产品条码能撕掉吗，维修说明为什么要求保留？",
            "烟机那张条码贴纸不好看，撕了会影响售后人员核对什么？",
        ],
    ),
    (
        "V07",
        "condition",
        [["office-006"], ["chairs-005"]],
        ["办公家具需正确组合按说明组装，涵盖材料工艺功能缺陷"],
        [
            "办公家具的材料、工艺和功能保证，对组合与组装方式有什么前提？",
            "办公桌椅拼装方式随意也能直接套用功能缺陷保证吗？",
        ],
    ),
    (
        "V08",
        "condition",
        [["kitchen-010"]],
        ["厨房家具固定墙上、承重需足、墙面螺丝不含包装，不确定找专业人员"],
        [
            "厨房家具上墙对承重墙有什么要求，包装是否附墙面固定螺丝？",
            "装厨房柜子能随便选墙吗，固定到墙里的螺丝是不是箱内一定有？",
        ],
    ),
    (
        "V09",
        "condition",
        [["knives-007"]],
        ["刀具外观改变通常排除，严重影响功能例外"],
        [
            "宜家厨刀只有外观改变但功能正常，就属于保证缺陷吗，例外是什么？",
            "刀身样子变了还能用，外观变化都保吗，什么程度需要区别看？",
        ],
    ),
    (
        "V10",
        "multi_evidence",
        [["cookware-017", "cookware-020"]],
        ["普通铸铁初次开锅不超过150度至少1小时新锅3次；搪瓷铸铁热水清洁剂手洗不用钢丝球"],
        [
            "普通铸铁新锅开锅的温度时长次数，与搪瓷铸铁日常清洗方法分别是什么？",
            "我有一口新铸铁锅和一口搪瓷铸铁锅，前者怎么首次开锅，后者用什么洗、能用钢丝球吗？",
        ],
    ),
    (
        "V11",
        "multi_evidence",
        [["kitchen-033", "kitchen-034"]],
        ["家电售后拨随附文档号码、预读说明、准备8位货号，收据或铭牌可找"],
        [
            "家电售后联系哪个号码，事先准备的货品编号几位、在哪里找？",
            "厨房电器要报修，先看哪份电话说明，客服要的八位编号从哪里抄？",
        ],
    ),
    (
        "V12",
        "no_answer",
        [],
        ["无法确定缺少物流记录的具体退款到账日期"],
        [
            "退款单C3-REFUND-08准确哪一天到账，当前资料能查到吗？",
            "资料里有没有C3-REFUND-08的银行入账日期，能给确定日期吗？",
        ],
    ),
    (
        "V13",
        "no_answer",
        [],
        ["官方资料没有个人优惠券有效期记录"],
        [
            "个人优惠券C3-COUPON-12具体哪天失效，质保资料能确认吗？",
            "帮我确定账户中C3-COUPON-12券的到期日，这些文件有没有记录？",
        ],
    ),
    (
        "V14",
        "date_scope",
        [["SIM-DATE-V-A"]],
        ["项目模拟，8月31日自提预约费10元"],
        [
            "2026-08-31，C3演示商店自提预约费是多少？",
            "按八月三十一日的C3模拟规则，自提预约应付多少费用？",
        ],
    ),
    (
        "V15",
        "conflict",
        [["SIM-CONFLICT-V-A", "SIM-CONFLICT-V-B"]],
        ["项目模拟，加急配送费顾客/商家承担矛盾"],
        [
            "C3冲突商店加急配送费用到底由顾客还是商家承担？",
            "C3模拟加急配送规则对承担费用的人写得不一样，能直接确定一方吗？",
        ],
    ),
    (
        "T01",
        "identifier",
        [["kitchen-020"]],
        ["KNOXHULT UTRUSTA电动推进开门器列5年"],
        [
            "KNOXHULT使用的UTRUSTA电动推进式开门器属于哪档保证？",
            "诺克胡厨房配乌斯塔电动开门器，是这份文件的五年部件吗？",
        ],
    ),
    (
        "T02",
        "identifier",
        [["bathroom-001"]],
        ["HAVBÄCK与TÄNNFORSEN浴室系列列10年材料工艺保证"],
        [
            "HAVBÄCK和TÄNNFORSEN浴室系列的材料工艺保证是几年？",
            "哈弗贝克、坦弗森浴室家具，文件给了哪档品质保证？",
        ],
    ),
    (
        "T03",
        "identifier",
        [["kitchen-035"]],
        ["嵌入式燃气灶6年、人工煤气1年，易损1年；不是厨房25年"],
        [
            "嵌入式燃气灶与人工煤气情形的包修期限分别多久？",
            "按快照中的燃气灶维修规定，嵌入式灶都是六年吗，人工煤气有区别吗？",
        ],
    ),
    (
        "T04",
        "paraphrase",
        [["cookware-009"]],
        ["不锈钢锅清洗前彻底冷却、冷底微凹热后回平避免不平"],
        [
            "不锈钢锅洗之前要等凉吗，冷时底部微凹就说明坏了吗？",
            "钢锅放凉后底有点凹，为什么说明建议彻底冷却再洗？",
        ],
    ),
    (
        "T05",
        "paraphrase",
        [["knives-008"]],
        ["厨刀第一次使用前清洁冲洗晾干"],
        [
            "新宜家厨刀拆封后第一次使用前，应做哪些清洁处理？",
            "新刀拿出包装能马上切菜吗，使用前说明要求先做什么？",
        ],
    ),
    (
        "T06",
        "paraphrase",
        [["cookware-026"]],
        ["碳钢开锅后烹饪仍要油但较不锈钢少；随使用变深黑正常"],
        [
            "碳钢锅开锅后做饭还要放油吗，逐渐变黑是否正常？",
            "养好油膜的碳钢锅能完全不放油炒菜吗，用久变黑是怎么回事？",
        ],
    ),
    (
        "T07",
        "condition",
        [["seating-014"]],
        ["BESTA仅家庭用途原始购买者不可转让"],
        [
            "BESTÅ储物系列转卖给别人，保证能转给新主人吗？",
            "贝达储物柜二手转让后，新买家能继承原购买者的保证吗？",
        ],
    ),
    (
        "T08",
        "condition",
        [["kitchen-030"]],
        ["家电初始安装费用不包括，不能把维修重装混为初装"],
        [
            "家电最初安装的费用包括在这份品质保证里吗？",
            "买完厨房电器第一次装机，能直接要求用质保报销初装费吗？",
        ],
    ),
    (
        "T09",
        "condition",
        [["kitchen-036"]],
        ["燃气灶有效购机证明不可涂改、登记型号机身码应匹配"],
        [
            "燃气灶购机证明涂改，或安装卡与机器编码不符，会影响包修吗？",
            "灶具的购买票据改过，安装服务卡型号机身码还对不上，原文怎么规定？",
        ],
    ),
    (
        "T10",
        "multi_evidence",
        [
            ["office-005", "office-010"],
            ["chairs-004", "chairs-009"],
            ["office-005", "chairs-009"],
            ["chairs-004", "office-010"],
        ],
        ["办公家具10年需原收据，部分储物柜固定墙，墙用螺丝不附包装"],
        [
            "办公家具保证期限和凭证是什么，部分储物柜固定上墙的螺丝由谁准备？",
            "办公储物柜一方面保几年要留什么票，另一方面上墙螺丝箱内有没有？",
        ],
    ),
    (
        "T11",
        "multi_evidence",
        [["knives-009", "cookware-009"], ["knives-009", "cookware-004"]],
        ["厨刀建议手洗洗碗机可能损刀腐蚀，普通不锈钢锅可洗碗机"],
        [
            "宜家厨刀和普通不锈钢锅，对放洗碗机的建议相同吗？",
            "一把厨刀一口不锈钢锅，能一概扔进洗碗机吗，分别有什么依据？",
        ],
    ),
    (
        "T12",
        "no_answer",
        [],
        ["文件没有指定商场当日营业时间"],
        [
            "当前快照能确定我今天去的宜家商场几点关门吗？",
            "没有提供门店名称和营业安排，这些质保文件能告诉我今晚准确闭店时间吗？",
        ],
    ),
    (
        "T13",
        "no_answer",
        [],
        ["没有用户地址及门店距离信息，不能确定最近店及距离"],
        [
            "按我未提供的住址，最近宜家距离我家精确多少公里？",
            "这些资料能直接确定离我家最近的宜家是哪家、几公里吗？",
        ],
    ),
    (
        "T14",
        "date_scope",
        [["SIM-DATE-T-B"]],
        ["项目模拟，10月1日起咨询邮箱help-new@example.invalid"],
        [
            "2026-10-01，C3演示商店咨询资料应使用哪个邮箱？",
            "按十月一日有效的C3模拟说明，咨询邮件发新邮箱还是旧邮箱？",
        ],
    ),
    (
        "T15",
        "conflict",
        [["SIM-CONFLICT-T-A", "SIM-CONFLICT-T-B"]],
        ["项目模拟，退回方式仅门店/仅邮寄互斥，不能单选"],
        [
            "C3冲突商店商品退回必须到店还是必须邮寄？",
            "C3模拟文件对退回方式要求互相矛盾，能直接选到店或邮寄吗？",
        ],
    ),
]


def main():
    artifact = ROOT / ".local/e4_c3_artifacts"
    artifact.mkdir(exist_ok=True)
    if not (artifact / "historical_baseline.json").exists():
        paths = [
            p
            for folder in ("data", "skills", "configs/evaluation", ".local/e4_runs")
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
        ]
        paths += [
            ROOT / ".local/e4_artifacts/generation_budget.sqlite",
            ROOT / ".local/e4_artifacts/dev_gate.json",
            ROOT / ".local/e4_artifacts/vectors.json",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
        (artifact / "historical_baseline.json").write_bytes(
            dump({p.relative_to(ROOT).as_posix(): sha(p) for p in paths})
        )
    target = ROOT / "data/knowledge/e4_c3_v1"
    if target.exists():
        raise ValueError("C3 materials already exist; no overwrite")
    target.mkdir()
    chunks, _ = load_corpus(ROOT, "e4_v1")
    authored = []
    # Explicit, non-factual document index names avoid carrying METOD 25y into all kitchen sections.
    title = {"kitchen": "厨房产品品质保证资料", "bathroom": "浴室产品品质保证资料"}
    for old in chunks:
        if old.origin != "official":
            continue
        if old.chunk_id in {
            "E4-kitchen-032",
            "E4-kitchen-043",
            "E4-bathroom-009",
            "E4-sliding-007",
        }:
            continue  # orphan/navigation, not substantive evidence
        section = old.section
        if old.category == "kitchen":
            n = int(old.chunk_id.rsplit("-", 1)[1])
            scope = (
                "METOD 米多厨房系列25年保证"
                if n <= 12
                else "ENHET等厨房系列10年保证"
                if n == 13
                else "厨房混合型水龙头保证"
                if n <= 19
                else "KNOXHULT 诺克胡厨房系列5年保证"
                if n <= 21
                else "厨房家电保证"
                if n <= 34
                else "家用燃气灶维修服务"
                if n <= 38
                else "吸油烟机维修服务"
            )
            section = title["kitchen"] + " / " + scope
        else:
            parts = section.split(" / ")
            section = " / ".join(dict.fromkeys(parts))
        authored.append(
            old.model_copy(
                update={
                    "chunk_id": old.chunk_id.replace("E4-", "C3-"),
                    "section": section,
                    "version": "c3-" + old.version,
                }
            )
        )
    simulation = []
    for split, category, a, b, boundary in [
        ("D", "packaging", "建议包装保留30天", "建议包装保留60天", "2026-07-01"),
        ("V", "pickup", "自提预约费10元", "自提预约费0元", "2026-09-01"),
        (
            "T",
            "email",
            "咨询邮箱help-old@example.invalid",
            "咨询邮箱help-new@example.invalid",
            "2026-10-01",
        ),
    ]:
        from datetime import date, timedelta

        until = (date.fromisoformat(boundary) - timedelta(days=1)).isoformat()
        for label, text, start, end in [("A", a, "2026-01-01", until), ("B", b, boundary, None)]:
            simulation.append(
                (
                    f"C3-SIM-DATE-{split}-{label}",
                    f"[项目模拟，不是宜家政策] C3演示商店{category}：{text}。",
                    "demo_store",
                    category,
                    start,
                    end,
                )
            )
    for split, category, a, b in [
        ("D", "appointment", "上门预约必须提前2天", "上门预约必须提前5天"),
        ("V", "express", "加急配送费由顾客承担", "加急配送费由商家承担"),
        ("T", "return_mode", "商品只能到门店退回，不接受邮寄", "商品只能邮寄退回，不接受到店"),
    ]:
        for label, text in [("A", a), ("B", b)]:
            simulation.append(
                (
                    f"C3-SIM-CONFLICT-{split}-{label}",
                    f"[项目模拟，不是宜家政策] C3冲突商店{category}：{text}。",
                    "demo_conflict",
                    category,
                    "2026-01-01",
                    None,
                )
            )
    text = "\n".join(row[1] for row in simulation) + "\n"
    src = target / "simulations.txt"
    src.write_text(text, "utf-8")
    offset = 0
    import hashlib

    for cid, body, scope, cat, start, end in simulation:
        authored.append(
            KnowledgeChunk(
                chunk_id=cid,
                document_id="c3_" + cat,
                section="C3项目模拟 / " + cat,
                text=body,
                version="c3-sim-v1",
                source="project://e4-c3/" + cat,
                scope=scope,
                origin="simulated",
                category=cat,
                effective_from=start,
                effective_until=end,
                source_text_path=src.relative_to(ROOT).as_posix(),
                source_start=offset,
                source_end=offset + len(body),
                source_excerpt_sha256=hashlib.sha256(body.encode()).hexdigest(),
            )
        )
        offset += len(body) + 1
    authored.sort(key=lambda c: c.chunk_id)
    (target / "chunks.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in authored), "utf-8"
    )
    by_id = {c.chunk_id: c for c in authored}
    sources = {c.source_text_path: sha(ROOT / c.source_text_path) for c in authored}
    corpus = {
        "version": "e4_c3_v1",
        "parent": "e4_v1",
        "files": {**sources, "data/knowledge/e4_c3_v1/chunks.jsonl": sha(target / "chunks.jsonl")},
        "official_chunks": sum(c.origin == "official" for c in authored),
        "simulated_chunks": 12,
        "title_rule": (
            "Index/section labels are reviewed context, not verbatim source; never apply "
            "a global duration to unrelated subsections"
        ),
    }
    (target / "manifest.json").write_bytes(dump(corpus))
    public_dir = ROOT / "data/tasks/e4_c3_v1"
    private_dir = ROOT / "data/tasks/private/e4_c3_v1"
    public_dir.mkdir()
    private_dir.mkdir()
    rows = []
    for family, challenge, groups, checklist, questions in FAMILIES:
        split = {"D": "dev", "V": "validation", "T": "test"}[family[0]]
        converted = [["C3-" + cid for cid in group] for group in groups]
        for group in converted:
            for cid in group:
                assert cid in by_id, cid
        status = (
            "insufficient_evidence"
            if challenge == "no_answer"
            else "conflict"
            if challenge == "conflict"
            else "answered"
        )
        filters = {"scope": "ikea_cn"}
        if challenge in ("date_scope", "conflict"):
            exemplar = by_id[converted[0][0]]
            filters = {
                "scope": exemplar.scope,
                "category": exemplar.category,
                "as_of": {"D14": "2026-07-01", "V14": "2026-08-31", "T14": "2026-10-01"}.get(
                    family, "2026-11-01"
                ),
            }
        review = [
            {
                "id": cid,
                "path": by_id[cid].source_text_path,
                "start": by_id[cid].source_start,
                "end": by_id[cid].source_end,
                "quote": by_id[cid].text,
            }
            for cid in sorted({cid for group in converted for cid in group})
        ]
        for n, q in enumerate(questions, 1):
            pub = {"task_id": f"C3-{family}-{n}", "question": q, "filters": filters}
            gold = {
                "task_id": pub["task_id"],
                "family_id": family,
                "challenge": challenge,
                "expected_status": status,
                "support_groups": converted,
                "semantic_checklist": checklist,
                "source_review": review,
                "reviewer_kind": "assistant",
                "required_patterns": [],
                "forbidden_patterns": [],
                "not_independent_human_gold": True,
            }
            rows.append({"public": pub, "criteria": gold, "split": split})
    for split in ("dev", "validation", "test"):
        part = [r for r in rows if r["split"] == split]
        assert len(part) == 30
        for directory, key in ((public_dir, "public"), (private_dir, "criteria")):
            (directory / f"{split}.jsonl").write_text(
                "".join(json.dumps(r[key], ensure_ascii=False) + "\n" for r in part), "utf-8"
            )
    overlaps = []
    for left, right in itertools.combinations(rows, 2):
        if left["split"] == right["split"]:
            continue
        similarity = SequenceMatcher(
            None, left["public"]["question"], right["public"]["question"]
        ).ratio()
        if similarity >= 0.72:
            overlaps.append(
                {
                    "left": left["public"]["task_id"],
                    "right": right["public"]["task_id"],
                    "similarity": similarity,
                }
            )
    old = [
        json.loads(line)
        for split in ("dev", "validation", "test")
        for line in (ROOT / f"data/tasks/e4_v2/{split}.jsonl").read_text("utf-8").splitlines()
    ]
    nearest = []
    for row in rows:
        matches = [
            (SequenceMatcher(None, row["public"]["question"], o["question"]).ratio(), o["task_id"])
            for o in old
        ]
        score, oid = max(matches)
        nearest.append(
            {"task": row["public"]["task_id"], "nearest_old_task": oid, "similarity": score}
        )
    report = {
        "tasks": 90,
        "families": 45,
        "splits": {"dev": 30, "validation": 30, "test": 30},
        "challenges": dict(Counter(r["criteria"]["challenge"] for r in rows)),
        "cross_split_lexical_overlap": overlaps,
        "nearest_historical_questions": nearest,
        "limitations": [
            "Same merchant/source domain, shared warranty primitives; not wholly unseen knowledge",
            "Assistant authored and reviewed, no independent human gold",
            "Within-family phrasings are correlated; report family counts",
            "New compositions/components from old sources; not 45 guaranteed semantically "
            "independent facts",
        ],
    }
    (private_dir / "review.json").write_bytes(dump(report))
    files = [
        p for folder in (target, public_dir, private_dir) for p in folder.rglob("*") if p.is_file()
    ]
    manifest = {
        "suite": "e4_c3_v1",
        "files": {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
        "splits": report["splits"],
        "families": 45,
        "challenges": report["challenges"],
        "reviewer_kind": "assistant",
    }
    (ROOT / "data/manifests/e4_c3_v1.json").write_bytes(dump(manifest))
    print(
        json.dumps(
            {"tasks": 90, "families": 45, "chunks": len(authored), "cross_split_flags": overlaps},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
