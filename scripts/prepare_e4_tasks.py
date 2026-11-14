"""Rebuild E4 authoring materials from reviewed fact families; never overwrite a sealed suite."""

import itertools
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from aftersalesagent.knowledge.corpus import dump, load_corpus

ROOT = Path(__file__).resolve().parents[1]

FAMILIES = [
    {
        "family_id": "D01",
        "split": "dev",
        "challenge": "condition",
        "questions": [
            "床垫送给别人后，十年质保能跟着转给新主人吗？",
            "二手接手宜家床垫的人，是否自动继承原买家的质保权益？",
            "床垫原始购买者把床垫转让了，品质保证也能转让吗？",
        ],
        "references": [
            ("mattress::本品质保证旨在保障产品原始购买者的权益，不可转让"),
        ],
        "semantic_checklist": [
            "仅保障原始购买者",
            "不可转让",
        ],
        "required_patterns": [
            "不可转让|不能转让|不.*转让",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D02",
        "split": "dev",
        "challenge": "exception",
        "questions": [
            "儿童床垫、沙发床床垫能直接套用这份床垫十年质保吗？",
            "买的是小孩用的床垫或沙发里的床垫，也在普通床垫保证范围内吗？",
            "普通床垫条款是否包括儿童床垫和沙发床床垫？",
        ],
        "references": [
            "mattress::本品质保证不适用于床褥、沙发床床垫、儿童床垫",
        ],
        "semantic_checklist": [
            "明确两类床垫被排除",
            "不能因同属床垫就承诺质保",
        ],
        "required_patterns": [
            "不适用|不包括|不涵盖|不在",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D03",
        "split": "dev",
        "challenge": "identifier",
        "questions": [
            "KOMPLEMENT 首饰插件和抽屉垫也在衣柜质保内吗？",
            "康普蒙系列的首饰收纳插件、抽屉里面的垫子，算免费保证部件吗？",
            ("PAX 配件里，KOMPLEMENT 首饰插件与抽屉垫有没有被排除？"),
        ],
        "references": [
            "wardrobe::KOMPLEMENT 康普蒙 首饰插件",
        ],
        "semantic_checklist": [
            "首饰插件及抽屉垫均被排除",
        ],
        "required_patterns": [
            "不涵盖|不包括|排除|不在",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D04",
        "split": "dev",
        "challenge": "paraphrase",
        "questions": [
            "PAX 衣柜安装后需不需要固定在墙上？",
            "帕克思大衣柜摆稳了就行，还是必须上墙固定？",
            "按保养组装要求，PAX 框架应怎样固定，墙面不同要注意什么？",
        ],
        "references": [
            "wardrobe::并将衣柜框架固定在墙上",
        ],
        "semantic_checklist": [
            "按说明固定到墙上",
            "不同墙材需不同固定件",
        ],
        "required_patterns": [
            "墙",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D05",
        "split": "dev",
        "challenge": "exception",
        "questions": [
            "拆掉 SKYTTA 地轨胶带后地板留印，属于滑门质保吗？",
            "斯基塔移门的地面胶条撕下来伤了地板，可以按滑门保证赔吗？",
            ("SKYTTA 安装胶带移除造成地板表面损坏，这份品质保证是否涵盖？"),
        ],
        "references": [
            ("sliding::本品质保证不涵盖取下胶带后地板表面留下的印记或损坏"),
        ],
        "semantic_checklist": [
            "胶带移除所致地板印记或损坏不涵盖",
        ],
        "required_patterns": [
            "不涵盖|不包括|不属于|不在",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D06",
        "split": "dev",
        "challenge": "paraphrase",
        "questions": [
            "SKYTTA 轨道和脚轮平时怎样保养，才能避免开合不顺？",
            "斯基塔推拉门底部总积灰，按说明该怎么维护滚轮和地轨？",
            "滑门日常护理中，SKYTTA 地面轨道和脚轮为什么需要定期清洁？",
        ],
        "references": [
            "sliding::定期清洁地面轨道和脚轮",
        ],
        "semantic_checklist": [
            "定期清洁地轨与脚轮",
            "保障开合并避免脚轮损坏",
        ],
        "required_patterns": [
            "清洁|清理",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D07",
        "split": "dev",
        "challenge": "exception",
        "questions": [
            "OPPEJEN 淋浴隔断的滑片和门密封条，也保十年吗？",
            "奥佩耶恩淋浴门本体十年，那密封胶条和滑片能保多久？",
            "OPPEJEN 的门密封条、部件滑片是否同样享受十年保证？",
        ],
        "references": [
            "bathroom::部件滑片和门密封条不适用10年品质保证",
        ],
        "semantic_checklist": [
            "这两类部件为三年",
            "不套用十年",
        ],
        "required_patterns": [
            "3|三",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D08",
        "split": "dev",
        "challenge": "condition",
        "questions": [
            "花洒被水中渣滓损坏，淋浴配件三年质保会涵盖吗？",
            "水里的杂质把宜家淋浴配件弄坏了，也算保证范围吗？",
            "浴室淋浴配件因水中渣滓受损，能否按三年质保直接认定覆盖？",
        ],
        "references": [
            (
                "bathroom::本品质保证不涵盖正常的磨损、划痕和刮擦，"
                "以及由撞击、意外或水中渣滓造成的损坏"
            ),
        ],
        "semantic_checklist": [
            "水中渣滓所致损坏不涵盖",
            "不承诺直接批准",
        ],
        "required_patterns": [
            "不涵盖|不包括|不属于|排除",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D09",
        "split": "dev",
        "challenge": "identifier",
        "questions": [
            "电动沙发的框架、电机和开关分别保几年？",
            "宜家电动沙发的动力装置和开关，能跟普通沙发一样算十年吗？",
            "电动沙发系列中，框架、电机、开关的保证期限是多少？",
        ],
        "references": [
            "seating::电动沙发框架，电机以及开关质保5年",
        ],
        "semantic_checklist": [
            "列出的电动沙发部件为五年",
            "不能套普通十年",
        ],
        "required_patterns": [
            "5|五",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D10",
        "split": "dev",
        "challenge": "exception",
        "questions": [
            "客厅沙发的布套、真皮套和人造革套都在框架质保里吗？",
            "宜家沙发外面的皮套或布套坏了，是否跟框架享有相同品质保证？",
            "这份客厅家具保证是否涵盖布艺、真皮、人造革沙发套？",
        ],
        "references": [
            "seating::布艺沙发套\n真皮沙发套\n人造革沙发套",
        ],
        "semantic_checklist": [
            "三类沙发套均排除",
            "不能把框架保证泛化到套",
        ],
        "required_patterns": [
            "不涵盖|不包括|排除|不在",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D11",
        "split": "dev",
        "challenge": "multi_evidence",
        "questions": [
            "METOD 与 KNOXHULT 厨房系列的质保分别几年？",
            "米多和诺克胡两套厨房柜子，保证年限一样吗？",
            ("比较 METOD 米多和 KNOXHULT 诺克胡厨房系列的保证期限。"),
        ],
        "references": [
            "kitchen::METOD 米多 厨房系列 享有25年品质保证",
            "kitchen::5年品质保证\n本品质保证涵盖KNOXHULT",
        ],
        "semantic_checklist": [
            "METOD 二十五年",
            "KNOXHULT 五年",
        ],
        "required_patterns": [
            "25|二十五",
            "5|五",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D12",
        "split": "dev",
        "challenge": "multi_evidence",
        "questions": [
            "室内厨房水龙头与 KALLSJÖN 户外水龙头保证年限一样吗？",
            "卡尔松户外龙头是不是也能按室内厨房龙头保十年？",
            "比较室内厨房混合型水龙头和 KALLSJÖN 的质保期限。",
        ],
        "references": [
            ("kitchen::室内厨房混合型水龙头的品质保证自购买之日起十(10)年有效"),
            ("kitchen::KALLSJÖN 卡尔松 户外混合型水龙头享有两(2)年品质保证"),
        ],
        "semantic_checklist": [
            "室内十年",
            "卡尔松户外两年",
            "保留家庭等条款条件",
        ],
        "required_patterns": [
            "10|十",
            "2|两|二",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D13",
        "split": "dev",
        "challenge": "exception",
        "questions": [
            "一般宜家家电和 TILLREDA、LAGAN 的质保各几年？",
            "提瑞达、拉根的小家电能不能也按普通家电保五年？",
            ("厨房电器保证期限中，TILLREDA 和 LAGAN 有什么例外？"),
        ],
        "references": [
            (
                "kitchen::TILLREDA 提瑞达 和LAGAN 拉"
                "根 系列家电的品质保证自最初购买之日起两(2)年内有效"
            ),
        ],
        "semantic_checklist": [
            "普通家电五年",
            "两个例外系列两年",
        ],
        "required_patterns": [
            "5|五",
            "2|两|二",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D14",
        "split": "dev",
        "challenge": "identifier",
        "questions": [
            ("FÖRDUBBLA、UPPFYLLD、SMÅBIT 刀具享受这份十五年免费质保吗？"),
            "福度布拉、乌普菲尔德、斯莫比这些刀，是否也包含在长质保里？",
            ("厨房刀具条款对 FÖRDUBBLA、UPPFYLLD、SMÅBIT 有哪些排除？"),
        ],
        "references": [
            "knives::产品名称为FÖRDUBBLA 福度布拉",
        ],
        "semantic_checklist": [
            "这三个系列被排除",
            "不把大部分刀具泛化为所有",
        ],
        "required_patterns": [
            "不在|不涵盖|不包括|排除",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D15",
        "split": "dev",
        "challenge": "exception",
        "questions": [
            "带不粘涂层的锅，能享受无涂层炊具的十五年保证吗？",
            "宜家不粘锅涂层坏了，可直接套用锅具十五年质保吗？",
            "锅具有不粘涂层时，这份十五年免费质保是否适用？",
        ],
        "references": [
            "cookware::带不粘涂层的宜家锅具不在此品质保证范围内",
        ],
        "semantic_checklist": [
            "带不粘涂层锅具排除",
            "不能承诺涂层受保",
        ],
        "required_patterns": [
            "不在|不适用|不涵盖|不包括",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D16",
        "split": "dev",
        "challenge": "condition",
        "questions": [
            "带铜质表面的不锈钢锅要怎么洗才能维持保证有效？",
            "铜外层的不锈钢炊具能像普通不锈钢锅一样随便进洗碗机吗？",
            "铜表面不锈钢锅的质保条件里，对清洗方式有什么要求？",
        ],
        "references": [
            "cookware::带铜质表面的不锈钢锅具、铸铁",
        ],
        "semantic_checklist": [
            "为保证有效必须手洗",
            "区分普通不锈钢",
        ],
        "required_patterns": [
            "手洗",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D17",
        "split": "dev",
        "challenge": "condition",
        "questions": [
            "这份办公家具保证只限家用，还是办公和公共场合也适用？",
            ("公司会议室使用的清单内宜家办公家具，会因不是居家使用而自动失去保证吗？"),
            "对于列明的办公家具，居家、办公及公共场合的适用范围是什么？",
        ],
        "references": [
            ("office::此品质保证适用于在居家环境和办公及公共场合使用的产品"),
        ],
        "semantic_checklist": [
            "覆盖列明的居家及办公公共环境",
            "仍受具体条款约束",
        ],
        "required_patterns": [
            "办公",
            "公共",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D18",
        "split": "dev",
        "challenge": "identifier",
        "questions": [
            ("ROTHULT 智能锁和 EILIF 屏风，在办公家具保证清单里吗？"),
            "罗胡特智能锁、艾丽福屏风是不是办公家具十年保证的例外？",
            "办公家具条款是否明确排除了 ROTHULT 和 EILIF？",
        ],
        "references": [
            ("office::• ROTHULT 罗胡特 智能锁\n• EILIF 艾丽福 屏风"),
        ],
        "semantic_checklist": [
            "两者明确排除",
            "不扩大到全部办公用品",
        ],
        "required_patterns": [
            "不涵盖|不包括|排除|不在",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D19",
        "split": "dev",
        "challenge": "no_answer",
        "questions": [
            "订单 CN-E4-901 的床垫维修报价具体多少元？",
            ("帮我查一下个人订单 CN-E4-901 这次修床垫实际要交多少钱。"),
            ("只根据这些公开政策，能确定 CN-E4-901 的具体维修账单金额吗？"),
        ],
        "references": [],
        "semantic_checklist": [
            "公开资料没有该订单账单",
            "不编造金额",
            "可说明需联系商场核实",
        ],
        "required_patterns": [],
        "expected_status": "insufficient_evidence",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "D20",
        "split": "dev",
        "challenge": "date_scope",
        "questions": [
            "请按2010年当时有效的宜家床垫政策回答保证年限。",
            "2010年的床垫质保究竟几年？只采用当时有效资料。",
            "以2010-01-01为准，当前快照能证明当时床垫保证多久吗？",
        ],
        "references": [],
        "semantic_checklist": [
            "未知生效日期的快照不能证明历史适用",
            "保留资料不足",
        ],
        "required_patterns": [],
        "expected_status": "insufficient_evidence",
        "filters": {
            "scope": "ikea_cn",
            "as_of": "2010-01-01",
        },
    },
    {
        "family_id": "V01",
        "split": "validation",
        "challenge": "condition",
        "questions": [
            "家用床垫严重下陷或把手损坏，属于条款列明的缺陷类型吗？",
            ("床垫不是普通磨损，而是严重凹陷、把手损坏，材料工艺保证有没有提到？"),
            "床垫材料工艺保证列举的情形中，有没有严重下陷和把手损坏？",
        ],
        "references": [
            ("mattress::裂缝、弹簧损坏、把手损坏、床垫架损坏和严重下陷"),
        ],
        "semantic_checklist": [
            "家用前提",
            "列举包含严重下陷和把手损坏",
            "是否批准仍需检验",
        ],
        "required_patterns": [
            "下陷|凹陷",
            "把手",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V02",
        "split": "validation",
        "challenge": "identifier",
        "questions": [
            "PAX 的合叶门、合叶、滑门机件是否属于列明保证组件？",
            "帕克思衣柜的门铰链和推拉门机械件，算衣柜条款里的部件吗？",
            "衣柜配件保证清单包括 PAX 合叶门及滑门机件吗？",
        ],
        "references": [
            "wardrobe::• 合叶门及合叶\n• 滑门、滑门机件",
        ],
        "semantic_checklist": [
            "清单包括这些组件",
            "家庭使用下材料工艺缺陷",
            "不承诺任意损坏",
        ],
        "required_patterns": [
            "合叶|铰链",
            "滑门|推拉门",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V03",
        "split": "validation",
        "challenge": "paraphrase",
        "questions": [
            "SKYTTA 应固定在天花板和地板上吗，不同天花板材料怎么办？",
            "斯基塔移门安装时，上下两端要固定在哪里，配件能一概通用吗？",
            ("按 SKYTTA 组装说明，天花板和地面怎么固定，天花板材质变化有何影响？"),
        ],
        "references": [
            "sliding::将滑门固定在天花板和地板上",
        ],
        "semantic_checklist": [
            "固定到天花板与地板",
            "不同天花板需不同安装件",
            "疑问咨询专家",
        ],
        "required_patterns": [
            "天花板",
            "地板|地面",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V04",
        "split": "validation",
        "challenge": "multi_evidence",
        "questions": [
            "浴室洗脸池和手持花洒、淋浴管，保证年限都是十年吗？",
            "洗脸盆与淋浴软管是不是同一个质保期限？",
            "对比浴室洗脸池和手持花洒、淋浴管的免费保证期限。",
        ],
        "references": [
            ("bathroom::浴室洗脸池，混合水龙头和恒温淋浴水混合器10年品质保证"),
            "bathroom::适用于混合型水龙头的ÅBÄCKEN",
        ],
        "semantic_checklist": [
            "洗脸池十年",
            "手持花洒及淋浴管等配件三年",
        ],
        "required_patterns": [
            "10|十",
            "3|三",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V05",
        "split": "validation",
        "challenge": "identifier",
        "questions": [
            "斯德哥尔摩沙发系列的保证期限和购买凭证要求是什么？",
            "STOCKHOLM 斯德哥尔摩沙发保多久，需要原始订单信息吗？",
            "购买斯德哥尔摩沙发后，从何时算品质保证，凭什么证明购买？",
        ],
        "references": [
            "seating::品质保证25年内有效",
        ],
        "semantic_checklist": [
            "二十五年",
            "自购买日起",
            "提供原始订单信息",
        ],
        "required_patterns": [
            "25|二十五",
            "订单",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V06",
        "split": "validation",
        "challenge": "date_scope",
        "questions": [
            "2025年买的 BESTÅ 能直接适用页面注明的新十年保证吗？",
            ("贝达储物柜购于2025年，新条款写2026年1月1日之后购买，这能直接套用吗？"),
            ("现有 BESTÅ 十年条款的购买日期限制是什么，2025年买的是否满足？"),
        ],
        "references": [
            ("seating::注意：该质保仅适用于2026年1月1日之后购买的产品"),
        ],
        "semantic_checklist": [
            "不满足注明的购买日期前提",
            "不能据此断言无任何其他权利",
        ],
        "required_patterns": [
            "2026",
            "不满足|不适用|不能直接|不能.*适用|不符合",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V07",
        "split": "validation",
        "challenge": "condition",
        "questions": [
            "厨房部件停产后两年的添置服务，有库存限制吗？",
            "柜门停产不到两年，宜家是否保证一定有替换件可买？",
            "厨房部分部件的两年售后添置服务，是无条件供货还是取决于库存？",
        ],
        "references": [
            "kitchen::只要有库存，你可以为你的厨房添置柜门",
        ],
        "semantic_checklist": [
            "只针对部分部件",
            "停产后最多两年且有库存",
            "不保证必有货",
        ],
        "required_patterns": [
            "库存",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V08",
        "split": "validation",
        "challenge": "condition",
        "questions": [
            "家电质保期间修过或换了新部件，保证期限会重新起算吗？",
            "宜家家电换了零件后，会自动再获得一整段新质保吗？",
            "质保维修是否延长家电本体或新部件的质保期？",
        ],
        "references": [
            ("kitchen::在质保期内享受质保服务，不会延长家电或新部件的质保期"),
        ],
        "semantic_checklist": [
            "不会延长原家电或新部件质保期",
        ],
        "required_patterns": [
            "不会延长|不延长|不.*重新|不能延长",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V09",
        "split": "validation",
        "challenge": "paraphrase",
        "questions": [
            "厨房刀具正常家用保证，对清洗和磨刀有什么要求？",
            "宜家厨刀的十五年保证能否省掉手洗和定期磨刀这些保养步骤？",
            "刀具条款如何定义正常使用，是否包括手洗与定期打磨？",
        ],
        "references": [
            ("knives::正常使用是指一天一次地使用刀具（切食物）及手洗刀具，以及定期磨刀"),
        ],
        "semantic_checklist": [
            "正常使用包含一天一次切食物、手洗和定期磨刀",
            "不能视为免保养",
        ],
        "required_patterns": [
            "手洗",
            "磨刀|打磨",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V10",
        "split": "validation",
        "challenge": "condition",
        "questions": [
            "锅具十五年保证里，正常使用频率是如何定义的？",
            "餐厅一天反复用锅几十次，能当作条款定义的一般家用频率吗？",
            "无不粘涂层锅具的正常使用，是否指每天烹饪清洗一次？",
        ],
        "references": [
            "cookware::正常使用是指每天使用锅具进行烹饪并清洗一次",
        ],
        "semantic_checklist": [
            "正常家用每天烹饪清洗一次",
            "不泛化到餐厅高频专业用途",
        ],
        "required_patterns": [
            "每天|每日",
            "一次|1次",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V11",
        "split": "validation",
        "challenge": "identifier",
        "questions": [
            "碳钢锅用烤箱开锅，说明写多少度、多久？",
            "新买的宜家碳钢炊具放烤箱养锅，应设什么温度和时间？",
            "碳钢煎锅采用烤箱开锅步骤时，预热温度与单次加热时长是多少？",
        ],
        "references": [
            "cookware::预热烤箱至200°C（392°F）",
        ],
        "semantic_checklist": [
            "二百度",
            "约三十到四十分钟",
            "按说明冷却并重复，不混用铸铁步骤",
        ],
        "required_patterns": [
            "200|二百",
            "30.*40|三十.*四十",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V12",
        "split": "validation",
        "challenge": "paraphrase",
        "questions": [
            "办公家具组装后多久要再拧紧螺丝，以后还要检查吗？",
            "办公室桌柜装好用过一阵子，螺丝应该怎样复检？",
            "这份办公家具保养说明对两周后的螺丝和每年检查有什么要求？",
        ],
        "references": [
            "office::使用两周后拧紧螺丝，每年需检查数次",
        ],
        "semantic_checklist": [
            "使用两周后拧紧",
            "每年检查数次",
        ],
        "required_patterns": [
            "两周|2周|二周",
            "每年",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V13",
        "split": "validation",
        "challenge": "no_answer",
        "questions": [
            "没给出系列和损坏原因，能保证我家所有沙发都免费换新吗？",
            "我只说买过一个宜家沙发，能直接承诺任何问题都换新吗？",
            ("沙发具体型号、购买信息和故障原因都未知，能直接确认免费换新品资格吗？"),
        ],
        "references": [],
        "semantic_checklist": [
            "缺少产品与故障条件",
            "不承诺所有沙发或任意故障免费换新",
            "需核实",
        ],
        "required_patterns": [],
        "expected_status": "insufficient_evidence",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "V14",
        "split": "validation",
        "challenge": "date_scope",
        "questions": [
            "2026-07-01，E4演示商店补寄配件的运费由谁承担？",
            ("只按2026年7月1日有效的模拟条款，E4演示商店补寄配件是谁付运费？"),
            ("E4演示商店在2026年7月1日补寄配件，旧版顾客付邮费的条款还适用吗？"),
        ],
        "references": [
            "e4_accessory::运费由商家承担",
        ],
        "semantic_checklist": [
            "七月一日起商家承担",
            "不沿用六月旧版",
            "声明项目模拟",
        ],
        "required_patterns": [
            "商家",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "demo_store",
            "as_of": "2026-07-01",
            "category": "e4_accessory",
        },
    },
    {
        "family_id": "V15",
        "split": "validation",
        "challenge": "conflict",
        "questions": [
            "E4冲突商店对包装破损商品的退货期限到底是7天还是14天？",
            "模拟资料里包装破损商品有七天和十四天两种退货期限，该怎么回答？",
            "E4冲突商店退货期限A、B都有效，包装破损商品应按哪个期限？",
        ],
        "references": [
            "e4_return::退货期限条款A",
            "e4_return::退货期限条款B",
        ],
        "semantic_checklist": [
            "指出同范围同日期的七天与十四天冲突",
            "引用两条",
            "不擅自择一",
            "声明模拟",
        ],
        "required_patterns": [
            "7|七",
            "14|十四",
        ],
        "expected_status": "conflict",
        "filters": {
            "scope": "demo_conflict",
            "category": "e4_return",
            "as_of": "2026-11-01",
        },
    },
    {
        "family_id": "T01",
        "split": "test",
        "challenge": "paraphrase",
        "questions": [
            "新床垫身体适应期多久，卷叠包装一般几天恢复形状？",
            "刚拆的床垫有点硬，需要多久适应，卷包床垫多久能恢复？",
            "床垫第一次使用说明对身体适应和卷叠后恢复分别建议多长时间？",
        ],
        "references": [
            "mattress::你可以用一个月的时间让你的身体适应床垫",
            ("mattress::使用3-4天后，卷叠包装床垫即可恢复原来的形状"),
        ],
        "semantic_checklist": [
            "适应约一个月",
            "卷叠床垫三到四天恢复",
            "不把它说成绝对不能立即使用",
        ],
        "required_patterns": [
            "一个月|1个月|一月",
            "3.*4|三.*四",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T02",
        "split": "test",
        "challenge": "paraphrase",
        "questions": [
            "PAX 衣柜日常能用什么清洁剂和布擦拭？",
            "帕克思衣柜怎么清洁，擦完后需要再用干布吗？",
            "根据衣柜保养说明，PAX 表面清洁是否应使用中性清洁剂再擦干？",
        ],
        "references": [
            "wardrobe::用布块沾中性清洁剂充分擦洗,再用干布擦拭即可",
        ],
        "semantic_checklist": [
            "中性清洁剂配布擦洗",
            "再用干布擦拭",
        ],
        "required_patterns": [
            "中性",
            "干布",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T03",
        "split": "test",
        "challenge": "condition",
        "questions": [
            "SKYTTA 未按组装说明改装后坏了，保证是否适用？",
            "斯基塔滑门自己改造、没有照说明书做，能直接按质保处理吗？",
            "滑门条款是否排除不按组装说明进行的改装？",
        ],
        "references": [
            ("sliding::本品质保证不适用于未按组装说明所述方式改装的产品"),
        ],
        "semantic_checklist": [
            "不按组装说明改装的产品不适用",
            "不直接批准",
        ],
        "required_patterns": [
            "不适用|不涵盖|不包括|排除",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T04",
        "split": "test",
        "challenge": "identifier",
        "questions": [
            "ÅBÄCKEN 喷雾水嘴是十年还是三年免费保证？",
            "阿贝肯水龙头喷雾嘴是不是也跟洗脸池一样保十年？",
            "浴室条款给 ÅBÄCKEN 阿贝肯喷雾水嘴多少年质保？",
        ],
        "references": [
            ("bathroom::适用于混合型水龙头的ÅBÄCKEN 阿贝肯 喷雾水嘴"),
        ],
        "semantic_checklist": [
            "喷雾水嘴三年",
            "区分十年洗脸池或混合器",
        ],
        "required_patterns": [
            "3|三",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T05",
        "split": "test",
        "challenge": "exception",
        "questions": [
            "POÄNG 靠垫、藤竹座椅和矮凳都在其他客厅家具保证内吗？",
            "波昂靠垫、天然藤条竹子做的座椅，也按普通沙发十年条款保吗？",
            ("其他沙发椅凳系列的排除项里，有没有 POÄNG 靠垫和天然纤维座椅？"),
        ],
        "references": [
            ("seating::POÄNG 波昂 靠垫\n采用藤条、竹子或其他天然纤维制作的座椅类家具"),
        ],
        "semantic_checklist": [
            "列明的靠垫与天然纤维家具及矮凳排除",
        ],
        "required_patterns": [
            "不涵盖|不包括|排除|不在",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T06",
        "split": "test",
        "challenge": "identifier",
        "questions": [
            ("BESTÅ BURS 和贝达灰色毛毡储物盒，有没有被贝达保证排除？"),
            "贝达伯斯系列、灰色毛毡收纳盒也能用 BESTÅ 的保证吗？",
            ("BESTÅ 新保证条款不涵盖的产品中，是否包括 BURS 和灰色毛毡盒？"),
        ],
        "references": [
            ("seating::• BESTÅ 贝达 灰色毛毡储物盒\n• BESTÅ BURS 贝达 伯斯 系列"),
        ],
        "semantic_checklist": [
            "两者被明确排除",
            "不扩展系列覆盖",
        ],
        "required_patterns": [
            "不涵盖|不包括|排除|不在",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T07",
        "split": "test",
        "challenge": "identifier",
        "questions": [
            ("SUNNERSTA 和 FYNDIG 水槽能用厨房25年、10年或5年保证吗？"),
            "苏纳思厨房、芬蒂格水槽是不是三档厨房保证的例外？",
            ("厨房长保证的排除清单对 SUNNERSTA、FYNDIG 和把手怎么规定？"),
        ],
        "references": [
            ("kitchen::拉钮、把手、SUNNERSTA 苏纳思 厨房系列和FYNDIG 芬蒂格 水槽"),
        ],
        "semantic_checklist": [
            "所列产品不享三档保证",
            "不把METOD部件泛化",
        ],
        "required_patterns": [
            "不享|不涵盖|不包括|排除|不在",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T08",
        "split": "test",
        "challenge": "condition",
        "questions": [
            "家电自己运回家途中损坏，与宜家配送途中损坏，条款如何区分？",
            "电器运输中坏了，自提和宜家送货的责任一样吗？",
            "宜家家电保证对顾客自行运输及宜家配送造成的损坏分别怎么说？",
        ],
        "references": [
            "kitchen::如果顾客自行将产品运送回家或其他地址",
        ],
        "semantic_checklist": [
            "自行运输损坏宜家不负责",
            "宜家配送损坏宜家负责但不在品质保证中",
            "不能宣称已认定实际交易责任",
        ],
        "required_patterns": [
            "自行|自提|自己",
            "宜家",
            "不.*品质保证|不.*质保|保证.*之外",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T09",
        "split": "test",
        "challenge": "condition",
        "questions": [
            "宜家厨刀能用不锈钢磨刀器吗，应选什么材料？",
            "磨宜家厨房刀时，是不是随便买个不锈钢磨刀棒就行？",
            "刀具保养对磨刀器硬度和陶瓷、金刚石、油石有什么要求？",
        ],
        "references": [
            "knives::需要使用陶瓷、金刚石或油石制成的磨刀器",
        ],
        "semantic_checklist": [
            "磨刀器材质要更硬",
            "陶瓷金刚石油石可选",
            "不能用不锈钢磨刀器",
        ],
        "required_patterns": [
            "陶瓷|金刚石|油石",
            "不.*不锈钢|切勿.*不锈钢",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T10",
        "split": "test",
        "challenge": "condition",
        "questions": [
            "带木制部件的锅具上油后，隔多久再上第二次油？",
            "锅的木头把手想做护理，上完一遍植物油后要等多长时间？",
            "炊具木制部件保养是否要求擦掉多余油，24小时后再上油？",
        ],
        "references": [
            "cookware::为保护木制部件免受油脂和尘埃侵蚀",
        ],
        "semantic_checklist": [
            "可接触食物的油",
            "擦除多余油",
            "二十四小时后再上一次",
        ],
        "required_patterns": [
            "24|二十四",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T11",
        "split": "test",
        "challenge": "condition",
        "questions": [
            "普通铸铁锅能用洗洁精清洗吗，为什么？",
            "没有搪瓷层的铸铁炊具，日常是否建议只用热水而不用清洁剂？",
            "铸铁保养条款怎么说清洁剂对表面油脂层的影响？",
        ],
        "references": [
            ("cookware::请勿使用清洁剂，否则会导致锅具材料干燥，并去除铸铁表面所需的油脂层"),
        ],
        "semantic_checklist": [
            "普通铸铁只用热水不使用清洁剂",
            "会去油层",
            "不混同搪瓷铸铁",
        ],
        "required_patterns": [
            "不.*清洁剂|请勿.*清洁剂|避免.*清洁剂",
            "油",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T12",
        "split": "test",
        "challenge": "identifier",
        "questions": [
            "MITTZON 的办公桌、房间隔板和隔音屏是不是列明的办公家具？",
            "米特丛办公桌与房间分隔、吸音屏是否在办公产品保证名单中？",
            "办公家具清单对 MITTZON 米特丛列出了哪些产品类型？",
        ],
        "references": [
            ("office::MITTZON 米特丛 办公桌、桌子、专注工作区、房间隔板和隔音屏"),
        ],
        "semantic_checklist": [
            "列明办公桌桌子专注工作区房间隔板隔音屏",
            "仍受具体条款",
        ],
        "required_patterns": [
            "隔板|分隔",
            "隔音|吸音",
        ],
        "expected_status": "answered",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T13",
        "split": "test",
        "challenge": "no_answer",
        "questions": [
            "能给出我这次投诉获批赔偿的精确概率吗，比如百分之几？",
            "这些质保资料能算出个人投诉成功率的准确百分比吗？",
            "没有投诉处理记录，只看政策能确定我的赔偿批准概率吗？",
        ],
        "references": [],
        "semantic_checklist": [
            "资料没有个案统计或审批结论",
            "不编造百分比",
        ],
        "required_patterns": [],
        "expected_status": "insufficient_evidence",
        "filters": {
            "scope": "ikea_cn",
        },
    },
    {
        "family_id": "T14",
        "split": "test",
        "challenge": "date_scope",
        "questions": [
            "请以2099年有效政策判断厨房混合型水龙头质保。",
            "2099-01-01时宜家室内厨房龙头还能保证十年吗？",
            "现有快照能证明2099年当时有效的厨房龙头保证规则吗？",
        ],
        "references": [],
        "semantic_checklist": [
            "快照未知生效区间不能证明2099年政策",
            "不能把当前期限延伸到未来",
        ],
        "required_patterns": [],
        "expected_status": "insufficient_evidence",
        "filters": {
            "scope": "ikea_cn",
            "as_of": "2099-01-01",
        },
    },
    {
        "family_id": "T15",
        "split": "test",
        "challenge": "conflict",
        "questions": [
            "E4冲突商店未拆封商品退回，运费究竟由顾客还是商家承担？",
            ("模拟运费条款A让顾客付，B让商家付，未拆封商品退货能直接定谁付吗？"),
            "E4冲突商店两条同样适用的退回运费规则相反，该如何告知用户？",
        ],
        "references": [
            "e4_shipping::运费条款A",
            "e4_shipping::运费条款B",
        ],
        "semantic_checklist": [
            "说明顾客承担和商家承担相冲突",
            "引用两条",
            "不擅自确定",
            "声明模拟",
        ],
        "required_patterns": [
            "顾客",
            "商家",
        ],
        "expected_status": "conflict",
        "filters": {
            "scope": "demo_conflict",
            "category": "e4_shipping",
            "as_of": "2026-11-01",
        },
    },
]


def main():
    public_dir = ROOT / "data/tasks/e4_v1"
    private_dir = ROOT / "data/tasks/private/e4_v1"
    if public_dir.exists() or private_dir.exists():
        raise SystemExit("E4 tasks already exist; no implicit overwrite")
    chunks, _ = load_corpus(ROOT, "e4_v1")
    rows = []
    for item in FAMILIES:
        if len(item["questions"]) != 3:
            raise ValueError("Every family needs three authored phrasings")
        alternatives = []
        source_review = []
        for reference in item["references"]:
            category, quote = reference.split("::", 1)
            matches = [c for c in chunks if c.category == category and quote in c.text]
            if not matches:
                raise ValueError(
                    "No complete source span for " + item["family_id"] + ": " + reference
                )
            alternatives.append([c.chunk_id for c in matches])
            source_review.append(
                {
                    "quote": quote,
                    "spans": [
                        {
                            "id": c.chunk_id,
                            "path": c.source_text_path,
                            "start": c.source_start,
                            "end": c.source_end,
                        }
                        for c in matches
                    ],
                }
            )
        groups = (
            sorted({tuple(sorted(set(group))) for group in itertools.product(*alternatives)})
            if alternatives
            else []
        )
        criteria = {
            k: v
            for k, v in item.items()
            if k not in {"questions", "references", "filters", "split"}
        }
        criteria.update(
            {
                "support_groups": groups,
                "source_review": source_review,
                "reviewer_kind": "assistant",
                "review_status": "source_checked",
                "forbidden_patterns": [],
                "not_independent_human_gold": True,
            }
        )
        for n, question in enumerate(item["questions"], 1):
            rows.append(
                {
                    "public": {
                        "task_id": f"E4-{item['family_id']}-{n}",
                        "question": question,
                        "filters": item["filters"],
                    },
                    "criteria": criteria,
                    "split": item["split"],
                }
            )
    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    for split in ("dev", "validation", "test"):
        cases = [row for row in rows if row["split"] == split]
        (public_dir / (split + ".jsonl")).write_text(
            "".join(json.dumps(c["public"], ensure_ascii=False) + "\n" for c in cases), "utf-8"
        )
        (private_dir / (split + ".jsonl")).write_text(
            "".join(
                json.dumps({"task_id": c["public"]["task_id"], **c["criteria"]}, ensure_ascii=False)
                + "\n"
                for c in cases
            ),
            "utf-8",
        )
    pairs = []
    for left, right in itertools.combinations(rows, 2):
        if left["split"] == right["split"]:
            continue
        a = re.sub(r"[^\w\u4e00-\u9fff]", "", left["public"]["question"].lower())
        b = re.sub(r"[^\w\u4e00-\u9fff]", "", right["public"]["question"].lower())
        score = SequenceMatcher(None, a, b).ratio()
        if score >= 0.72:
            pairs.append(
                {
                    "left": left["public"]["task_id"],
                    "right": right["public"]["task_id"],
                    "similarity": score,
                    "status": "needs_review",
                }
            )
    report = {
        "families": len(FAMILIES),
        "tasks": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "challenge_counts": dict(Counter(r["criteria"]["challenge"] for r in rows)),
        "cross_split_near_duplicates": pairs,
        "independence_note": (
            "Three phrasings per fact family; not 150 independent facts"
            ". Shared domain primitives remain. Assistant-authored and "
            "reviewed, not independent human gold."
        ),
    }
    (private_dir / "review.json").write_bytes(dump(report))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
