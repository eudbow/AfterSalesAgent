# Retail 模拟售后知识库 v1

来源为固定 tau2 v1.0.1 政策与当前应用源码。不是现实商家政策。八份中文文档为助手核对的派生说明，未经独立人工审核。没有新增执行资格或自定义赔偿政策。

构建：`.venv/Scripts/python.exe scripts/prepare_retail_corpus.py`。
审计：`.venv/Scripts/python.exe scripts/audit_retail_corpus.py`。

来源区间按 Python UTF-8 read_text 的换行归一后字符索引 [start,end) 计算；文件哈希按原始字节计算。生效日期未知，不能用构建日期替代。

rules.jsonl 是章节级规则记录，不能将记录数宣称为独立原子规则数。coverage.json 为 Chunk 到原规则的旁表。每篇文档保留完整条件，FAQ 不单独重复切块。未实现自动关联补充：后续调用方可按旁表加载规范条款。

限制：尚未接入订单 Agent；BM25 召回不保证答案正确或自动拒答。原始英文政策块用于来源查阅，中文跨语言召回主要依赖派生文档。特殊改址状态存在原文与实现差异，已在文档记录；未修业务代码。退款基于首条原支付记录的实现细节需在未来多支付场景继续核对。转人工仅模拟记录，退货没有实际邮件发送。
