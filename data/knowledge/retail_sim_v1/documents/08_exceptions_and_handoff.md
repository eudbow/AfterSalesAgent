<!-- {"sources": ["Retail agent policy", "Exchange delivered order", "Modify payment", "Application protocol", "Implemented capabilities"], "applies_to": "异常与转人工"} -->
# 异常、未知政策与模拟转人工

## 适用范围
Retail 模拟商家：异常与转人工。

## 条件与限制
对不符合政策的请求应拒绝；能力范围外请求使用模拟转人工。当前没有依据确认退货运费、手续费、包装要求、品类禁退、赔偿或退货到账时间；资料未规定不等于免费、无限期或无条件允许。

## 信息、步骤与结果
说明已知事实和缺失信息，需要时调用现有转人工工具记录摘要。返回 handoff_requested 只代表模拟交接请求已记录，不代表真人接通。不承诺照片上传、物流理赔或邮件送达。

## 常见问题与边界
问：商品破损可以赔多少钱？答：当前资料没有赔偿规则，不能编造金额；可说明能力限制并请求模拟人工处理。

## 来源
见首行来源章节映射及 coverage.json；origin=derived_guidance，不能覆盖原规则。
