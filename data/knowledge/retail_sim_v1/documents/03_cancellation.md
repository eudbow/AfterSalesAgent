<!-- {"sources": ["Cancel pending order", "Application protocol", "Implemented capabilities"], "applies_to": "取消"} -->
# 取消订单与取消退款

## 适用范围
Retail 模拟商家：取消。

## 条件与限制
只有 pending 订单可取消。原因只允许 no longer needed（不再需要）或 ordered by mistake（误下单）。需确认订单号和原因。上游规定取消后 cancelled，原渠道退款：礼品卡即时，其他方式 5–7 个工作日。这是取消政策，不是退货到账规定。

## 信息、步骤与结果
核验身份、查询状态、收集有效原因，准备确认卡。独立确认后以工具结果说明是否取消。模拟后端追加退款记录并回补礼品卡余额，不代表真实银行支付已到账。

## 常见问题与边界
问：订单已经处理了还能取消吗？答：processed 不满足 pending 条件；不能承诺取消成功。

## 来源
见首行来源章节映射及 coverage.json；origin=derived_guidance，不能覆盖原规则。
