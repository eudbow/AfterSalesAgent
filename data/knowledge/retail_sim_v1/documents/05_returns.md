<!-- {"sources": ["Return delivered order", "Application protocol", "Implemented capabilities"], "applies_to": "退货"} -->
# 退货申请与办理结果

## 适用范围
Retail 模拟商家：退货。

## 条件与限制
退货订单必须 delivered。需要订单号、要退的 item_id 列表和退款方式。退款只能到原支付方式或已有礼品卡；应用还校验支付方式属于当前用户，商品数量不超订单内容。

## 信息、步骤与结果
核验身份并查询订单，确认所有退货商品与退款方式，准备提议并等待独立确认。成功后状态是 return requested，保存退货商品和退款方式，不等于商品已寄回或退款到账。上游提及邮件，但当前执行函数未发送邮件，不能声称邮件已发送。

## 常见问题与边界
问：只退其中两件可以吗？答：需提供属于该订单的具体规格及数量，由后端校验；提交申请不代表退款完成。

## 来源
见首行来源章节映射及 coverage.json；origin=derived_guidance，不能覆盖原规则。
