<!-- {"sources": ["Product", "Order", "Generic action rules", "Modify items", "Exchange delivered order", "Implemented capabilities"], "applies_to": "商品、状态解释"} -->
# 商品规格与订单状态

## 适用范围
Retail 模拟商家：商品、状态解释。

## 条件与限制
product_id 标识商品类型，item_id 标识具体规格，两者无关联不能混用。规格包含选项、可用性和价格；实时数值由工具读取。基础状态包括 pending、processed、delivered、cancelled；操作还可产生 return requested 等状态。

## 信息、步骤与结果
退货时按订单中的 item_id 确认规格和数量，不能凭商品名称替换标识。上游换货或修改规格只允许同商品不同选项，必须收集全部变更项且注意一次性限制；当前应用未实现这些写操作。

## 常见问题与边界
问：能把衣服换成鞋吗？答：上游换货不允许更换商品类型，且当前应用没有自动换货能力。

## 来源
见首行来源章节映射及 coverage.json；origin=derived_guidance，不能覆盖原规则。
