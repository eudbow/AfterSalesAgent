<!-- {"sources": ["Modify pending order", "Modify items", "Application protocol", "Implemented capabilities"], "applies_to": "改址"} -->
# 修改订单收货地址

## 适用范围
Retail 模拟商家：改址。

## 条件与限制
上游一般要求 pending。应用地址参数包括 address1、address2、city、state、country、zip，其中 address2 可以为空。必须收集完整字段，不能猜地址。

## 信息、步骤与结果
查询订单并检查状态，收集地址后准备提议，确认后更新地址。实现还接受 pending (item modified)，而上游商品修改章节写 pending (items modifed) 并限制后续修改：这是待处理差异，不能自动归一或承诺此状态可办理。

## 常见问题与边界
问：改订单地址会修改默认地址吗？答：当前工具修改该订单地址；不能宣称同时修改用户默认地址。

## 来源
见首行来源章节映射及 coverage.json；origin=derived_guidance，不能覆盖原规则。
