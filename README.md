# AfterSalesAgent

基于 FastAPI 和 LangGraph 构建的单 Agent 电商售后应用。

支持订单查询、取消订单、修改地址、申请退货、检索增强的知识咨询、MCP 工具和版本化工作流 Skill。订单变更需要用户明确确认。

本仓库包含源码、测试、配置、脚本、网页界面和项目数据，不包含 API 凭据、运行时数据库、日志及内部开发文档。

应用使用模拟零售数据，不连接真实商家系统。部分评测脚本需要额外的本地模型文件或环境配置。

零售来源材料来自 [tau2-bench](https://github.com/sierra-research/tau2-bench) 的固定版本快照，保留了其原始[许可证](data/upstream/tau2-bench/LICENSE)和来源清单。第三方材料保留原始归属说明及其适用条款。
