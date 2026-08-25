# Changelog

本项目版本记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)。

## [Unreleased]

## [0.5.0] - 2026-07-21

### Added
- MCP 工具链路（`tools.provider: local | mcp`）：银行工具由 MCP server（FastMCP）
  提供，agent 侧经 MCP 协议消费，LangGraph 图零改动；`local` 进程内直调保留为默认
  （离线确定性测试不变），与 LLM provider 同一套配置切换哲学
- 传输层双模式：stdio（agent 自动拉起 server 子进程，零配置）与 streamable HTTP
  （工具服务独立部署，`--http` 启动，支持 `--host/--port`）
- 信任边界设计：风险等级、最低角色、授权钩子取自 agent 侧本地 ToolSpec，不采信
  服务器元数据（MCP 注解按规范是 untrusted hints，不构成安全边界）；注册集合 =
  服务器能力 ∩ 本地信任声明——服务器多出的工具默认拒绝，本地声明而服务器缺失则
  启动即报错；`user_id` 等身份参数由框架注入，不进入 LLM 可见的参数 schema
- `mcp/client.py`：官方 async-only SDK 的同步桥接——后台事件循环线程以单任务
  持有整个会话（anyio cancel scope 要求同任务进出），同步 handler 经
  `run_coroutine_threadsafe` 提交；异步性封在单个文件内，图/服务层/测试保持同步
- `configs/config.mcp.yaml` 演示配置；`mock_bank` 业务操作抽成单一实现，
  本地 handler 与 MCP server 共用
- 测试 +5（共 19 条）：MCP 端到端（stdio 拉起真实子进程，验证越权在 agent 侧
  拦截、审批闸门不变、服务端账本生效）、服务端错误传播、信任交集注册、
  缺失工具 fail-fast、HTTP 传输链路

## [0.3.1] - 2026-07-13

### Added
- Ollama 原生 API 客户端（`provider: ollama`）：支持 `think` 开关控制推理模型
  思考模式（OpenAI 兼容端点会忽略该参数）。实测 qwen3.5:9b 关思考提速 3-10 倍
  但意图分类不可靠，默认保持开启，作为可配置的速度/质量权衡

### Fixed
- 缺参追问后用户的反问/补充会误触发其他工具（真实 LLM 下反问被误分类为余额查询
  并直接执行）：新增 `pending_tool` 跨轮状态，等参期间补充/反问继续原流程并合并
  参数，明确切换意图才放弃
- 参数抽取从助手历史回复中抄参数值（用户查 ACC-003 却抽出历史里的 ACC-001，
  掩盖越权拦截）：抽取上下文只保留用户消息，prompt 增加"不得猜测"约束
- 跨任务参数残留（转账后查余额，收款账户泄漏进查询参数）：本轮请求显式标记 +
  近因优先规则 + few-shot 示例，qwen3.5:9b 实测 3/3 稳定
- 模糊请求（"查一下账户"）不再默认执行余额查询：分类 prompt 收紧，
  chitchat 节点追问后结合上下文重新路由

### Changed
- 项目更名为 banking-agent-hitl（对齐 GitHub 仓库名；Python 包名保持 banking_agent）
- LangSmith 项目名改为配置驱动（`observability.langsmith_project`），
  测试中强制关闭 tracing

## [0.3.0] - 2026-07-10

### Added
- Ollama 本地推理配置 `configs/config.ollama.yaml`（qwen3.5:9b 实测全场景跑通：
  RAG 引用回答、拒答、余额查询、越权拦截、转账审批）
- CLI `--config` 参数支持指定配置文件
- OpenAI-compatible 客户端剥离推理模型的 `<think>` 思考块

## [0.2.0] - 2026-07-10

### Added
- 多轮对话记忆：服务层每轮从审计库加载最近 8 条历史注入所有 LLM 调用
  （意图分类、参数抽取、答案生成、闲聊）
- 低置信追问的上下文修复：追问后用户的补充与原问题合并检索
  （`pending_clarify_query` 经 checkpointer 跨轮持久化，追问后强制回到政策问答意图）
- JSON 解析容错：真实 LLM 输出夹带说明文字时提取首个对象字面量
- 意图标签解析容错：从 LLM 输出中匹配合法标签而非要求精确相等

## [0.1.1] - 2026-07-10

### Fixed
- staff 转账时付款账户被强制覆盖为 None 的 bug（只对 customer 强制本人账户出金）
- 对无待审批操作的会话调用 `resolve_approval` 不再崩溃，返回 error 状态（API 409）
- 审批挂起期间用户新消息不再被吞，改为重新提示待审批
- 配置加载时校验 `chunk_overlap < chunk_size` 与阈值不倒挂

### Removed
- 未使用的 `security.transaction_force_approval_amount` 配置（与"敏感操作一律人工审批"的需求矛盾）

## [0.1.0] - 2026-07-10

### Added
- 项目骨架：LLM 层（OpenAI-compatible 覆盖 OpenAI/vLLM/Ollama + Mock）、RAG 层
  （FAISS + 三态判定：回答/追问/拒答）、LangGraph 工作流（意图分流 + 权限校验 +
  `interrupt()` 人工审批 + SqliteSaver 中断恢复）、声明式工具注册（风险等级/授权钩子）、
  SQLite 全链路审计（对话/检索命中/工具链路/审批）
- CLI demo（交互 + scripted）、FastAPI 骨架（/chat、/approvals）
- 测试：政策问答、工具调用、安全注入三类 7 条，Mock LLM + Mock Embedding 离线运行
- 阈值标定：bge-small-zh 命中 ~0.76 / 无关 ~0.32 → high=0.60 low=0.40
