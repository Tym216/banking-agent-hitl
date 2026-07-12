# Changelog

本项目版本记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)。

## [Unreleased]

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
