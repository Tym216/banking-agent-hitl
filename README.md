# banking-agent-hitl

**中文** | [English](README.en.md)

银行客服 Agent 原型：政策问答（RAG）、业务查询、工单创建、敏感操作人工审批。
基于 LangGraph 工作流编排，支持中断/恢复，全链路审计落库。

## 架构

```
用户 → CLI / FastAPI
         │
   LangGraph 工作流（SqliteSaver 持久化，支持中断恢复）
   classify_intent ─┬─ policy_qa（RAG 三态：回答/追问/拒答）
                    ├─ prepare_tool → check_permission ─┬─ 拒绝/缺参 → 结束
                    │                                   ├─ 只读 → execute_tool
                    │                                   └─ 敏感 → interrupt() 人工审批 → 执行/驳回
                    └─ chitchat
         │
   SQLite 审计（对话、检索命中、工具链路、审批记录）
```

安全设计核心：**权限裁决与审批闸门是纯框架代码，不依赖 LLM 输出**——
即使 prompt 被注入"免审批"指令，敏感工具也无法绕过 `interrupt()` 审批环节。

## 目录结构

```
├── configs/config.yaml          # 全部配置（LLM/Embedding/RAG阈值/存储），环境变量可覆盖
├── data/knowledge_base/         # 知识库文档（md/txt/pdf）
├── src/banking_agent/
│   ├── config.py                # pydantic-settings 配置加载
│   ├── llm/                     # LLM 层：openai_compat（OpenAI/vLLM/Ollama 通用）+ mock
│   ├── rag/                     # Embedding / FAISS 索引 / 三态检索器 / 文档加载(含PDF)
│   ├── auth/permissions.py      # 角色 + 工具授权 + 强制审批裁决
│   ├── tools/                   # 声明式工具注册（风险等级/参数schema/授权钩子）
│   ├── graph/                   # LangGraph 状态、节点、图编排、AgentService
│   ├── storage/                 # SQLite 审计表 + AuditLogger
│   ├── api/app.py               # FastAPI 骨架（/chat、/approvals）
│   ├── bootstrap.py             # 装配入口
│   └── cli.py                   # CLI demo
└── tests/                       # 政策问答 / 工具调用 / 安全注入 三类测试
```

## 快速开始

```bash
python3 -m venv venv-banking-agent
venv-banking-agent/bin/pip install -e . && venv-banking-agent/bin/pip install sentence-transformers pytest

# 预置演示对话（含 RAG 三态、越权拦截、转账审批）
venv-banking-agent/bin/python -m banking_agent.cli --scripted

# 交互模式
venv-banking-agent/bin/python -m banking_agent.cli

# 使用 Ollama 本地推理（需 ollama pull qwen3.5:9b）
venv-banking-agent/bin/python -m banking_agent.cli --config configs/config.ollama.yaml --scripted

# API 服务
venv-banking-agent/bin/uvicorn banking_agent.api.app:app --reload

# 测试（离线，Mock LLM + Mock Embedding）
venv-banking-agent/bin/pytest
```

版本记录见 [CHANGELOG.md](CHANGELOG.md)。

## 切换 LLM 来源（不改代码）

编辑 `configs/config.yaml`、指定 `--config` 配置文件（如 `configs/config.ollama.yaml`），或用环境变量：

```bash
# vLLM
export BANKING_AGENT_LLM__PROVIDER=openai_compat
export BANKING_AGENT_LLM__BASE_URL=http://localhost:8000/v1
export BANKING_AGENT_LLM__MODEL=Qwen/Qwen2.5-7B-Instruct

# Ollama
export BANKING_AGENT_LLM__BASE_URL=http://localhost:11434/v1
export BANKING_AGENT_LLM__MODEL=qwen2.5:7b

# OpenAI
export BANKING_AGENT_LLM__BASE_URL=https://api.openai.com/v1
export OPENAI_API_KEY=sk-...
```

## RAG 三态判定

| 条件 | 行为 |
|---|---|
| top 相似度 ≥ `high_threshold` | 基于引用回答 |
| 介于两阈值之间 | 低置信，生成追问 |
| < `low_threshold` 或无命中 | 拒答（防幻觉） |

阈值在 `configs/config.yaml` 的 `rag` 段配置，需按所用 embedding 模型标定。

## 审批流（中断/恢复）

```python
reply = service.chat(thread_id, user, "向账户 ACC-002 转账 500 元")
# → {"status": "pending_approval", "approval_request": {...}}
# 图状态已持久化到 checkpoints.db，进程重启也能恢复

reply = service.resolve_approval(thread_id, approved=True, approver="supervisor")
# → {"status": "completed", "response": "操作已完成..."}
```
