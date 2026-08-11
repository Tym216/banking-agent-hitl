# banking-agent-hitl

[中文](README.md) | **English**

A banking customer-service agent prototype: policy Q&A (RAG), account queries,
ticket creation, and mandatory human approval for sensitive operations.
Built on LangGraph with interruptible/resumable workflows and full audit logging.

## Architecture

```
User → CLI / FastAPI
         │
   LangGraph workflow (SqliteSaver checkpoints, interrupt/resume)
   classify_intent ─┬─ policy_qa (three-state RAG: answer / clarify / refuse)
                    ├─ prepare_tool → check_permission ─┬─ denied / missing params → END
                    │                                   ├─ readonly → execute_tool
                    │                                   └─ sensitive → interrupt() approval → execute / reject
                    └─ chitchat
         │
   SQLite audit (messages, retrieval hits, tool-call chain, approvals)
```

Core security design: **permission decisions and the approval gate are pure
framework code, independent of LLM output** — even if a prompt injection claims
"admin has pre-approved this", sensitive tools cannot bypass the `interrupt()`
approval step.

## Project Layout

```
├── configs/config.yaml          # All config (LLM/embedding/RAG thresholds/storage), env-var overridable
├── data/knowledge_base/         # Knowledge base documents (md/txt/pdf)
├── src/banking_agent/
│   ├── config.py                # pydantic-settings config loading
│   ├── llm/                     # LLM layer: openai_compat (OpenAI/vLLM/Ollama) + mock
│   ├── rag/                     # Embeddings / FAISS index / three-state retriever / loaders (incl. PDF)
│   ├── auth/permissions.py      # Roles + tool authorization + forced-approval decisions
│   ├── tools/                   # Declarative tool registry (risk level / param schema / auth hooks)
│   ├── mcp/                     # MCP tool path: server (FastMCP) / client (sync bridge) / provider (trust-side registration)
│   ├── graph/                   # LangGraph state, nodes, workflow, AgentService
│   ├── storage/                 # SQLite audit tables + AuditLogger
│   ├── api/app.py               # FastAPI skeleton (/chat, /approvals)
│   ├── bootstrap.py             # Assembly entry point
│   └── cli.py                   # CLI demo
└── tests/                       # Policy QA / tool-calling / security-injection / MCP tool-path test suites
```

## Quick Start

```bash
python3 -m venv venv-banking-agent
venv-banking-agent/bin/pip install -e . && venv-banking-agent/bin/pip install sentence-transformers pytest

# Scripted demo (three-state RAG, cross-account denial, transfer approval)
venv-banking-agent/bin/python -m banking_agent.cli --scripted

# Interactive mode
venv-banking-agent/bin/python -m banking_agent.cli

# Local inference via Ollama (requires: ollama pull qwen3.5:9b)
venv-banking-agent/bin/python -m banking_agent.cli --config configs/config.ollama.yaml --scripted

# MCP tool-path demo (banking tools served by an MCP server; stdio auto-spawns the subprocess)
venv-banking-agent/bin/python -m banking_agent.cli --config configs/config.mcp.yaml --scripted

# API server
venv-banking-agent/bin/uvicorn banking_agent.api.app:app --reload

# Tests (offline: mock LLM + mock embeddings)
venv-banking-agent/bin/pytest
```

See [CHANGELOG.md](CHANGELOG.md) for version history.

## Switching LLM Providers (no code changes)

Edit `configs/config.yaml`, pass `--config` (e.g. `configs/config.ollama.yaml`),
or use environment variables:

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

## MCP Tool Integration (no graph changes)

The tool source is config-switchable (`tools.provider: local | mcp`): `local` calls
tools in-process; with `mcp`, banking tools are served by an MCP server
(`src/banking_agent/mcp/server.py`) and consumed by the agent over the MCP protocol,
with zero changes to the LangGraph graph.

```bash
# stdio (default): the agent auto-spawns the server subprocess, nothing to deploy
venv-banking-agent/bin/python -m banking_agent.cli --config configs/config.mcp.yaml --scripted

# streamable HTTP: tools deployed as a standalone service
venv-banking-agent/bin/python -m banking_agent.mcp.server --http     # terminal 1
# set mcp.transport to http in configs/config.mcp.yaml, then run the agent  # terminal 2
```

Trust-boundary design:

- **The server declares capabilities; the client declares trust**: risk levels, minimum
  roles and authorization hooks come from the agent-side local policy — MCP annotations
  (`readOnlyHint` etc.) are merely untrusted hints per the spec, not a security boundary.
  Registered set = server capabilities ∩ local trust declarations; undeclared server
  tools are never registered, and a locally declared tool missing on the server fails
  fast at startup.
- Permission checks and the `interrupt()` approval gate stay in the agent framework
  layer — tools move out of the process, the gate does not move.
- User-identity parameters (e.g. a ticket's `user_id`) are injected by the framework
  and never appear in the LLM-visible parameter schema.
- Adding a tool = one `@server.tool` function on the server + one `ToolSpec` with
  security metadata on the agent side.

## Three-State RAG

| Condition | Behavior |
|---|---|
| top similarity ≥ `high_threshold` | answer with citations |
| between thresholds | low confidence → ask a clarifying question |
| < `low_threshold` or no hits | refuse (hallucination guard) |

Thresholds live under `rag` in `configs/config.yaml` and must be calibrated
per embedding model.

## Approval Flow (interrupt / resume)

```python
reply = service.chat(thread_id, user, "Transfer 500 CNY to ACC-002")
# → {"status": "pending_approval", "approval_request": {...}}
# Graph state is checkpointed to checkpoints.db — survives process restarts

reply = service.resolve_approval(thread_id, approved=True, approver="supervisor")
# → {"status": "completed", "response": "..."}
```
