"""FastAPI 接口骨架：对话 + 审批两组接口。

启动：uvicorn banking_agent.api.app:app --reload
说明：demo 阶段用 header 传 user_id 模拟登录，生产应替换为真实鉴权（JWT/SSO）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from banking_agent.bootstrap import DEMO_USERS, create_service
from banking_agent.graph.workflow import AgentService

_service: AgentService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    _service = create_service()
    yield


app = FastAPI(title="banking-agent-hitl", lifespan=lifespan)


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ApprovalDecision(BaseModel):
    approved: bool
    approver: str
    reason: str = ""


def _get_service() -> AgentService:
    assert _service is not None
    return _service


@app.post("/chat")
def chat(req: ChatRequest, x_user_id: str = Header(default="u_alice")) -> dict[str, Any]:
    user = DEMO_USERS.get(x_user_id)
    if user is None:
        raise HTTPException(401, f"未知用户: {x_user_id}")
    return _get_service().chat(req.thread_id, user, req.message)


@app.post("/approvals/{thread_id}/decision")
def decide(thread_id: str, decision: ApprovalDecision) -> dict[str, Any]:
    result = _get_service().resolve_approval(
        thread_id, decision.approved, decision.approver, decision.reason
    )
    if result.get("status") == "error":
        raise HTTPException(409, result["message"])
    return result
