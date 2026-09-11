"""FastAPI 接口骨架：登录 + 对话 + 审批三组接口。

启动：uvicorn banking_agent.api.app:app --reload
鉴权：/login 用用户名密码换取用户身份；/chat 与 /approvals 用 x-user-id
      header 标识当前用户（demo 简化，生产替换为 JWT/SSO，见 AuthProvider 抽象）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from banking_agent.auth.accounts import UserStore, init_users
from banking_agent.auth.permissions import User
from banking_agent.bootstrap import create_service
from banking_agent.config import load_config
from banking_agent.graph.workflow import AgentService
from banking_agent.storage import connect

_service: AgentService | None = None
_users: UserStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service, _users
    cfg = load_config()
    _service = create_service(config=cfg)
    # users 表与审计表同库(agent.db),用独立连接
    _users = init_users(connect(cfg.resolve_path(cfg.storage.db_path)))
    yield


app = FastAPI(title="banking-agent-hitl", lifespan=lifespan)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""


def _get_service() -> AgentService:
    assert _service is not None
    return _service


def _get_users() -> UserStore:
    assert _users is not None
    return _users


def _current_user(x_user_id: str | None) -> User:
    if not x_user_id:
        raise HTTPException(401, "缺少 x-user-id header")
    user = _get_users().get_by_id(x_user_id)
    if user is None:
        raise HTTPException(401, f"未知用户: {x_user_id}")
    return user


@app.post("/login")
def login(req: LoginRequest) -> dict[str, Any]:
    user = _get_users().authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(401, "用户名或密码错误")
    return {"status": "ok", "user": user.to_dict()}


@app.post("/chat")
def chat(req: ChatRequest, x_user_id: str = Header(default="")) -> dict[str, Any]:
    user = _current_user(x_user_id)
    return _get_service().chat(req.thread_id, user, req.message)


@app.post("/approvals/{thread_id}/decision")
def decide(thread_id: str, decision: ApprovalDecision,
           x_user_id: str = Header(default="")) -> dict[str, Any]:
    approver = _current_user(x_user_id)
    result = _get_service().resolve_approval(
        thread_id, decision.approved, approver, decision.reason
    )
    if result.get("status") == "error":
        raise HTTPException(403, result["message"])
    return result


@app.get("/approvals/pending")
def list_pending(x_user_id: str = Header(default="")) -> dict[str, Any]:
    """待审批列表：仅 staff 且 can_approve 可见，客户/无权限 staff 403。"""
    user = _current_user(x_user_id)
    result = _get_service().list_pending_approvals(user)
    if result.get("status") == "error":
        raise HTTPException(403, result["message"])
    return result
