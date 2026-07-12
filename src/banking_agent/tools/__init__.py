from banking_agent.tools.base import RiskLevel, ToolRegistry, ToolSpec
from banking_agent.tools.readonly import GET_ACCOUNT_BALANCE
from banking_agent.tools.sensitive import CREATE_TICKET, SUBMIT_TRANSACTION


def create_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(GET_ACCOUNT_BALANCE)
    registry.register(CREATE_TICKET)
    registry.register(SUBMIT_TRANSACTION)
    return registry


__all__ = ["RiskLevel", "ToolRegistry", "ToolSpec", "create_default_registry"]
