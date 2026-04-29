"""Risk management - THE GATEKEEPER to execution.

IMPORTANT: Only the risk/ module may import from execution/.
All trading decisions must flow through the Risk Engine.
"""

from src.risk.engine import RiskEngine, RiskDecision, CircuitBreakerState, PositionCheck
from src.risk.kelly import calculate_kelly_size, kelly_from_edge, KellyResult

__all__ = [
    "RiskEngine",
    "RiskDecision",
    "CircuitBreakerState",
    "PositionCheck",
    "calculate_kelly_size",
    "kelly_from_edge",
    "KellyResult",
]
