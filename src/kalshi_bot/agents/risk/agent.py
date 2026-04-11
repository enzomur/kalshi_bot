"""Weather Risk Agent - validates trades with weather-specific risk checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from kalshi_bot.agents.base import BaseAgent
from kalshi_bot.agents.risk.concentration_monitor import (
    ConcentrationLimits,
    ConcentrationMonitor,
)
from kalshi_bot.agents.risk.correlation_tracker import CorrelationTracker
from kalshi_bot.agents.weather.market_mapper import WeatherMarketMapper
from kalshi_bot.core.types import ArbitrageOpportunity
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database
    from kalshi_bot.agents.weather.agent import WeatherResearchAgent

logger = get_logger(__name__)


class RiskDecision(str, Enum):
    """Risk check decision."""

    APPROVE = "approve"
    REDUCE = "reduce"   # Approve with reduced size
    REJECT = "reject"


@dataclass
class RiskCheckResult:
    """Result of a weather risk check."""

    decision: RiskDecision
    reason: str
    original_quantity: int
    approved_quantity: int
    checks_passed: list[str]
    checks_failed: list[str]

    @property
    def approved(self) -> bool:
        return self.decision in (RiskDecision.APPROVE, RiskDecision.REDUCE)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "original_quantity": self.original_quantity,
            "approved_quantity": self.approved_quantity,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
        }


class WeatherRiskAgent(BaseAgent):
    """
    Agent that performs weather-specific risk checks.

    Extends the base RiskManager with:
    - Weather market concentration limits
    - Location-based exposure limits
    - Correlation-based risk adjustment
    - Forecast confidence requirements
    """

    def __init__(
        self,
        db: "Database",
        weather_agent: "WeatherResearchAgent | None" = None,
        max_weather_exposure_pct: float = 0.30,
        max_single_location_pct: float = 0.15,
        min_forecast_confidence: float = 0.60,
        enabled: bool = True,
    ) -> None:
        """
        Initialize the Weather Risk Agent.

        Args:
            db: Database connection
            weather_agent: Weather research agent for forecast data
            max_weather_exposure_pct: Max portfolio % in weather markets
            max_single_location_pct: Max portfolio % in single location
            min_forecast_confidence: Minimum NWS confidence required
            enabled: Whether the agent is enabled
        """
        super().__init__(
            db=db,
            name="weather_risk",
            update_interval_seconds=300,  # 5 minutes
            enabled=enabled,
        )

        self._weather_agent = weather_agent
        self._market_mapper = WeatherMarketMapper()

        limits = ConcentrationLimits(
            max_weather_exposure_pct=max_weather_exposure_pct,
            max_single_location_pct=max_single_location_pct,
        )
        self._concentration_monitor = ConcentrationMonitor(db, limits)
        self._correlation_tracker = CorrelationTracker(db)

        self._min_forecast_confidence = min_forecast_confidence

        # Current positions cache
        self._current_positions: list[dict] = []
        self._portfolio_value: float = 0.0

    def set_weather_agent(self, agent: "WeatherResearchAgent") -> None:
        """Set the weather agent reference."""
        self._weather_agent = agent

    async def _run_cycle(self) -> None:
        """Execute periodic risk monitoring."""
        # Load correlations from database
        await self._correlation_tracker.load_from_database()

        # Update concentration snapshot if we have positions
        if self._current_positions and self._portfolio_value > 0:
            snapshot = self._concentration_monitor.compute_concentration(
                self._current_positions,
                self._portfolio_value,
            )
            violations = self._concentration_monitor.check_limits(snapshot)

            if violations:
                logger.warning(f"Weather concentration violations: {violations}")

            await self._concentration_monitor.save_snapshot(snapshot)

            self._status.metrics = {
                "weather_exposure_pct": snapshot.weather_pct,
                "max_location_pct": snapshot.max_location_pct,
                "max_location": snapshot.max_location,
                "violations": violations,
            }

    def update_positions(
        self,
        positions: list[dict],
        portfolio_value: float,
    ) -> None:
        """
        Update current positions for risk calculations.

        Args:
            positions: List of position dicts
            portfolio_value: Total portfolio value
        """
        self._current_positions = positions
        self._portfolio_value = portfolio_value

        # Recompute concentration
        self._concentration_monitor.compute_concentration(
            positions,
            portfolio_value,
        )

    async def pre_trade_check(
        self,
        opportunity: ArbitrageOpportunity,
        positions: list[dict],
        portfolio_value: float,
    ) -> RiskCheckResult:
        """
        Perform pre-trade risk checks for weather markets.

        Args:
            opportunity: Trading opportunity
            positions: Current positions
            portfolio_value: Total portfolio value

        Returns:
            RiskCheckResult with decision and details
        """
        ticker = opportunity.markets[0] if opportunity.markets else ""
        quantity = opportunity.max_quantity

        checks_passed = []
        checks_failed = []

        # Check if this is a weather market
        mapping = self._market_mapper.parse_ticker(ticker)
        if not mapping:
            # Not a weather market, approve without weather-specific checks
            return RiskCheckResult(
                decision=RiskDecision.APPROVE,
                reason="Not a weather market",
                original_quantity=quantity,
                approved_quantity=quantity,
                checks_passed=["non_weather_market"],
                checks_failed=[],
            )

        # Update concentration snapshot
        self._current_positions = positions
        self._portfolio_value = portfolio_value
        snapshot = self._concentration_monitor.compute_concentration(
            positions,
            portfolio_value,
        )

        # Check 1: Forecast confidence
        if self._weather_agent:
            estimate = self._weather_agent.get_probability(ticker)
            if estimate:
                if estimate.confidence >= self._min_forecast_confidence:
                    checks_passed.append(
                        f"forecast_confidence ({estimate.confidence:.1%})"
                    )
                else:
                    checks_failed.append(
                        f"forecast_confidence ({estimate.confidence:.1%} < {self._min_forecast_confidence:.1%})"
                    )
            else:
                # No forecast available - might want to be cautious
                checks_failed.append("no_forecast_available")
        else:
            checks_passed.append("forecast_check_skipped")

        # Check 2: Weather exposure limit
        if snapshot.weather_pct < self._concentration_monitor.get_limits().max_weather_exposure_pct:
            checks_passed.append(f"weather_exposure ({snapshot.weather_pct:.1%})")
        else:
            checks_failed.append(
                f"weather_exposure ({snapshot.weather_pct:.1%} >= "
                f"{self._concentration_monitor.get_limits().max_weather_exposure_pct:.1%})"
            )

        # Check 3: Single location limit
        location = mapping.location_code
        location_exposure = snapshot.by_location.get(location, 0)
        location_pct = location_exposure / portfolio_value if portfolio_value > 0 else 0

        if location_pct < self._concentration_monitor.get_limits().max_single_location_pct:
            checks_passed.append(f"location_exposure ({location}: {location_pct:.1%})")
        else:
            checks_failed.append(
                f"location_exposure ({location}: {location_pct:.1%} >= "
                f"{self._concentration_monitor.get_limits().max_single_location_pct:.1%})"
            )

        # Check 4: Correlated exposure
        correlated = self._correlation_tracker.get_correlated_locations(
            location, min_correlation=0.50
        )
        correlated_exposure = location_exposure
        for corr_loc, corr_coef in correlated:
            correlated_exposure += snapshot.by_location.get(corr_loc, 0) * corr_coef

        correlated_pct = correlated_exposure / portfolio_value if portfolio_value > 0 else 0
        if correlated_pct < self._concentration_monitor.get_limits().max_correlated_exposure_pct:
            checks_passed.append(f"correlated_exposure ({correlated_pct:.1%})")
        else:
            checks_failed.append(
                f"correlated_exposure ({correlated_pct:.1%} >= "
                f"{self._concentration_monitor.get_limits().max_correlated_exposure_pct:.1%})"
            )

        # Determine final decision
        if not checks_failed:
            return RiskCheckResult(
                decision=RiskDecision.APPROVE,
                reason="All weather risk checks passed",
                original_quantity=quantity,
                approved_quantity=quantity,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        # Calculate reduced quantity based on available capacity
        available_capacity = self._concentration_monitor.get_available_capacity(
            location,
            mapping.weather_type.value,
            portfolio_value,
        )

        # Convert capacity to quantity (assuming $1 per contract)
        max_quantity = int(available_capacity)

        if max_quantity > 0:
            approved_quantity = min(quantity, max_quantity)
            return RiskCheckResult(
                decision=RiskDecision.REDUCE,
                reason=f"Reduced to {approved_quantity} due to limits",
                original_quantity=quantity,
                approved_quantity=approved_quantity,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
            )

        # Reject if no capacity available
        return RiskCheckResult(
            decision=RiskDecision.REJECT,
            reason="; ".join(checks_failed),
            original_quantity=quantity,
            approved_quantity=0,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )

    def get_status(self) -> dict[str, Any]:
        """Get agent status with risk-specific metrics."""
        status = super().get_status()

        snapshot = self._concentration_monitor.get_current_snapshot()
        if snapshot:
            status["concentration"] = snapshot.to_dict()

        status["limits"] = {
            "max_weather_exposure_pct": self._concentration_monitor.get_limits().max_weather_exposure_pct,
            "max_single_location_pct": self._concentration_monitor.get_limits().max_single_location_pct,
            "min_forecast_confidence": self._min_forecast_confidence,
        }

        return status
