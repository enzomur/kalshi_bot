"""Execution risk model for non-atomic multi-leg arbitrage trades.

Based on: "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets"
(arXiv:2508.03474v1)

Models the risk of sequential leg execution where:
- Legs are executed one at a time (non-atomic)
- Prices may move between legs
- Some legs may fail to fill
- Market conditions may change

The paper recommends a $0.05 minimum profit threshold to account for these risks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kalshi_bot.core.types import OrderBook, Side
from kalshi_bot.execution.vwap import VWAPCalculator, MultiLegVWAPResult
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PriceMovementModel:
    """Model of expected price movement between leg executions."""

    market_ticker: str
    expected_move_cents: float  # Expected absolute price move
    volatility_cents: float  # Standard deviation of price moves
    time_horizon_seconds: float  # Time period for the model
    confidence_interval: tuple[float, float]  # 95% CI for price


@dataclass
class LegFillProbability:
    """Probability model for leg execution."""

    market_ticker: str
    side: Side
    price: int
    quantity: int
    fill_probability: float  # P(fill at price within time)
    expected_time_seconds: float  # Expected time to fill
    partial_fill_probability: float  # P(partial fill only)


@dataclass
class ExecutionRiskEstimate:
    """Comprehensive execution risk assessment."""

    opportunity_id: str
    leg_risks: list[LegFillProbability]
    price_movement_risk: list[PriceMovementModel]
    overall_success_probability: float  # P(all legs fill successfully)
    expected_slippage_cents: float  # Expected total slippage
    slippage_std_cents: float  # Standard deviation of slippage
    var_95_cents: float  # 95% Value at Risk (max expected loss)
    execution_time_estimate: float  # Expected total execution time in seconds
    recommended_profit_threshold: float  # Minimum profit to justify risk


@dataclass
class ExecutionDecision:
    """Decision on whether to execute an opportunity."""

    should_execute: bool
    reason: str
    confidence: float  # 0-1 confidence in decision
    adjusted_profit: float  # Profit after risk adjustment
    risk_adjusted_quantity: int  # Recommended quantity
    warnings: list[str] = field(default_factory=list)


class ExecutionRiskModeler:
    """
    Model execution risk for multi-leg arbitrage trades.

    Accounts for:
    - Sequential leg execution (non-atomic)
    - Price movement between legs
    - Fill probability at different prices
    - Liquidity constraints
    - Time decay of opportunity
    """

    def __init__(
        self,
        vwap_calculator: VWAPCalculator | None = None,
        default_fill_time_seconds: float = 2.0,
        default_volatility_cents: float = 1.0,
        confidence_level: float = 0.95,
    ) -> None:
        """
        Initialize risk modeler.

        Args:
            vwap_calculator: VWAP calculator for slippage analysis
            default_fill_time_seconds: Default expected fill time per leg
            default_volatility_cents: Default price volatility per second
            confidence_level: Confidence level for VaR calculation
        """
        self.vwap = vwap_calculator or VWAPCalculator()
        self.default_fill_time = default_fill_time_seconds
        self.default_volatility = default_volatility_cents
        self.confidence_level = confidence_level

    def estimate_leg_fill_probability(
        self,
        orderbook: OrderBook,
        side: Side,
        price: int,
        quantity: int,
        time_seconds: float = 10.0,
    ) -> LegFillProbability:
        """
        Estimate probability of filling a leg at given price.

        Args:
            orderbook: Current order book
            side: YES or NO
            price: Limit price in cents
            quantity: Number of contracts
            time_seconds: Time window for fill

        Returns:
            LegFillProbability with fill estimates
        """
        # Get available liquidity at or better than price
        if side == Side.YES:
            levels = orderbook.yes_asks
            is_buy = True
        else:
            levels = orderbook.no_asks
            is_buy = True

        available_at_price = 0
        for level in levels:
            if is_buy and level.price <= price:
                available_at_price += level.quantity
            elif not is_buy and level.price >= price:
                available_at_price += level.quantity

        # Calculate fill probability based on available liquidity
        if available_at_price >= quantity:
            fill_prob = 0.95  # High probability if liquidity exists
        elif available_at_price > 0:
            fill_prob = 0.5 + 0.45 * (available_at_price / quantity)
        else:
            # No immediate liquidity, estimate based on time
            fill_prob = min(0.3 + 0.05 * time_seconds, 0.7)

        # Partial fill probability
        partial_fill_prob = 0.0
        if available_at_price > 0 and available_at_price < quantity:
            partial_fill_prob = 0.3

        # Estimate fill time
        if available_at_price >= quantity:
            expected_time = self.default_fill_time
        else:
            expected_time = self.default_fill_time * (1 + quantity / max(available_at_price, 1))

        return LegFillProbability(
            market_ticker=orderbook.market_ticker,
            side=side,
            price=price,
            quantity=quantity,
            fill_probability=fill_prob,
            expected_time_seconds=min(expected_time, time_seconds),
            partial_fill_probability=partial_fill_prob,
        )

    def estimate_price_movement(
        self,
        ticker: str,
        time_between_legs: float,
        current_price: int | None = None,
        historical_volatility: float | None = None,
    ) -> PriceMovementModel:
        """
        Estimate price movement between leg executions.

        Args:
            ticker: Market ticker
            time_between_legs: Time in seconds between legs
            current_price: Current price in cents (for proportional volatility)
            historical_volatility: Historical volatility if available

        Returns:
            PriceMovementModel with movement estimates
        """
        # Use historical volatility if available, otherwise default
        vol_per_second = historical_volatility or self.default_volatility

        # Scale volatility by square root of time
        vol_scaled = vol_per_second * np.sqrt(time_between_legs)

        # Expected move is approximately 0.8 * volatility (for normal distribution)
        expected_move = 0.8 * vol_scaled

        # 95% confidence interval
        z_95 = 1.96
        ci_low = -z_95 * vol_scaled
        ci_high = z_95 * vol_scaled

        return PriceMovementModel(
            market_ticker=ticker,
            expected_move_cents=expected_move,
            volatility_cents=vol_scaled,
            time_horizon_seconds=time_between_legs,
            confidence_interval=(ci_low, ci_high),
        )

    def calculate_execution_risk(
        self,
        legs: list[dict[str, Any]],
        orderbooks: dict[str, OrderBook],
        opportunity_id: str = "",
    ) -> ExecutionRiskEstimate:
        """
        Calculate comprehensive execution risk for an opportunity.

        Args:
            legs: Leg definitions with market, side, price, quantity
            orderbooks: Order books for each market
            opportunity_id: Identifier for the opportunity

        Returns:
            ExecutionRiskEstimate with full risk assessment
        """
        leg_risks: list[LegFillProbability] = []
        price_risks: list[PriceMovementModel] = []
        total_time = 0.0

        for i, leg in enumerate(legs):
            ticker = leg["market"]
            orderbook = orderbooks.get(ticker)

            if orderbook is None:
                # Conservative estimate for missing orderbook
                leg_risk = LegFillProbability(
                    market_ticker=ticker,
                    side=Side(leg["side"]),
                    price=leg["price"],
                    quantity=leg["quantity"],
                    fill_probability=0.5,
                    expected_time_seconds=5.0,
                    partial_fill_probability=0.3,
                )
            else:
                leg_risk = self.estimate_leg_fill_probability(
                    orderbook,
                    Side(leg["side"]),
                    leg["price"],
                    leg["quantity"],
                )

            leg_risks.append(leg_risk)
            total_time += leg_risk.expected_time_seconds

            # Estimate price movement between this leg and next
            if i < len(legs) - 1:
                time_to_next = leg_risk.expected_time_seconds
                price_risk = self.estimate_price_movement(
                    ticker,
                    time_to_next,
                    current_price=leg["price"],
                )
                price_risks.append(price_risk)

        # Overall success probability (product of individual fill probabilities)
        overall_prob = np.prod([r.fill_probability for r in leg_risks])

        # Expected slippage from VWAP
        vwap_result = self.vwap.calculate_multi_leg_vwap(legs, orderbooks)
        expected_slippage = vwap_result.total_slippage_cents

        # Slippage standard deviation from price movement
        slippage_var = sum(r.volatility_cents ** 2 for r in price_risks)
        slippage_std = np.sqrt(slippage_var)

        # 95% VaR (Value at Risk)
        z_value = 1.96  # 95% confidence
        var_95 = expected_slippage + z_value * slippage_std

        # Recommended profit threshold based on risk
        # Paper uses $0.05, we adjust based on calculated risk
        base_threshold = 0.05
        risk_adjustment = (var_95 / 100) * 0.5  # Convert cents to dollars
        recommended_threshold = max(base_threshold, base_threshold + risk_adjustment)

        return ExecutionRiskEstimate(
            opportunity_id=opportunity_id,
            leg_risks=leg_risks,
            price_movement_risk=price_risks,
            overall_success_probability=overall_prob,
            expected_slippage_cents=expected_slippage,
            slippage_std_cents=slippage_std,
            var_95_cents=var_95,
            execution_time_estimate=total_time,
            recommended_profit_threshold=recommended_threshold,
        )

    def calculate_minimum_profit_threshold(
        self,
        risk: ExecutionRiskEstimate,
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate minimum profit threshold based on risk.

        Args:
            risk: Execution risk estimate
            confidence: Confidence level (default 95%)

        Returns:
            Minimum profit in dollars to justify execution
        """
        # Base threshold from paper
        base = 0.05

        # Adjust for slippage risk
        if confidence == 0.95:
            z_value = 1.96
        elif confidence == 0.99:
            z_value = 2.58
        else:
            from scipy.stats import norm
            z_value = norm.ppf(confidence)

        slippage_adjustment = (
            risk.expected_slippage_cents + z_value * risk.slippage_std_cents
        ) / 100

        # Adjust for fill probability
        fill_adjustment = (1 - risk.overall_success_probability) * 0.10

        return base + slippage_adjustment + fill_adjustment

    def should_execute(
        self,
        legs: list[dict[str, Any]],
        orderbooks: dict[str, OrderBook],
        expected_profit: float,
        vwap_result: MultiLegVWAPResult | None = None,
        opportunity_id: str = "",
    ) -> ExecutionDecision:
        """
        Determine whether to execute an opportunity based on risk.

        Args:
            legs: Leg definitions
            orderbooks: Order books
            expected_profit: Expected profit in dollars (before slippage)
            vwap_result: Pre-computed VWAP result (optional)
            opportunity_id: Opportunity identifier

        Returns:
            ExecutionDecision with recommendation
        """
        warnings: list[str] = []

        # Calculate risk
        risk = self.calculate_execution_risk(legs, orderbooks, opportunity_id)

        # Get or compute VWAP
        if vwap_result is None:
            vwap_result = self.vwap.calculate_multi_leg_vwap(
                legs, orderbooks, expected_profit
            )

        # Calculate adjusted profit
        slippage_cost = risk.expected_slippage_cents * legs[0]["quantity"] / 100
        adjusted_profit = expected_profit - slippage_cost

        # Get minimum threshold
        min_threshold = self.calculate_minimum_profit_threshold(risk)

        # Check conditions
        reasons: list[str] = []

        # 1. Check if profitable after slippage
        if adjusted_profit < min_threshold:
            reasons.append(
                f"Adjusted profit ${adjusted_profit:.3f} below threshold ${min_threshold:.3f}"
            )

        # 2. Check fill probability
        if risk.overall_success_probability < 0.7:
            reasons.append(
                f"Low fill probability: {risk.overall_success_probability:.1%}"
            )
            warnings.append("Fill probability below 70%")

        # 3. Check if VWAP shows all legs can be filled
        if not vwap_result.all_filled:
            reasons.append("Insufficient liquidity for all legs")
            warnings.append("Some legs cannot be fully filled")

        # 4. Check VaR
        var_dollars = risk.var_95_cents * legs[0]["quantity"] / 100
        if var_dollars > adjusted_profit * 0.5:
            warnings.append(f"High VaR: ${var_dollars:.2f} vs profit ${adjusted_profit:.2f}")

        # Decision
        should_execute = len(reasons) == 0

        if should_execute:
            reason = "All risk checks passed"
            confidence = min(risk.overall_success_probability, 0.95)
        else:
            reason = "; ".join(reasons)
            confidence = 0.0

        # Calculate risk-adjusted quantity
        # Reduce quantity if risk is high
        risk_factor = risk.overall_success_probability
        base_quantity = legs[0]["quantity"] if legs else 0
        risk_adjusted_qty = int(base_quantity * risk_factor)

        return ExecutionDecision(
            should_execute=should_execute,
            reason=reason,
            confidence=confidence,
            adjusted_profit=adjusted_profit,
            risk_adjusted_quantity=max(risk_adjusted_qty, 1),
            warnings=warnings,
        )

    def estimate_optimal_execution_speed(
        self,
        risk: ExecutionRiskEstimate,
        max_slippage_cents: float = 5.0,
    ) -> dict[str, Any]:
        """
        Estimate optimal execution speed tradeoff.

        Faster execution = less price movement risk but more slippage
        Slower execution = more price movement risk but potentially better fills

        Args:
            risk: Execution risk estimate
            max_slippage_cents: Maximum acceptable slippage

        Returns:
            Recommendations for execution speed
        """
        # Calculate slippage vs time tradeoff
        current_slippage = risk.expected_slippage_cents
        price_movement_risk = sum(r.volatility_cents for r in risk.price_movement_risk)

        if current_slippage < max_slippage_cents and price_movement_risk < max_slippage_cents:
            recommendation = "aggressive"
            description = "Execute quickly - low slippage and price movement risk"
        elif current_slippage > max_slippage_cents:
            recommendation = "patient"
            description = "Use limit orders and wait - high immediate slippage"
        else:
            recommendation = "balanced"
            description = "Balance speed with price - moderate risk"

        return {
            "recommendation": recommendation,
            "description": description,
            "current_slippage_cents": current_slippage,
            "price_movement_risk_cents": price_movement_risk,
            "estimated_execution_time_seconds": risk.execution_time_estimate,
        }
