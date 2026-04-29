"""Kelly criterion position sizing.

The Kelly criterion determines the optimal fraction of bankroll to wager
on a bet with positive expected value. We use fractional Kelly to reduce
variance while still capturing most of the expected growth.

For binary markets:
    Kelly = (p * b - q) / b

Where:
    p = probability of winning
    q = 1 - p (probability of losing)
    b = net odds (payout / stake - 1)

For Kalshi markets specifically:
    - If you buy YES at price P cents, you risk P cents to win (100-P) cents
    - Net odds b = (100 - P) / P
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class KellyResult:
    """Result of Kelly calculation."""

    full_kelly_fraction: float  # Full Kelly as fraction of bankroll
    position_fraction: float  # Fractional Kelly as fraction of bankroll
    position_dollars: float  # Dollar amount to bet
    position_contracts: int  # Number of contracts
    edge: float  # Expected edge (model_prob - market_prob)
    expected_value: float  # Expected value per dollar wagered
    is_positive_ev: bool  # Whether the bet has positive expected value


def calculate_kelly_size(
    model_probability: float,
    market_price_cents: int,
    direction: Literal["yes", "no"],
    bankroll: float,
    kelly_fraction: float = 0.25,
    max_position_dollars: float | None = None,
    max_position_contracts: int | None = None,
) -> KellyResult:
    """
    Calculate Kelly-optimal position size for a Kalshi market.

    Args:
        model_probability: Model's estimated probability of YES outcome (0-1)
        market_price_cents: Current market price in cents (1-99)
        direction: Which side to bet ("yes" or "no")
        bankroll: Total available capital in dollars
        kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
        max_position_dollars: Maximum position size in dollars
        max_position_contracts: Maximum number of contracts

    Returns:
        KellyResult with position sizing details

    Examples:
        >>> # Model thinks YES has 60% chance, market price is 50 cents
        >>> result = calculate_kelly_size(0.60, 50, "yes", 1000.0)
        >>> result.position_contracts
        50  # Quarter Kelly suggests 50 contracts at $0.50 each = $25

        >>> # Same scenario but betting NO (40% chance according to model)
        >>> result = calculate_kelly_size(0.60, 50, "no", 1000.0)
        >>> result.is_positive_ev
        False  # NO bet is negative EV if model says 60% YES
    """
    # Validate inputs
    if not 0 < model_probability < 1:
        return KellyResult(
            full_kelly_fraction=0.0,
            position_fraction=0.0,
            position_dollars=0.0,
            position_contracts=0,
            edge=0.0,
            expected_value=0.0,
            is_positive_ev=False,
        )

    if not 1 <= market_price_cents <= 99:
        return KellyResult(
            full_kelly_fraction=0.0,
            position_fraction=0.0,
            position_dollars=0.0,
            position_contracts=0,
            edge=0.0,
            expected_value=0.0,
            is_positive_ev=False,
        )

    if bankroll <= 0:
        return KellyResult(
            full_kelly_fraction=0.0,
            position_fraction=0.0,
            position_dollars=0.0,
            position_contracts=0,
            edge=0.0,
            expected_value=0.0,
            is_positive_ev=False,
        )

    # Convert market price to probability
    market_probability = market_price_cents / 100

    # Determine effective probabilities based on direction
    if direction == "yes":
        p = model_probability  # Probability we win the bet
        price = market_price_cents  # Cost per contract
    else:
        p = 1 - model_probability  # Probability NO wins = probability YES loses
        price = 100 - market_price_cents  # NO price is inverse of YES price

    q = 1 - p  # Probability we lose

    # Calculate net odds
    # If we pay P cents, we win (100 - P) cents on success
    win_amount = 100 - price  # Profit if we win (in cents)
    b = win_amount / price  # Net odds ratio

    # Calculate edge
    if direction == "yes":
        edge = model_probability - market_probability
    else:
        edge = (1 - model_probability) - (1 - market_probability)
        edge = market_probability - model_probability  # Simplified: negative of YES edge

    # Calculate expected value per dollar wagered
    # EV = p * win - q * stake = p * (100-price)/100 - q * price/100
    ev_per_dollar = (p * win_amount - q * price) / price

    # Check if positive EV
    is_positive_ev = ev_per_dollar > 0

    if not is_positive_ev:
        return KellyResult(
            full_kelly_fraction=0.0,
            position_fraction=0.0,
            position_dollars=0.0,
            position_contracts=0,
            edge=edge,
            expected_value=ev_per_dollar,
            is_positive_ev=False,
        )

    # Calculate full Kelly fraction
    # Kelly = (p * b - q) / b
    full_kelly = (p * b - q) / b

    # Clamp to valid range (should be 0-1 for positive EV bets)
    full_kelly = max(0.0, min(1.0, full_kelly))

    # Apply fractional Kelly
    position_fraction = full_kelly * kelly_fraction

    # Calculate dollar amount
    position_dollars = bankroll * position_fraction

    # Apply maximum limits
    if max_position_dollars is not None:
        position_dollars = min(position_dollars, max_position_dollars)

    # Calculate contracts (each contract costs 'price' cents)
    position_contracts = int(position_dollars / (price / 100))

    # Apply contract limit
    if max_position_contracts is not None:
        position_contracts = min(position_contracts, max_position_contracts)

    # Recalculate actual position dollars based on contracts
    actual_position_dollars = position_contracts * (price / 100)

    return KellyResult(
        full_kelly_fraction=full_kelly,
        position_fraction=actual_position_dollars / bankroll if bankroll > 0 else 0.0,
        position_dollars=actual_position_dollars,
        position_contracts=position_contracts,
        edge=edge,
        expected_value=ev_per_dollar,
        is_positive_ev=True,
    )


def kelly_from_edge(
    edge: float,
    market_price_cents: int,
    direction: Literal["yes", "no"],
    bankroll: float,
    confidence: float = 1.0,
    kelly_fraction: float = 0.25,
    max_position_dollars: float | None = None,
) -> KellyResult:
    """
    Calculate Kelly size from a given edge (shortcut when you have edge directly).

    Args:
        edge: Edge as probability difference (e.g., 0.10 for 10% edge)
        market_price_cents: Current market price in cents
        direction: Which side ("yes" or "no")
        bankroll: Total available capital
        confidence: Confidence in the edge estimate (scales down position)
        kelly_fraction: Fraction of full Kelly to use
        max_position_dollars: Maximum position size

    Returns:
        KellyResult with position sizing details
    """
    market_probability = market_price_cents / 100

    if direction == "yes":
        model_probability = market_probability + edge
    else:
        model_probability = market_probability - edge

    # Clamp to valid range
    model_probability = max(0.01, min(0.99, model_probability))

    # Scale edge by confidence
    effective_kelly_fraction = kelly_fraction * confidence

    return calculate_kelly_size(
        model_probability=model_probability,
        market_price_cents=market_price_cents,
        direction=direction,
        bankroll=bankroll,
        kelly_fraction=effective_kelly_fraction,
        max_position_dollars=max_position_dollars,
    )
