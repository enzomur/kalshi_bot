"""Momentum calculation for settlement trading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class MomentumSignal:
    """A momentum signal for a market approaching settlement."""

    ticker: str
    event_ticker: str
    current_price: int
    price_6h_ago: int | None
    price_24h_ago: int | None
    hours_to_settlement: float
    momentum: float  # Rate of price change toward 0 or 100
    direction: str  # 'yes' (price rising) or 'no' (price falling)
    confidence: float
    volume_24h: int

    @property
    def is_converging_yes(self) -> bool:
        """Check if price is converging toward 100 (YES settlement)."""
        return self.direction == "yes" and self.current_price >= 70

    @property
    def is_converging_no(self) -> bool:
        """Check if price is converging toward 0 (NO settlement)."""
        return self.direction == "no" and self.current_price <= 30


class MomentumCalculator:
    """
    Calculates momentum for markets approaching settlement.

    Momentum = rate of price change * consistency of direction

    High momentum + high price (>70) = likely YES settlement
    High momentum + low price (<30) = likely NO settlement
    """

    # Minimum hours to settlement to consider
    MIN_HOURS = 1.0

    # Maximum hours to settlement (beyond this, momentum is less predictive)
    MAX_HOURS = 48.0

    # Minimum price change to consider momentum meaningful
    MIN_PRICE_CHANGE = 5  # 5 cents over period

    # Minimum volume to trust the signal
    MIN_VOLUME = 10

    def __init__(
        self,
        min_hours: float | None = None,
        max_hours: float | None = None,
    ) -> None:
        self._min_hours = min_hours or self.MIN_HOURS
        self._max_hours = max_hours or self.MAX_HOURS

    def calculate_momentum(
        self,
        ticker: str,
        event_ticker: str,
        current_price: int,
        price_6h_ago: int | None,
        price_24h_ago: int | None,
        hours_to_settlement: float,
        volume_24h: int = 0,
    ) -> MomentumSignal | None:
        """
        Calculate momentum signal for a market.

        Args:
            ticker: Market ticker
            event_ticker: Event ticker
            current_price: Current price in cents
            price_6h_ago: Price 6 hours ago (optional)
            price_24h_ago: Price 24 hours ago (optional)
            hours_to_settlement: Hours until market settles
            volume_24h: 24h trading volume

        Returns:
            MomentumSignal if momentum is significant, None otherwise
        """
        # Check time bounds
        if hours_to_settlement < self._min_hours:
            return None
        if hours_to_settlement > self._max_hours:
            return None

        # Need at least one historical price
        if price_6h_ago is None and price_24h_ago is None:
            return None

        # Calculate price changes
        change_6h = None
        change_24h = None

        if price_6h_ago is not None:
            change_6h = current_price - price_6h_ago

        if price_24h_ago is not None:
            change_24h = current_price - price_24h_ago

        # Determine direction from available data
        if change_6h is not None and change_24h is not None:
            # Both periods agree on direction = stronger signal
            if change_6h > 0 and change_24h > 0:
                direction = "yes"
                consistency = 1.0
            elif change_6h < 0 and change_24h < 0:
                direction = "no"
                consistency = 1.0
            else:
                # Mixed signals - use 6h (more recent)
                direction = "yes" if change_6h > 0 else "no"
                consistency = 0.5
        elif change_6h is not None:
            direction = "yes" if change_6h > 0 else "no"
            consistency = 0.7
        else:
            direction = "yes" if change_24h > 0 else "no"
            consistency = 0.6

        # Calculate momentum magnitude
        # Use the most meaningful change
        primary_change = change_6h if change_6h is not None else change_24h
        abs_change = abs(primary_change)

        if abs_change < self.MIN_PRICE_CHANGE:
            return None

        # Momentum = normalized price change * time factor
        # Closer to settlement = higher momentum value for same price change
        time_factor = 1.0 + (self._max_hours - hours_to_settlement) / self._max_hours
        momentum = (abs_change / 100.0) * time_factor * consistency

        # Confidence based on multiple factors
        confidence = self._calculate_confidence(
            current_price=current_price,
            direction=direction,
            abs_change=abs_change,
            hours_to_settlement=hours_to_settlement,
            consistency=consistency,
            volume_24h=volume_24h,
        )

        return MomentumSignal(
            ticker=ticker,
            event_ticker=event_ticker,
            current_price=current_price,
            price_6h_ago=price_6h_ago,
            price_24h_ago=price_24h_ago,
            hours_to_settlement=hours_to_settlement,
            momentum=momentum,
            direction=direction,
            confidence=confidence,
            volume_24h=volume_24h,
        )

    def _calculate_confidence(
        self,
        current_price: int,
        direction: str,
        abs_change: int,
        hours_to_settlement: float,
        consistency: float,
        volume_24h: int,
    ) -> float:
        """
        Calculate confidence in the momentum signal.

        Factors:
        - Price proximity to 0/100 (extreme prices = higher confidence)
        - Magnitude of price change (bigger moves = higher confidence)
        - Time to settlement (closer = higher confidence)
        - Direction consistency (both periods agree = higher confidence)
        - Volume (more trading = higher confidence)
        """
        confidence = 0.0

        # Price proximity factor (0-0.3)
        # Higher if price is already near the target
        if direction == "yes" and current_price >= 70:
            proximity = (current_price - 70) / 30.0  # 0 at 70, 1 at 100
            confidence += 0.3 * proximity
        elif direction == "no" and current_price <= 30:
            proximity = (30 - current_price) / 30.0  # 0 at 30, 1 at 0
            confidence += 0.3 * proximity

        # Change magnitude factor (0-0.25)
        change_factor = min(abs_change / 20.0, 1.0)  # Cap at 20 cent change
        confidence += 0.25 * change_factor

        # Time factor (0-0.25)
        # Closer to settlement = more confident
        time_factor = max(0, 1.0 - hours_to_settlement / self._max_hours)
        confidence += 0.25 * time_factor

        # Consistency factor (0-0.1)
        confidence += 0.1 * consistency

        # Volume factor (0-0.1)
        if volume_24h >= self.MIN_VOLUME:
            volume_factor = min(volume_24h / 100.0, 1.0)
            confidence += 0.1 * volume_factor

        return min(confidence, 1.0)

    def should_trade(self, signal: MomentumSignal, min_confidence: float = 0.5) -> bool:
        """
        Determine if a momentum signal is strong enough to trade.

        Args:
            signal: The momentum signal
            min_confidence: Minimum confidence threshold

        Returns:
            True if the signal warrants a trade
        """
        if signal.confidence < min_confidence:
            return False

        # Only trade convergent markets
        if signal.direction == "yes" and signal.current_price < 70:
            return False
        if signal.direction == "no" and signal.current_price > 30:
            return False

        return True
