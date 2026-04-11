"""Generates candidate trading signals for backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class SignalCategory(str, Enum):
    """Categories of trading signals."""

    PRICE = "price"
    VOLUME = "volume"
    MOMENTUM = "momentum"
    WEATHER = "weather"
    TEMPORAL = "temporal"
    SENTIMENT = "sentiment"


@dataclass
class SignalCandidate:
    """A candidate trading signal to be tested."""

    signal_id: str
    name: str
    description: str
    category: SignalCategory
    feature_formula: str  # Python expression to compute the signal
    expected_direction: int  # 1 for bullish (YES), -1 for bearish (NO), 0 for neutral
    min_threshold: float  # Minimum signal value to trigger
    max_threshold: float  # Maximum signal value to trigger (for range signals)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "feature_formula": self.feature_formula,
            "expected_direction": self.expected_direction,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
        }


class SignalGenerator:
    """
    Generates candidate trading signals.

    Signals are mathematical expressions computed from market features
    that may have predictive power for market outcomes.
    """

    # Pre-defined signal candidates
    BUILTIN_SIGNALS = [
        # Price-based signals
        SignalCandidate(
            signal_id="momentum_reversal_1h",
            name="1-Hour Momentum Reversal",
            description="Markets with strong recent momentum may revert",
            category=SignalCategory.MOMENTUM,
            feature_formula="price_momentum_1h * -1",
            expected_direction=-1,
            min_threshold=0.05,
            max_threshold=0.50,
        ),
        SignalCandidate(
            signal_id="volatility_squeeze",
            name="Volatility Squeeze",
            description="Low volatility often precedes big moves",
            category=SignalCategory.PRICE,
            feature_formula="1.0 - price_volatility",
            expected_direction=0,  # Direction unclear
            min_threshold=0.90,
            max_threshold=1.0,
        ),
        SignalCandidate(
            signal_id="volume_spike",
            name="Volume Spike",
            description="Unusual volume may indicate informed trading",
            category=SignalCategory.VOLUME,
            feature_formula="volume_momentum",
            expected_direction=1,
            min_threshold=0.30,
            max_threshold=1.0,
        ),
        SignalCandidate(
            signal_id="price_near_50",
            name="Price Near 50%",
            description="Markets near 50% have highest uncertainty",
            category=SignalCategory.PRICE,
            feature_formula="1.0 - abs(current_price - 0.5) * 2",
            expected_direction=0,
            min_threshold=0.80,
            max_threshold=1.0,
        ),
        SignalCandidate(
            signal_id="end_of_day_drift",
            name="End of Day Drift",
            description="Markets may drift toward resolution late in day",
            category=SignalCategory.TEMPORAL,
            feature_formula="(hour_of_day / 23.0) * (1.0 - hours_to_expiry)",
            expected_direction=1,
            min_threshold=0.50,
            max_threshold=1.0,
        ),
        # Weather-specific signals
        SignalCandidate(
            signal_id="nws_high_confidence",
            name="NWS High Confidence",
            description="High NWS confidence suggests reliable forecast",
            category=SignalCategory.WEATHER,
            feature_formula="nws_confidence",
            expected_direction=1,
            min_threshold=0.80,
            max_threshold=1.0,
        ),
        SignalCandidate(
            signal_id="nws_price_divergence",
            name="NWS vs Market Divergence",
            description="Large gap between NWS probability and market price",
            category=SignalCategory.WEATHER,
            feature_formula="abs(nws_probability - current_price)",
            expected_direction=0,
            min_threshold=0.15,
            max_threshold=1.0,
        ),
        SignalCandidate(
            signal_id="weather_late_trade",
            name="Weather Late Trading",
            description="Weather markets close to expiry with fresh forecast",
            category=SignalCategory.WEATHER,
            feature_formula="forecast_recency * (1.0 - forecast_hours_out)",
            expected_direction=1,
            min_threshold=0.50,
            max_threshold=1.0,
        ),
        SignalCandidate(
            signal_id="momentum_continuation_24h",
            name="24-Hour Momentum Continuation",
            description="Strong 24h momentum may continue",
            category=SignalCategory.MOMENTUM,
            feature_formula="price_momentum_24h",
            expected_direction=1,
            min_threshold=0.10,
            max_threshold=0.50,
        ),
        SignalCandidate(
            signal_id="spread_tightening",
            name="Spread Tightening",
            description="Tightening spreads indicate increased confidence",
            category=SignalCategory.VOLUME,
            feature_formula="1.0 - spread",
            expected_direction=1,
            min_threshold=0.80,
            max_threshold=1.0,
        ),
    ]

    def __init__(self) -> None:
        """Initialize the signal generator."""
        self._custom_signals: list[SignalCandidate] = []

    def get_all_signals(self) -> list[SignalCandidate]:
        """Get all signal candidates (builtin + custom)."""
        return self.BUILTIN_SIGNALS + self._custom_signals

    def get_signals_by_category(
        self,
        category: SignalCategory,
    ) -> list[SignalCandidate]:
        """Get signals of a specific category."""
        return [s for s in self.get_all_signals() if s.category == category]

    def add_custom_signal(self, signal: SignalCandidate) -> None:
        """Add a custom signal candidate."""
        self._custom_signals.append(signal)
        logger.info(f"Added custom signal: {signal.name}")

    def compute_signal_value(
        self,
        signal: SignalCandidate,
        features: dict[str, float],
    ) -> float | None:
        """
        Compute a signal value from market features.

        Args:
            signal: Signal candidate
            features: Market features dict

        Returns:
            Signal value or None if computation fails
        """
        try:
            # Create a safe evaluation context with just the features
            safe_dict = {
                "abs": abs,
                "min": min,
                "max": max,
            }
            safe_dict.update(features)

            value = eval(signal.feature_formula, {"__builtins__": {}}, safe_dict)
            return float(value)
        except Exception as e:
            logger.debug(f"Signal computation failed for {signal.signal_id}: {e}")
            return None

    def is_signal_triggered(
        self,
        signal: SignalCandidate,
        features: dict[str, float],
    ) -> bool:
        """Check if a signal is triggered by the given features."""
        value = self.compute_signal_value(signal, features)
        if value is None:
            return False

        return signal.min_threshold <= value <= signal.max_threshold
