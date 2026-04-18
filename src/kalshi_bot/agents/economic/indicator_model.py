"""Indicator model for predicting economic data direction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from kalshi_bot.agents.economic.fred_client import EconomicDataPoint

if TYPE_CHECKING:
    pass


@dataclass
class EconomicPrediction:
    """A prediction for an economic data release."""

    series_id: str
    direction: str  # 'above' or 'below' consensus
    confidence: float
    explanation: str
    leading_indicators: dict[str, float]


class IndicatorModel:
    """
    Model for predicting economic data releases using leading indicators.

    Uses relationships between economic series to predict upcoming releases.

    Key relationships:
    - GDP: Industrial production, retail sales, employment lead GDP
    - CPI: Energy prices, wage growth lead inflation
    - Jobs: Initial claims, job openings lead payrolls
    """

    # Leading indicators for each target series
    LEADING_INDICATORS = {
        "GDP": [
            ("INDPRO", 0.3),  # Industrial Production
            ("RSXFS", 0.2),  # Retail Sales
            ("PAYEMS", 0.2),  # Employment
            ("UMCSENT", 0.1),  # Consumer Sentiment
            ("T10Y2Y", 0.2),  # Yield curve
        ],
        "CPIAUCSL": [
            ("DCOILWTICO", 0.3),  # Oil prices
            ("AHETPI", 0.3),  # Hourly earnings
            ("PPIACO", 0.2),  # Producer prices
            ("M2SL", 0.2),  # Money supply
        ],
        "PAYEMS": [
            ("ICSA", -0.4),  # Initial claims (inverse)
            ("JTSJOL", 0.3),  # Job openings
            ("AWHMAN", 0.2),  # Hours worked
            ("INDPRO", 0.1),  # Industrial production
        ],
        "UNRATE": [
            ("ICSA", 0.5),  # Initial claims
            ("PAYEMS", -0.3),  # Employment (inverse)
            ("JTSJOL", -0.2),  # Job openings (inverse)
        ],
    }

    # Historical accuracy by series
    HISTORICAL_ACCURACY = {
        "GDP": 0.55,
        "CPIAUCSL": 0.60,
        "PAYEMS": 0.55,
        "UNRATE": 0.58,
    }

    def __init__(self) -> None:
        """Initialize indicator model."""
        self._indicator_cache: dict[str, list[EconomicDataPoint]] = {}

    def cache_indicator(
        self,
        series_id: str,
        data_points: list[EconomicDataPoint],
    ) -> None:
        """Cache indicator data for predictions."""
        self._indicator_cache[series_id] = data_points

    def predict(
        self,
        target_series: str,
        consensus_value: float | None = None,
    ) -> EconomicPrediction | None:
        """
        Predict direction for an economic release.

        Args:
            target_series: The series being predicted
            consensus_value: Market consensus estimate

        Returns:
            EconomicPrediction or None if insufficient data
        """
        if target_series not in self.LEADING_INDICATORS:
            return None

        indicators = self.LEADING_INDICATORS[target_series]
        signals = {}
        total_weight = 0.0
        weighted_signal = 0.0

        for indicator_id, weight in indicators:
            if indicator_id not in self._indicator_cache:
                continue

            data = self._indicator_cache[indicator_id]
            if len(data) < 2:
                continue

            # Calculate recent trend
            recent = data[0].value
            previous = data[1].value

            if previous != 0:
                pct_change = (recent - previous) / abs(previous)

                # Normalize signal to -1 to 1
                signal = max(-1, min(1, pct_change * 10))

                # Apply weight (negative weight inverts relationship)
                weighted_signal += signal * weight
                total_weight += abs(weight)
                signals[indicator_id] = pct_change

        if total_weight == 0:
            return None

        # Normalize
        final_signal = weighted_signal / total_weight

        # Determine direction and confidence
        if final_signal > 0.1:
            direction = "above"
            confidence = min(0.7, 0.5 + abs(final_signal) * 0.3)
        elif final_signal < -0.1:
            direction = "below"
            confidence = min(0.7, 0.5 + abs(final_signal) * 0.3)
        else:
            # Signal too weak
            return None

        # Adjust confidence by historical accuracy
        base_accuracy = self.HISTORICAL_ACCURACY.get(target_series, 0.50)
        confidence = (confidence + base_accuracy) / 2

        explanation = self._build_explanation(target_series, direction, signals)

        return EconomicPrediction(
            series_id=target_series,
            direction=direction,
            confidence=confidence,
            explanation=explanation,
            leading_indicators=signals,
        )

    def _build_explanation(
        self,
        target_series: str,
        direction: str,
        signals: dict[str, float],
    ) -> str:
        """Build human-readable explanation for prediction."""
        parts = [f"Predicting {target_series} will come in {direction} consensus."]

        for indicator, change in signals.items():
            if change > 0:
                parts.append(f"{indicator} up {change:.1%}")
            else:
                parts.append(f"{indicator} down {abs(change):.1%}")

        return " ".join(parts)

    def get_required_indicators(self, target_series: str) -> list[str]:
        """Get list of indicator series needed for a target."""
        if target_series not in self.LEADING_INDICATORS:
            return []

        return [ind_id for ind_id, _ in self.LEADING_INDICATORS[target_series]]

    @property
    def supported_series(self) -> list[str]:
        """Get list of series we can make predictions for."""
        return list(self.LEADING_INDICATORS.keys())
