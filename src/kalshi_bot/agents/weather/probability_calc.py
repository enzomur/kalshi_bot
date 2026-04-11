"""Converts NWS forecasts to probability estimates for Kalshi markets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from kalshi_bot.agents.weather.market_mapper import (
    ThresholdDirection,
    WeatherMarketMapping,
    WeatherType,
)
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.agents.weather.nws_client import Forecast

logger = get_logger(__name__)


@dataclass
class ProbabilityEstimate:
    """Probability estimate with confidence score."""

    ticker: str
    probability: float  # 0-1
    confidence: float   # 0-1 (how confident we are in this estimate)
    forecast_temp: float | None  # Forecasted temperature
    threshold: float    # Market threshold
    hours_until_event: float
    explanation: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "probability": self.probability,
            "confidence": self.confidence,
            "forecast_temp": self.forecast_temp,
            "threshold": self.threshold,
            "hours_until_event": self.hours_until_event,
            "explanation": self.explanation,
        }


class WeatherProbabilityCalculator:
    """
    Calculates probabilities from NWS forecasts.

    Key insights:
    - NWS forecasts have uncertainty that increases with time
    - Temperature forecasts are typically accurate within:
        - 1-2 days: ±2-3°F
        - 3-5 days: ±4-5°F
        - 6-7 days: ±6-8°F
    - Use a normal distribution to model forecast uncertainty
    """

    # Standard deviation of forecast error (in °F) by days out
    TEMP_ERROR_BY_DAY = {
        0: 2.0,   # Same day
        1: 2.5,   # 1 day out
        2: 3.0,   # 2 days out
        3: 4.0,   # 3 days out
        4: 4.5,   # 4 days out
        5: 5.0,   # 5 days out
        6: 6.0,   # 6 days out
        7: 7.0,   # 7+ days out
    }

    # Confidence decay factor per day
    CONFIDENCE_DECAY_PER_DAY = 0.08

    # Base confidence for NWS forecasts
    BASE_CONFIDENCE = 0.85

    def calculate_probability(
        self,
        mapping: WeatherMarketMapping,
        forecast: "Forecast",
    ) -> ProbabilityEstimate | None:
        """
        Calculate probability of a weather event occurring.

        Args:
            mapping: Parsed weather market info
            forecast: NWS forecast for the location

        Returns:
            ProbabilityEstimate or None if unable to calculate
        """
        if mapping.weather_type == WeatherType.TEMPERATURE:
            return self._calculate_temp_probability(mapping, forecast)
        elif mapping.weather_type == WeatherType.RAIN:
            return self._calculate_rain_probability(mapping, forecast)
        elif mapping.weather_type == WeatherType.SNOW:
            return self._calculate_snow_probability(mapping, forecast)
        else:
            logger.warning(f"Unsupported weather type: {mapping.weather_type}")
            return None

    def _calculate_temp_probability(
        self,
        mapping: WeatherMarketMapping,
        forecast: "Forecast",
    ) -> ProbabilityEstimate | None:
        """Calculate probability for temperature threshold markets."""
        event_datetime = datetime.combine(
            mapping.event_date,
            datetime.max.time() if mapping.threshold_direction == ThresholdDirection.ABOVE
            else datetime.min.time()
        )

        hours_until = (event_datetime - datetime.utcnow()).total_seconds() / 3600
        days_until = max(0, int(hours_until / 24))

        # Get forecasted high/low for the date
        high, low = forecast.get_high_low_for_date(event_datetime)

        if mapping.threshold_direction == ThresholdDirection.ABOVE:
            forecast_temp = high
        else:
            forecast_temp = low

        if forecast_temp is None:
            logger.debug(f"No forecast temp available for {mapping.ticker}")
            return None

        # Get forecast uncertainty (standard deviation)
        std_dev = self.TEMP_ERROR_BY_DAY.get(min(days_until, 7), 7.0)

        # Calculate probability using normal CDF
        threshold = mapping.threshold_value
        z_score = (threshold - forecast_temp) / std_dev

        if mapping.threshold_direction == ThresholdDirection.ABOVE:
            # P(temp > threshold) = 1 - CDF(threshold)
            probability = 1 - self._normal_cdf(z_score)
        else:
            # P(temp < threshold) = CDF(threshold)
            probability = self._normal_cdf(z_score)

        # Calculate confidence (decays with forecast distance)
        confidence = self.BASE_CONFIDENCE * (1 - self.CONFIDENCE_DECAY_PER_DAY * days_until)
        confidence = max(0.3, min(1.0, confidence))

        # Adjust for margin from threshold
        margin = abs(forecast_temp - threshold)
        if margin > 10:
            confidence = min(confidence + 0.1, 0.95)  # High confidence if far from threshold
        elif margin < 3:
            confidence = max(confidence - 0.1, 0.3)  # Lower confidence if close

        direction_str = "above" if mapping.threshold_direction == ThresholdDirection.ABOVE else "below"
        explanation = (
            f"Forecast: {forecast_temp}°F, threshold: {threshold}°F {direction_str}. "
            f"{days_until} days out (±{std_dev:.1f}°F uncertainty)."
        )

        return ProbabilityEstimate(
            ticker=mapping.ticker,
            probability=probability,
            confidence=confidence,
            forecast_temp=float(forecast_temp),
            threshold=threshold,
            hours_until_event=hours_until,
            explanation=explanation,
        )

    def _calculate_rain_probability(
        self,
        mapping: WeatherMarketMapping,
        forecast: "Forecast",
    ) -> ProbabilityEstimate | None:
        """Calculate probability for rain markets."""
        event_datetime = datetime.combine(mapping.event_date, datetime.min.time())
        hours_until = (event_datetime - datetime.utcnow()).total_seconds() / 3600
        days_until = max(0, int(hours_until / 24))

        # Find precipitation probability for the day
        max_precip_prob = 0
        for period in forecast.periods:
            if period.start_time.date() == mapping.event_date:
                if period.precipitation_probability is not None:
                    max_precip_prob = max(max_precip_prob, period.precipitation_probability)

        # If no explicit probability, infer from forecast text
        if max_precip_prob == 0:
            for period in forecast.periods:
                if period.start_time.date() == mapping.event_date:
                    forecast_lower = period.short_forecast.lower()
                    if "rain" in forecast_lower or "showers" in forecast_lower:
                        if "slight" in forecast_lower or "chance" in forecast_lower:
                            max_precip_prob = 30
                        elif "likely" in forecast_lower:
                            max_precip_prob = 70
                        else:
                            max_precip_prob = 50
                    elif "cloudy" in forecast_lower:
                        max_precip_prob = 10

        probability = max_precip_prob / 100.0
        confidence = self.BASE_CONFIDENCE * (1 - self.CONFIDENCE_DECAY_PER_DAY * days_until)
        confidence = max(0.4, min(0.9, confidence))

        explanation = f"NWS precipitation probability: {max_precip_prob}% for {mapping.event_date}."

        return ProbabilityEstimate(
            ticker=mapping.ticker,
            probability=probability,
            confidence=confidence,
            forecast_temp=None,
            threshold=0.01,
            hours_until_event=hours_until,
            explanation=explanation,
        )

    def _calculate_snow_probability(
        self,
        mapping: WeatherMarketMapping,
        forecast: "Forecast",
    ) -> ProbabilityEstimate | None:
        """Calculate probability for snow markets."""
        event_datetime = datetime.combine(mapping.event_date, datetime.min.time())
        hours_until = (event_datetime - datetime.utcnow()).total_seconds() / 3600
        days_until = max(0, int(hours_until / 24))

        # Check forecast for snow mentions
        snow_probability = 0
        for period in forecast.periods:
            if period.start_time.date() == mapping.event_date:
                forecast_lower = period.short_forecast.lower()
                if "snow" in forecast_lower:
                    if "heavy" in forecast_lower:
                        snow_probability = 80
                    elif "light" in forecast_lower or "flurries" in forecast_lower:
                        snow_probability = 40
                    elif "chance" in forecast_lower:
                        snow_probability = 30
                    elif "likely" in forecast_lower:
                        snow_probability = 60
                    else:
                        snow_probability = 50

        probability = snow_probability / 100.0
        confidence = self.BASE_CONFIDENCE * (1 - self.CONFIDENCE_DECAY_PER_DAY * days_until)
        confidence = max(0.3, min(0.85, confidence))

        explanation = f"Snow probability estimated at {snow_probability}% from forecast text."

        return ProbabilityEstimate(
            ticker=mapping.ticker,
            probability=probability,
            confidence=confidence,
            forecast_temp=None,
            threshold=mapping.threshold_value,
            hours_until_event=hours_until,
            explanation=explanation,
        )

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """
        Approximation of the standard normal CDF.

        Uses the Abramowitz and Stegun approximation.
        """
        # Constants for approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / math.sqrt(2)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return 0.5 * (1.0 + sign * y)
