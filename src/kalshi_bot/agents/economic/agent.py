"""Economic Data Agent - trades economic markets using FRED API data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from kalshi_bot.agents.base import BaseAgent
from kalshi_bot.agents.economic.calendar import EconomicCalendar, ScheduledRelease
from kalshi_bot.agents.economic.fred_client import FREDClient
from kalshi_bot.agents.economic.indicator_model import IndicatorModel, EconomicPrediction
from kalshi_bot.core.types import ArbitrageOpportunity, ArbitrageType, Side
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class EconomicOpportunity:
    """A trading opportunity from economic data prediction."""

    ticker: str
    release: ScheduledRelease
    prediction: EconomicPrediction
    current_price: int
    edge: float
    side: str
    quantity: int
    expected_profit: float

    def to_arbitrage_opportunity(self) -> ArbitrageOpportunity:
        """Convert to ArbitrageOpportunity for execution."""
        opportunity_id = (
            f"economic_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid4())[:8]}"
        )

        side = Side.YES if self.side == "yes" else Side.NO
        price = self.current_price if side == Side.YES else (100 - self.current_price)

        leg = {
            "market_ticker": self.ticker,
            "side": side.value,
            "price": price,
            "quantity": self.quantity,
            "action": "buy",
        }

        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            arbitrage_type=ArbitrageType.SINGLE_MARKET,
            markets=[self.ticker],
            expected_profit=self.expected_profit * 100,
            expected_profit_pct=self.edge,
            confidence=self.prediction.confidence,
            legs=[leg],
            max_quantity=self.quantity,
            total_cost=self.quantity * price / 100,
            fees=0.07 * (price / 100) * (1 - price / 100) * self.quantity,
            net_profit=self.expected_profit,
            metadata={
                "source": "economic_data",
                "series_id": self.release.series_id,
                "release_name": self.release.release_name,
                "days_until_release": self.release.days_until,
                "prediction_direction": self.prediction.direction,
                "explanation": self.prediction.explanation,
            },
        )


class EconomicDataAgent(BaseAgent):
    """
    Agent that trades economic markets using FRED API leading indicators.

    Strategy:
    - Track upcoming GDP, CPI, jobs releases
    - Use leading indicators to predict direction
    - Pre-position before releases when indicators signal direction
    """

    # Minimum edge to trade
    MIN_EDGE = 0.08

    # Minimum confidence
    MIN_CONFIDENCE = 0.55

    # Maximum days before settlement to trade
    MAX_DAYS_TO_SETTLEMENT = 7.0

    # Price bounds
    MIN_PRICE = 20
    MAX_PRICE = 80

    # Kelly fraction
    KELLY_FRACTION = 0.08

    # Max percentage of capital per trade
    MAX_POSITION_PCT = 0.02

    # Market ticker patterns for economic data
    TICKER_PATTERNS = {
        "GDP": ["GDP", "GDPC", "RGDP"],
        "CPIAUCSL": ["CPI", "INFLATION"],
        "PAYEMS": ["JOBS", "PAYROLL", "NFP"],
        "UNRATE": ["UNEMPLOYMENT", "JOBLESS"],
    }

    def __init__(
        self,
        db: "Database",
        api_client: "KalshiAPIClient",
        fred_api_key: str | None = None,
        min_edge: float = 0.08,
        max_days_to_settlement: float = 7.0,
        update_interval_minutes: int = 60,
        enabled: bool = True,
    ) -> None:
        """
        Initialize Economic Data Agent.

        Args:
            db: Database connection
            api_client: Kalshi API client
            fred_api_key: FRED API key (optional, uses env var if not provided)
            min_edge: Minimum edge threshold
            max_days_to_settlement: Max days before settlement to trade
            update_interval_minutes: How often to update predictions
            enabled: Whether the agent is enabled
        """
        super().__init__(
            db=db,
            name="economic_data",
            update_interval_seconds=update_interval_minutes * 60,
            enabled=enabled,
        )

        self._api_client = api_client
        self._min_edge = min_edge
        self._max_days = max_days_to_settlement

        self._fred = FREDClient(api_key=fred_api_key)
        self._calendar = EconomicCalendar(db)
        self._model = IndicatorModel()

        # Cache of current opportunities
        self._opportunities: dict[str, EconomicOpportunity] = {}

    async def _run_cycle(self) -> None:
        """Execute one cycle of economic data analysis."""
        logger.info("Economic data agent: updating predictions")

        # Load calendar
        await self._calendar.load_releases()

        # Get upcoming releases
        upcoming = self._calendar.get_upcoming(max_days=self._max_days)

        if not upcoming:
            # Try to estimate releases
            upcoming = await self._calendar.estimate_next_releases()
            upcoming = [r for r in upcoming if r.days_until <= self._max_days]

        if not upcoming:
            logger.debug("No upcoming economic releases within window")
            return

        # Fetch leading indicators
        await self._fetch_indicators()

        # Generate predictions
        predictions = []
        for release in upcoming:
            prediction = self._model.predict(release.series_id)
            if prediction and prediction.confidence >= self.MIN_CONFIDENCE:
                predictions.append((release, prediction))

                # Update calendar
                await self._calendar.update_prediction(
                    release.series_id,
                    release.release_time,
                    prediction.direction,
                    prediction.confidence,
                )

        # Find corresponding markets
        opportunities = []
        for release, prediction in predictions:
            opp = await self._find_market_opportunity(release, prediction)
            if opp:
                opportunities.append(opp)
                self._opportunities[opp.ticker] = opp

        # Update metrics
        self._status.metrics = {
            "upcoming_releases": len(upcoming),
            "predictions_made": len(predictions),
            "opportunities_found": len(opportunities),
        }

        logger.info(
            f"Economic agent: {len(predictions)} predictions, "
            f"{len(opportunities)} opportunities"
        )

    async def _fetch_indicators(self) -> None:
        """Fetch leading indicator data from FRED."""
        # Get all required indicators
        all_indicators = set()
        for series in self._model.supported_series:
            all_indicators.update(self._model.get_required_indicators(series))

        for indicator_id in all_indicators:
            try:
                data = await self._fred.get_series(indicator_id, limit=10)
                if data:
                    self._model.cache_indicator(indicator_id, data)
            except Exception as e:
                logger.debug(f"Failed to fetch {indicator_id}: {e}")

    async def _find_market_opportunity(
        self,
        release: ScheduledRelease,
        prediction: EconomicPrediction,
    ) -> EconomicOpportunity | None:
        """Find a Kalshi market matching the economic release."""
        # Search for markets matching the economic data
        patterns = self.TICKER_PATTERNS.get(release.series_id, [release.release_name])

        try:
            markets, _ = await self._api_client.get_markets(
                status="open",
                limit=100,
            )
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return None

        # Find matching market
        for market in markets:
            ticker = market.ticker.upper()
            title = market.title.upper()

            # Check if market matches any pattern
            matched = False
            for pattern in patterns:
                if pattern.upper() in ticker or pattern.upper() in title:
                    matched = True
                    break

            if not matched:
                continue

            # Check settlement timing
            close_time = market.close_time
            if close_time:
                try:
                    # Ensure timezone-aware comparison
                    now = datetime.now(timezone.utc)
                    days_to_close = (close_time - now).total_seconds() / 86400

                    if days_to_close > self._max_days:
                        continue
                except (ValueError, TypeError):
                    continue

            # Get current price
            current_price = market.last_price
            if current_price is None:
                if market.yes_bid is not None and market.yes_ask is not None:
                    current_price = (market.yes_bid + market.yes_ask) // 2
                else:
                    continue

            # Skip extreme prices
            if current_price < self.MIN_PRICE or current_price > self.MAX_PRICE:
                continue

            # Calculate edge based on prediction
            market_prob = current_price / 100.0

            # Prediction confidence maps to expected probability
            if prediction.direction == "above":
                # If we predict "above consensus", YES is more likely
                expected_prob = 0.5 + (prediction.confidence - 0.5) * 0.5
                side = "yes" if expected_prob > market_prob else "no"
            else:
                # If we predict "below consensus", NO is more likely
                expected_prob = 0.5 - (prediction.confidence - 0.5) * 0.5
                side = "no" if expected_prob < market_prob else "yes"

            # Calculate edge
            if side == "yes":
                edge = expected_prob - market_prob
            else:
                edge = market_prob - expected_prob

            if edge < self._min_edge:
                continue

            return EconomicOpportunity(
                ticker=market["ticker"],
                release=release,
                prediction=prediction,
                current_price=current_price,
                edge=edge,
                side=side,
                quantity=0,
                expected_profit=0.0,
            )

        return None

    async def get_trading_opportunities(
        self,
        available_capital: float,
        max_opportunities: int = 5,
    ) -> list[ArbitrageOpportunity]:
        """
        Get economic data trading opportunities for the bot.

        Args:
            available_capital: Available capital for trading
            max_opportunities: Maximum opportunities to return

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []

        # Sort by edge * confidence
        ranked = sorted(
            self._opportunities.values(),
            key=lambda o: o.edge * o.prediction.confidence,
            reverse=True,
        )

        for opp in ranked[:max_opportunities]:
            sized_opp = self._size_opportunity(opp, available_capital)
            if sized_opp and sized_opp.quantity > 0 and sized_opp.expected_profit > 0:
                opportunities.append(sized_opp.to_arbitrage_opportunity())

        return opportunities

    def _size_opportunity(
        self,
        opp: EconomicOpportunity,
        available_capital: float,
    ) -> EconomicOpportunity | None:
        """Calculate position size for an opportunity."""
        if opp.side == "yes":
            win_prob = 0.5 + opp.edge
            entry_price = opp.current_price
        else:
            win_prob = 0.5 + opp.edge
            entry_price = 100 - opp.current_price

        lose_prob = 1 - win_prob
        payout_ratio = (100 - entry_price) / entry_price if entry_price > 0 else 0

        if payout_ratio > 0:
            kelly = (payout_ratio * win_prob - lose_prob) / payout_ratio
            kelly = max(0, min(kelly, self.KELLY_FRACTION))
        else:
            kelly = 0

        if kelly <= 0:
            return None

        position_value = min(
            available_capital * kelly,
            available_capital * self.MAX_POSITION_PCT,
        )

        cost_per_contract = entry_price / 100.0
        quantity = int(position_value / cost_per_contract) if cost_per_contract > 0 else 0

        if quantity < 1:
            return None

        expected_profit = (win_prob * quantity) - (quantity * cost_per_contract)
        fees = 0.07 * (entry_price / 100) * (1 - entry_price / 100) * quantity
        expected_profit -= fees

        if expected_profit <= 0:
            return None

        logger.info(
            f"Economic opportunity: {opp.ticker} {opp.side.upper()} "
            f"@ {opp.current_price}c, {opp.prediction.direction} prediction, "
            f"conf={opp.prediction.confidence:.0%}, edge={opp.edge:.1%}, qty={quantity}"
        )

        return EconomicOpportunity(
            ticker=opp.ticker,
            release=opp.release,
            prediction=opp.prediction,
            current_price=opp.current_price,
            edge=opp.edge,
            side=opp.side,
            quantity=quantity,
            expected_profit=expected_profit,
        )

    async def close(self) -> None:
        """Clean up resources."""
        await self._fred.close()

    def get_status(self) -> dict[str, Any]:
        """Get agent status with economic-specific metrics."""
        status = super().get_status()
        status["upcoming_releases"] = len(self._calendar.upcoming_releases)
        status["active_opportunities"] = len(self._opportunities)
        status["supported_series"] = self._model.supported_series
        return status
