"""ML-based opportunity detection strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kalshi_bot.arbitrage.strategies.base import ArbitrageStrategy
from kalshi_bot.core.types import ArbitrageOpportunity, MarketData, OrderBook
from kalshi_bot.ml.inference.predictor import EdgePredictor
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.config.settings import Settings
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


class MLOpportunityStrategy(ArbitrageStrategy):
    """
    ML-based opportunity detection strategy.

    Uses trained ML models to identify mispriced markets where
    the model's probability estimate differs from the market price.
    Integrates with existing ArbitrageDetector infrastructure.
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        api_client: KalshiAPIClient,
    ) -> None:
        """
        Initialize ML strategy.

        Args:
            settings: Application settings
            db: Database connection
            api_client: Kalshi API client
        """
        super().__init__(settings)

        # Get ML settings
        ml_settings = getattr(settings, 'ml', None)

        min_edge = ml_settings.min_edge_threshold if ml_settings else 0.05
        min_confidence = ml_settings.min_confidence if ml_settings else 0.60

        self._predictor = EdgePredictor(
            db=db,
            api_client=api_client,
            min_edge=min_edge,
            min_confidence=min_confidence,
        )

        self._db = db
        self._api_client = api_client

        # Track strategy performance
        self._opportunities_found = 0
        self._opportunities_validated = 0

    async def detect(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
    ) -> list[ArbitrageOpportunity]:
        """
        Detect ML-based trading opportunities.

        This method is called by ArbitrageDetector as part of the
        regular opportunity scanning cycle.

        Args:
            markets: List of market data
            orderbooks: Dictionary of order books

        Returns:
            List of opportunities identified by ML
        """
        # Skip if no model available
        if not await self._predictor.ensure_model_loaded():
            logger.debug("ML strategy skipped: no model available")
            return []

        # Get available capital from portfolio (need to access this somehow)
        # For now, use a default value - bot integration will provide actual value
        available_capital = 100.0  # Default, will be overridden by bot

        # Get ML opportunities
        ml_opportunities = await self._predictor.get_trading_opportunities(
            available_capital=available_capital,
            max_opportunities=10,
        )

        opportunities = []
        for ml_opp in ml_opportunities:
            # Convert to ArbitrageOpportunity
            arb_opp = ml_opp.to_arbitrage_opportunity()

            # Validate opportunity
            if self.validate_opportunity(arb_opp):
                opportunities.append(arb_opp)
                self._opportunities_validated += 1

        self._opportunities_found += len(opportunities)

        if opportunities:
            logger.info(f"ML strategy found {len(opportunities)} opportunities")

        return opportunities

    async def detect_with_capital(
        self,
        available_capital: float,
        max_opportunities: int = 10,
    ) -> list[ArbitrageOpportunity]:
        """
        Detect opportunities with specified capital.

        Alternative method that allows specifying available capital directly.

        Args:
            available_capital: Capital available for trading
            max_opportunities: Maximum opportunities to return

        Returns:
            List of validated opportunities
        """
        if not await self._predictor.ensure_model_loaded():
            return []

        ml_opportunities = await self._predictor.get_trading_opportunities(
            available_capital=available_capital,
            max_opportunities=max_opportunities,
        )

        opportunities = []
        for ml_opp in ml_opportunities:
            arb_opp = ml_opp.to_arbitrage_opportunity()
            if self.validate_opportunity(arb_opp):
                opportunities.append(arb_opp)

        return opportunities

    def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """
        Validate ML opportunity meets minimum requirements.

        Applies standard arbitrage validation plus ML-specific checks.
        """
        # Standard validation
        if not super().validate_opportunity(opportunity):
            return False

        # ML-specific validation
        metadata = opportunity.metadata

        # Check edge is sufficient
        edge = metadata.get("edge", 0)
        if abs(edge) < self._min_profit_cents / 100:  # Convert cents to probability
            logger.debug(f"ML opportunity rejected: edge {edge:.3f} too small")
            return False

        # Check Kelly fraction is reasonable
        kelly = metadata.get("kelly_fraction", 0)
        if kelly <= 0:
            logger.debug("ML opportunity rejected: negative Kelly fraction")
            return False

        return True

    async def get_prediction_accuracy(self) -> dict:
        """Get current prediction accuracy statistics."""
        return await self._predictor.get_accuracy_stats()

    def get_status(self) -> dict:
        """Get strategy status."""
        predictor_status = self._predictor.get_status()

        return {
            "strategy_type": "ml_opportunity",
            "opportunities_found": self._opportunities_found,
            "opportunities_validated": self._opportunities_validated,
            "predictor": predictor_status,
        }
