"""Edge predictor for ML-based trading opportunities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from kalshi_bot.core.types import ArbitrageOpportunity, ArbitrageType, Side
from kalshi_bot.ml.feature_engineer import FeatureEngineer, FEATURE_NAMES
from kalshi_bot.ml.models import BasePredictionModel
from kalshi_bot.ml.training.trainer import ModelTrainer
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class EdgePrediction:
    """A prediction with edge calculation."""

    ticker: str
    event_ticker: str
    model_prob: float  # Model's P(YES)
    market_price: int  # Current market price in cents
    edge: float  # model_prob - market_price/100
    confidence: float  # Feature confidence
    side: str  # 'yes' or 'no' - which side has edge
    predicted_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def market_prob(self) -> float:
        """Market implied probability."""
        return self.market_price / 100.0

    @property
    def abs_edge(self) -> float:
        """Absolute edge magnitude."""
        return abs(self.edge)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "event_ticker": self.event_ticker,
            "model_prob": self.model_prob,
            "market_price": self.market_price,
            "market_prob": self.market_prob,
            "edge": self.edge,
            "abs_edge": self.abs_edge,
            "side": self.side,
            "confidence": self.confidence,
            "predicted_at": self.predicted_at.isoformat(),
        }


@dataclass
class TradingOpportunity:
    """An ML-identified trading opportunity."""

    prediction: EdgePrediction
    quantity: int
    expected_payout: float  # If prediction is correct
    expected_cost: float  # Cost to enter position
    fees: float  # Expected trading fees
    expected_profit: float  # Expected profit if correct
    kelly_fraction: float  # Kelly-recommended position size

    def to_arbitrage_opportunity(self) -> ArbitrageOpportunity:
        """Convert to ArbitrageOpportunity for integration with existing execution."""
        opportunity_id = f"ml_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid4())[:8]}"

        # Determine trade leg
        side = Side.YES if self.prediction.side == "yes" else Side.NO
        price = self.prediction.market_price if side == Side.YES else (100 - self.prediction.market_price)

        leg = {
            "market_ticker": self.prediction.ticker,
            "side": side.value,
            "price": price,
            "quantity": self.quantity,
            "action": "buy",
        }

        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            arbitrage_type=ArbitrageType.SINGLE_MARKET,  # ML edge is similar to single market
            markets=[self.prediction.ticker],
            expected_profit=self.expected_profit * 100,  # Convert to cents
            expected_profit_pct=self.expected_profit / self.expected_cost if self.expected_cost > 0 else 0,
            confidence=self.prediction.confidence,
            legs=[leg],
            max_quantity=self.quantity,
            total_cost=self.expected_cost,
            fees=self.fees,
            net_profit=self.expected_profit - self.fees,
            metadata={
                "source": "ml_edge",
                "model_prob": self.prediction.model_prob,
                "market_prob": self.prediction.market_prob,
                "edge": self.prediction.edge,
                "kelly_fraction": self.kelly_fraction,
            },
        )


class EdgePredictor:
    """
    Predicts trading edges using ML models.

    Compares model probability estimates against market prices to
    identify opportunities where the model disagrees with the market.
    """

    # Minimum edge required to consider trading (5%)
    MIN_EDGE = 0.05

    # Minimum confidence for predictions
    MIN_CONFIDENCE = 0.60

    # Kelly fraction cap
    MAX_KELLY_FRACTION = 0.25

    def __init__(
        self,
        db: Database,
        api_client: KalshiAPIClient,
        min_edge: float | None = None,
        min_confidence: float | None = None,
    ) -> None:
        """
        Initialize edge predictor.

        Args:
            db: Database connection
            api_client: Kalshi API client
            min_edge: Minimum edge threshold (default 0.05)
            min_confidence: Minimum confidence threshold (default 0.60)
        """
        self._db = db
        self._api_client = api_client
        self._feature_engineer = FeatureEngineer(db)
        self._trainer = ModelTrainer(db)

        self._min_edge = min_edge or self.MIN_EDGE
        self._min_confidence = min_confidence or self.MIN_CONFIDENCE

        self._model: BasePredictionModel | None = None
        self._model_loaded_at: datetime | None = None
        self._predictions_made = 0

    async def ensure_model_loaded(self) -> bool:
        """Ensure we have a model loaded. Returns True if model available."""
        # Reload model periodically (every hour) to pick up new training
        if self._model is not None and self._model_loaded_at:
            hours_since_load = (datetime.utcnow() - self._model_loaded_at).total_seconds() / 3600
            if hours_since_load < 1:
                return True

        # Try to load active model
        self._model = await self._trainer.get_active_model("logistic")
        if self._model is None:
            self._model = await self._trainer.get_active_model("gradient_boost")

        if self._model:
            self._model_loaded_at = datetime.utcnow()
            logger.info(f"Loaded model: {self._model.model_id}")
            return True

        logger.debug("No active model available")
        return False

    async def predict_edge(self, ticker: str) -> EdgePrediction | None:
        """
        Predict edge for a single market.

        Args:
            ticker: Market ticker

        Returns:
            EdgePrediction or None if cannot predict
        """
        if not await self.ensure_model_loaded():
            return None

        # Get current market data
        try:
            market = await self._api_client.get_market(ticker)
        except Exception as e:
            logger.debug(f"Failed to get market {ticker}: {e}")
            return None

        # Get market price
        market_price = market.last_price
        if market_price is None:
            if market.yes_bid is not None and market.yes_ask is not None:
                market_price = (market.yes_bid + market.yes_ask) // 2
            else:
                return None

        # Compute features
        features = await self._feature_engineer.compute_features(ticker)
        if features is None:
            return None

        # Make prediction
        X = features.to_array().reshape(1, -1)
        model_prob = float(self._model.predict_proba(X)[0])

        # Calculate edge
        market_prob = market_price / 100.0
        edge = model_prob - market_prob

        # Determine which side has edge
        if edge > 0:
            side = "yes"  # Model thinks YES is underpriced
        else:
            side = "no"  # Model thinks NO is underpriced
            edge = -edge  # Make edge positive for NO side

        prediction = EdgePrediction(
            ticker=ticker,
            event_ticker=market.event_ticker,
            model_prob=model_prob,
            market_price=market_price,
            edge=edge if side == "yes" else -edge,  # Store signed edge
            confidence=features.feature_confidence,
            side=side,
        )

        return prediction

    async def predict_edges(
        self,
        tickers: list[str],
    ) -> list[EdgePrediction]:
        """
        Predict edges for multiple markets.

        Args:
            tickers: List of market tickers

        Returns:
            List of EdgePredictions (only markets with sufficient edge)
        """
        predictions = []

        for ticker in tickers:
            prediction = await self.predict_edge(ticker)
            if prediction and prediction.abs_edge >= self._min_edge:
                predictions.append(prediction)

        # Sort by absolute edge (highest first)
        predictions.sort(key=lambda p: p.abs_edge, reverse=True)
        return predictions

    async def get_trading_opportunities(
        self,
        available_capital: float,
        max_opportunities: int = 10,
    ) -> list[TradingOpportunity]:
        """
        Get trading opportunities based on ML predictions.

        Args:
            available_capital: Capital available for trading
            max_opportunities: Maximum opportunities to return

        Returns:
            List of TradingOpportunities sorted by expected value
        """
        if not await self.ensure_model_loaded():
            return []

        # Get all open markets with snapshots
        tickers = await self._feature_engineer._db.fetch_all(
            """
            SELECT DISTINCT ms.ticker
            FROM market_snapshots ms
            WHERE ms.ticker NOT IN (SELECT ticker FROM market_settlements)
              AND ms.snapshot_at >= datetime('now', '-1 hour')
            """
        )

        ticker_list = [t["ticker"] for t in tickers]
        logger.info(f"Evaluating {len(ticker_list)} markets for ML opportunities")

        # Get predictions with edge
        predictions = await self.predict_edges(ticker_list)

        # Filter by confidence
        predictions = [p for p in predictions if p.confidence >= self._min_confidence]

        opportunities = []
        for pred in predictions[:max_opportunities * 2]:  # Get more to filter
            opp = await self._create_opportunity(pred, available_capital)
            if opp and opp.expected_profit > 0:
                opportunities.append(opp)

                # Log prediction for tracking
                await self._log_prediction(pred, opp)

        # Sort by expected profit
        opportunities.sort(key=lambda o: o.expected_profit, reverse=True)

        self._predictions_made += len(opportunities)
        return opportunities[:max_opportunities]

    async def _create_opportunity(
        self,
        prediction: EdgePrediction,
        available_capital: float,
    ) -> TradingOpportunity | None:
        """Create a trading opportunity from a prediction."""
        # Calculate Kelly fraction for position sizing
        # Kelly: f* = (bp - q) / b where b = odds, p = win prob, q = lose prob
        if prediction.side == "yes":
            win_prob = prediction.model_prob
            # For YES side: if we pay X cents, we win (100-X) cents
            price = prediction.market_price
        else:
            win_prob = 1 - prediction.model_prob
            # For NO side: if we pay (100-X) cents, we win X cents
            price = 100 - prediction.market_price

        lose_prob = 1 - win_prob
        payout_ratio = (100 - price) / price if price > 0 else 0

        if payout_ratio > 0:
            kelly = (payout_ratio * win_prob - lose_prob) / payout_ratio
        else:
            kelly = 0

        # Cap Kelly fraction
        kelly = max(0, min(kelly, self.MAX_KELLY_FRACTION))

        if kelly <= 0:
            return None

        # Calculate position size
        position_value = available_capital * kelly
        cost_per_contract = price / 100.0  # Convert cents to dollars
        quantity = int(position_value / cost_per_contract) if cost_per_contract > 0 else 0

        if quantity < 1:
            return None

        # Calculate expected values
        expected_cost = quantity * cost_per_contract
        expected_payout = quantity  # Each contract pays $1 if correct
        fees = self._calculate_fees(price, quantity)

        # Expected profit = (win_prob * payout) - (lose_prob * cost) - fees
        expected_profit = (win_prob * expected_payout) - (lose_prob * expected_cost) - fees

        return TradingOpportunity(
            prediction=prediction,
            quantity=quantity,
            expected_payout=expected_payout,
            expected_cost=expected_cost,
            fees=fees,
            expected_profit=expected_profit,
            kelly_fraction=kelly,
        )

    def _calculate_fees(self, price: int, quantity: int) -> float:
        """Calculate trading fees."""
        # Kalshi fee: 0.07 * P * (1-P) per contract
        p = price / 100.0
        fee_per_contract = 0.07 * p * (1 - p)
        return fee_per_contract * quantity

    async def _log_prediction(
        self,
        prediction: EdgePrediction,
        opportunity: TradingOpportunity,
    ) -> None:
        """Log prediction for accuracy tracking."""
        prediction_id = f"pred_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid4())[:8]}"

        await self._db.execute(
            """
            INSERT INTO ml_predictions (
                prediction_id, model_id, ticker, event_ticker,
                predicted_prob_yes, market_price, edge, confidence,
                features_used, trade_side, trade_quantity, predicted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction_id,
                self._model.model_id if self._model else "unknown",
                prediction.ticker,
                prediction.event_ticker,
                prediction.model_prob,
                prediction.market_price,
                prediction.edge,
                prediction.confidence,
                json.dumps({"feature_names": FEATURE_NAMES}),
                prediction.side,
                opportunity.quantity,
                prediction.predicted_at.isoformat(),
            ),
        )

    async def get_recent_predictions(
        self,
        limit: int = 50,
        settled_only: bool = False,
    ) -> list[dict]:
        """Get recent predictions for review."""
        if settled_only:
            return await self._db.fetch_all(
                """
                SELECT * FROM ml_predictions
                WHERE actual_outcome IS NOT NULL
                ORDER BY predicted_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        return await self._db.fetch_all(
            """
            SELECT * FROM ml_predictions
            ORDER BY predicted_at DESC
            LIMIT ?
            """,
            (limit,),
        )

    async def get_accuracy_stats(self, window_size: int = 50) -> dict:
        """Get prediction accuracy statistics."""
        result = await self._db.fetch_one(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(edge) as avg_edge,
                AVG(ABS(predicted_prob_yes - 0.5)) as avg_confidence
            FROM (
                SELECT * FROM ml_predictions
                WHERE actual_outcome IS NOT NULL
                ORDER BY predicted_at DESC
                LIMIT ?
            )
            """,
            (window_size,),
        )

        if result and result["total"] > 0:
            return {
                "total_predictions": result["total"],
                "correct_predictions": result["correct"],
                "accuracy": result["correct"] / result["total"],
                "avg_edge": result["avg_edge"],
                "avg_confidence": result["avg_confidence"],
            }

        return {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": None,
            "avg_edge": None,
            "avg_confidence": None,
        }

    def get_status(self) -> dict:
        """Get predictor status."""
        return {
            "model_loaded": self._model is not None,
            "model_id": self._model.model_id if self._model else None,
            "model_loaded_at": self._model_loaded_at.isoformat() if self._model_loaded_at else None,
            "predictions_made": self._predictions_made,
            "thresholds": {
                "min_edge": self._min_edge,
                "min_confidence": self._min_confidence,
                "max_kelly_fraction": self.MAX_KELLY_FRACTION,
            },
        }
