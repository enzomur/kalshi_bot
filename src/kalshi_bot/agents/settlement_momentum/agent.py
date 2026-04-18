"""Settlement Momentum Agent - trades markets converging to settlement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from kalshi_bot.agents.base import BaseAgent
from kalshi_bot.agents.settlement_momentum.momentum_calc import (
    MomentumCalculator,
    MomentumSignal,
)
from kalshi_bot.core.types import ArbitrageOpportunity, ArbitrageType, Side
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class SettlementOpportunity:
    """A trading opportunity from settlement momentum."""

    signal: MomentumSignal
    edge: float
    expected_profit: float
    quantity: int

    def to_arbitrage_opportunity(self) -> ArbitrageOpportunity:
        """Convert to ArbitrageOpportunity for execution."""
        opportunity_id = (
            f"settlement_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{str(uuid4())[:8]}"
        )

        side = Side.YES if self.signal.direction == "yes" else Side.NO
        price = (
            self.signal.current_price
            if side == Side.YES
            else (100 - self.signal.current_price)
        )

        leg = {
            "market_ticker": self.signal.ticker,
            "side": side.value,
            "price": price,
            "quantity": self.quantity,
            "action": "buy",
        }

        return ArbitrageOpportunity(
            opportunity_id=opportunity_id,
            arbitrage_type=ArbitrageType.SINGLE_MARKET,
            markets=[self.signal.ticker],
            expected_profit=self.expected_profit * 100,
            expected_profit_pct=self.edge,
            confidence=self.signal.confidence,
            legs=[leg],
            max_quantity=self.quantity,
            total_cost=self.quantity * price / 100,
            fees=0.07 * (price / 100) * (1 - price / 100) * self.quantity,
            net_profit=self.expected_profit,
            metadata={
                "source": "settlement_momentum",
                "momentum": self.signal.momentum,
                "hours_to_settlement": self.signal.hours_to_settlement,
                "direction": self.signal.direction,
                "price_change_6h": (
                    self.signal.current_price - self.signal.price_6h_ago
                    if self.signal.price_6h_ago
                    else None
                ),
            },
        )


class SettlementMomentumAgent(BaseAgent):
    """
    Agent that trades markets in final 24-48 hours where price is converging.

    Strategy:
    - Price > 70 and rising -> Buy YES (market thinks YES will settle)
    - Price < 30 and falling -> Buy NO (market thinks NO will settle)

    Uses existing market_snapshots data to calculate momentum.
    """

    # Price thresholds for convergence trades
    MIN_PRICE_FOR_YES = 70  # Only buy YES if price >= 70
    MAX_PRICE_FOR_NO = 30  # Only buy NO if price <= 30

    # Minimum momentum strength
    MIN_MOMENTUM = 0.10

    # Kelly fraction for position sizing
    KELLY_FRACTION = 0.10

    # Max percentage of capital per trade
    MAX_POSITION_PCT = 0.02

    def __init__(
        self,
        db: "Database",
        api_client: "KalshiAPIClient",
        max_hours_to_settlement: float = 48.0,
        min_momentum_strength: float = 0.10,
        min_price_for_yes: int = 70,
        max_price_for_no: int = 30,
        update_interval_minutes: int = 15,
        enabled: bool = True,
    ) -> None:
        """
        Initialize Settlement Momentum Agent.

        Args:
            db: Database connection
            api_client: Kalshi API client
            max_hours_to_settlement: Maximum hours before settlement to trade
            min_momentum_strength: Minimum momentum threshold
            min_price_for_yes: Minimum price to buy YES
            max_price_for_no: Maximum price to buy NO
            update_interval_minutes: How often to scan for opportunities
            enabled: Whether the agent is enabled
        """
        super().__init__(
            db=db,
            name="settlement_momentum",
            update_interval_seconds=update_interval_minutes * 60,
            enabled=enabled,
        )

        self._api_client = api_client
        self._max_hours = max_hours_to_settlement
        self._min_momentum = min_momentum_strength
        self._min_price_yes = min_price_for_yes
        self._max_price_no = max_price_for_no

        self._momentum_calc = MomentumCalculator(
            min_hours=1.0,
            max_hours=max_hours_to_settlement,
        )

        # Cache of current momentum signals
        self._signals: dict[str, MomentumSignal] = {}

    async def _run_cycle(self) -> None:
        """Execute one cycle of momentum analysis."""
        logger.info("Settlement momentum agent: analyzing markets")

        # Get markets settling soon
        markets = await self._get_settling_markets()
        if not markets:
            logger.debug("No markets settling within window")
            return

        # Calculate momentum for each
        signals = []
        for market in markets:
            signal = await self._analyze_market(market)
            if signal and self._momentum_calc.should_trade(signal, self._min_momentum):
                signals.append(signal)
                self._signals[market["ticker"]] = signal

        # Store signals in database
        await self._store_signals(signals)

        # Update metrics
        self._status.metrics = {
            "markets_analyzed": len(markets),
            "signals_generated": len(signals),
            "yes_signals": sum(1 for s in signals if s.direction == "yes"),
            "no_signals": sum(1 for s in signals if s.direction == "no"),
        }

        logger.info(
            f"Settlement momentum: analyzed {len(markets)} markets, "
            f"found {len(signals)} tradeable signals"
        )

    async def _get_settling_markets(self) -> list[dict]:
        """Get markets that are settling within the time window."""
        now = datetime.now(timezone.utc)
        max_settlement = now + timedelta(hours=self._max_hours)
        min_settlement = now + timedelta(hours=1)

        try:
            # Get markets from Kalshi
            markets, _ = await self._api_client.get_markets(
                status="open",
                limit=200,
            )
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

        settling_markets = []

        for market in markets:
            close_time = market.close_time
            if not close_time:
                continue

            # close_time is already a datetime from MarketData
            if min_settlement <= close_time <= max_settlement:
                hours_to_settlement = (close_time - now).total_seconds() / 3600
                # Convert to dict with extra fields for downstream processing
                market_dict = {
                    "ticker": market.ticker,
                    "event_ticker": market.event_ticker,
                    "last_price": market.last_price,
                    "yes_bid": market.yes_bid,
                    "yes_ask": market.yes_ask,
                    "volume": market.volume,
                    "_close_time": close_time,
                    "_hours_to_settlement": hours_to_settlement,
                }
                settling_markets.append(market_dict)

        return settling_markets

    async def _analyze_market(self, market: dict) -> MomentumSignal | None:
        """Analyze a single market for momentum."""
        ticker = market.get("ticker", "")
        event_ticker = market.get("event_ticker", "")
        hours_to_settlement = market.get("_hours_to_settlement", 0)

        # Get current price
        current_price = market.get("last_price")
        if current_price is None:
            yes_bid = market.get("yes_bid")
            yes_ask = market.get("yes_ask")
            if yes_bid is not None and yes_ask is not None:
                current_price = (yes_bid + yes_ask) // 2
            else:
                return None

        # Get historical prices from snapshots
        price_6h_ago, price_24h_ago = await self._get_historical_prices(ticker)

        # Get 24h volume
        volume_24h = market.get("volume") or 0

        # Calculate momentum
        signal = self._momentum_calc.calculate_momentum(
            ticker=ticker,
            event_ticker=event_ticker,
            current_price=current_price,
            price_6h_ago=price_6h_ago,
            price_24h_ago=price_24h_ago,
            hours_to_settlement=hours_to_settlement,
            volume_24h=volume_24h,
        )

        return signal

    async def _get_historical_prices(
        self, ticker: str
    ) -> tuple[int | None, int | None]:
        """Get historical prices from market_snapshots table."""
        now = datetime.utcnow()

        # 6 hours ago
        time_6h = now - timedelta(hours=6)
        row_6h = await self._db.fetch_one(
            """
            SELECT yes_price FROM market_snapshots
            WHERE ticker = ? AND snapshot_time <= ?
            ORDER BY snapshot_time DESC LIMIT 1
            """,
            (ticker, time_6h.isoformat()),
        )
        price_6h = row_6h["yes_price"] if row_6h else None

        # 24 hours ago
        time_24h = now - timedelta(hours=24)
        row_24h = await self._db.fetch_one(
            """
            SELECT yes_price FROM market_snapshots
            WHERE ticker = ? AND snapshot_time <= ?
            ORDER BY snapshot_time DESC LIMIT 1
            """,
            (ticker, time_24h.isoformat()),
        )
        price_24h = row_24h["yes_price"] if row_24h else None

        return price_6h, price_24h

    async def _store_signals(self, signals: list[MomentumSignal]) -> None:
        """Store momentum signals in the database."""
        for signal in signals:
            expires_at = datetime.utcnow() + timedelta(hours=signal.hours_to_settlement)

            await self._db.execute(
                """
                INSERT OR REPLACE INTO momentum_signals
                (ticker, event_ticker, hours_to_settlement, current_price,
                 price_24h_ago, price_6h_ago, momentum, direction, confidence,
                 volume_24h, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.ticker,
                    signal.event_ticker,
                    signal.hours_to_settlement,
                    signal.current_price,
                    signal.price_24h_ago,
                    signal.price_6h_ago,
                    signal.momentum,
                    signal.direction,
                    signal.confidence,
                    signal.volume_24h,
                    expires_at.isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )

    async def get_trading_opportunities(
        self,
        available_capital: float,
        max_opportunities: int = 5,
    ) -> list[ArbitrageOpportunity]:
        """
        Get settlement momentum opportunities for the bot.

        Args:
            available_capital: Available capital for trading
            max_opportunities: Maximum opportunities to return

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []

        # Filter and rank signals
        tradeable = []
        for signal in self._signals.values():
            if not self._momentum_calc.should_trade(signal, self._min_momentum):
                continue

            # Additional price filters
            if signal.direction == "yes" and signal.current_price < self._min_price_yes:
                continue
            if signal.direction == "no" and signal.current_price > self._max_price_no:
                continue

            tradeable.append(signal)

        # Sort by confidence * momentum
        tradeable.sort(
            key=lambda s: s.confidence * s.momentum,
            reverse=True,
        )

        for signal in tradeable[:max_opportunities]:
            opp = self._create_opportunity(signal, available_capital)
            if opp:
                opportunities.append(opp.to_arbitrage_opportunity())

        return opportunities

    def _create_opportunity(
        self,
        signal: MomentumSignal,
        available_capital: float,
    ) -> SettlementOpportunity | None:
        """Create a trading opportunity from a momentum signal."""
        # Calculate expected edge
        # If momentum is strong and price is extreme, expected settlement is clear
        if signal.direction == "yes":
            # Betting on YES settlement at 100
            expected_settlement = 100
            entry_price = signal.current_price
            win_prob = signal.confidence
        else:
            # Betting on NO settlement (YES settles at 0)
            expected_settlement = 0
            entry_price = 100 - signal.current_price
            win_prob = signal.confidence

        # Edge = expected value - entry price
        expected_value = win_prob * 100 + (1 - win_prob) * 0
        if signal.direction == "yes":
            edge = (expected_value - entry_price) / 100
        else:
            edge = (entry_price - expected_value) / 100
            edge = win_prob - (signal.current_price / 100)

        if edge <= 0:
            return None

        # Position sizing using Kelly
        if win_prob > 0:
            payout_ratio = (100 - entry_price) / entry_price if entry_price > 0 else 0
            if payout_ratio > 0:
                kelly = (payout_ratio * win_prob - (1 - win_prob)) / payout_ratio
                kelly = max(0, min(kelly, self.KELLY_FRACTION))
            else:
                kelly = 0
        else:
            kelly = 0

        if kelly <= 0:
            return None

        # Cap at max position percentage
        position_value = min(
            available_capital * kelly,
            available_capital * self.MAX_POSITION_PCT,
        )

        cost_per_contract = entry_price / 100.0
        quantity = int(position_value / cost_per_contract) if cost_per_contract > 0 else 0

        if quantity < 1:
            return None

        # Expected profit
        expected_profit = (win_prob * quantity) - (quantity * cost_per_contract)
        fees = 0.07 * (entry_price / 100) * (1 - entry_price / 100) * quantity
        expected_profit -= fees

        if expected_profit <= 0:
            return None

        logger.info(
            f"Settlement opportunity: {signal.ticker} {signal.direction.upper()} "
            f"@ {signal.current_price}c, momentum={signal.momentum:.2f}, "
            f"conf={signal.confidence:.0%}, qty={quantity}"
        )

        return SettlementOpportunity(
            signal=signal,
            edge=edge,
            expected_profit=expected_profit,
            quantity=quantity,
        )

    def get_status(self) -> dict[str, Any]:
        """Get agent status with settlement-specific metrics."""
        status = super().get_status()
        status["active_signals"] = len(self._signals)
        status["yes_signals"] = sum(
            1 for s in self._signals.values() if s.direction == "yes"
        )
        status["no_signals"] = sum(
            1 for s in self._signals.values() if s.direction == "no"
        )
        return status
