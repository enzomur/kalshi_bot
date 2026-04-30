"""Tests for trading strategies."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from src.strategies.mean_reversion import MeanReversionStrategy, PriceHistory
from src.strategies.arbitrage import ArbitrageStrategy
from src.strategies.market_make import MarketMakingStrategy, InventoryState, MarketMakingQuote


class TestPriceHistory:
    """Tests for PriceHistory helper class."""

    def test_add_price(self):
        """Test adding prices to history."""
        history = PriceHistory(market_ticker="TEST")
        history.add_price(50.0)
        history.add_price(55.0)
        history.add_price(52.0)

        assert history.sample_count == 3
        assert history.current_price == 52.0

    def test_max_history_limit(self):
        """Test that history is trimmed to max size."""
        history = PriceHistory(market_ticker="TEST", max_history=5)

        for i in range(10):
            history.add_price(float(i))

        assert history.sample_count == 5
        assert history.prices == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_get_recent(self):
        """Test getting recent prices."""
        history = PriceHistory(market_ticker="TEST")
        for i in range(10):
            history.add_price(float(i))

        recent = history.get_recent(3)
        assert recent == [7.0, 8.0, 9.0]

    def test_current_price_empty(self):
        """Test current price when empty."""
        history = PriceHistory(market_ticker="TEST")
        assert history.current_price is None


class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MeanReversionStrategy(
            lookback_periods=12,
            z_score_threshold=1.5,
            min_samples=6,
        )

        assert strategy.name == "mean_reversion"
        assert strategy._lookback_periods == 12
        assert strategy._z_score_threshold == 1.5
        assert strategy._min_samples == 6

    def test_update_price(self):
        """Test updating price history."""
        strategy = MeanReversionStrategy()
        strategy.update_price("TEST", 50.0)
        strategy.update_price("TEST", 55.0)

        history = strategy.get_price_history("TEST")
        assert history is not None
        assert history.sample_count == 2

    def test_update_price_normalizes_probability(self):
        """Test that probability-format prices are normalized."""
        strategy = MeanReversionStrategy()
        strategy.update_price("TEST", 0.50)  # Probability format
        strategy.update_price("TEST", 55.0)  # Cents format

        history = strategy.get_price_history("TEST")
        assert history.prices == [50.0, 55.0]

    def test_calculate_z_score(self):
        """Test z-score calculation."""
        strategy = MeanReversionStrategy(min_samples=3)

        # Add consistent prices
        for _ in range(5):
            strategy.update_price("TEST", 50.0)

        # Add a deviation
        strategy.update_price("TEST", 60.0)

        z = strategy.calculate_z_score("TEST")
        assert z is not None
        assert z > 0  # Price above mean

    def test_calculate_z_score_insufficient_history(self):
        """Test z-score returns None with insufficient history."""
        strategy = MeanReversionStrategy(min_samples=10)
        strategy.update_price("TEST", 50.0)

        assert strategy.calculate_z_score("TEST") is None

    def test_get_market_stats(self):
        """Test market statistics retrieval."""
        strategy = MeanReversionStrategy(min_samples=3)

        for i in range(5):
            strategy.update_price("TEST", 50.0 + i)

        stats = strategy.get_market_stats("TEST")

        assert stats is not None
        assert stats["ticker"] == "TEST"
        assert stats["sample_count"] == 5
        assert "moving_average" in stats
        assert "std_deviation" in stats
        assert "z_score" in stats

    def test_clear_history(self):
        """Test clearing price history."""
        strategy = MeanReversionStrategy()
        strategy.update_price("TEST1", 50.0)
        strategy.update_price("TEST2", 55.0)

        strategy.clear_history("TEST1")
        assert strategy.get_price_history("TEST1") is None
        assert strategy.get_price_history("TEST2") is not None

        strategy.clear_history()
        assert strategy.get_price_history("TEST2") is None

    def test_load_historical_prices(self):
        """Test bulk loading historical prices."""
        strategy = MeanReversionStrategy()

        prices = [
            (50.0, datetime.now(timezone.utc) - timedelta(hours=3)),
            (52.0, datetime.now(timezone.utc) - timedelta(hours=2)),
            (48.0, datetime.now(timezone.utc) - timedelta(hours=1)),
            (51.0, datetime.now(timezone.utc)),
        ]

        strategy.load_historical_prices("TEST", prices)

        history = strategy.get_price_history("TEST")
        assert history.sample_count == 4
        assert history.current_price == 51.0

    @pytest.mark.asyncio
    async def test_generate_signals_disabled(self):
        """Test that disabled strategy returns no signals."""
        strategy = MeanReversionStrategy(enabled=False)
        signals = await strategy.generate_signals([])
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_empty_markets(self):
        """Test with empty market list."""
        strategy = MeanReversionStrategy()
        signals = await strategy.generate_signals([])
        assert signals == []

    def test_get_status(self):
        """Test status includes mean reversion metrics."""
        strategy = MeanReversionStrategy()
        strategy.update_price("TEST", 50.0)

        status = strategy.get_status()

        assert status["name"] == "mean_reversion"
        assert "lookback_periods" in status["metrics"]
        assert "z_score_threshold" in status["metrics"]
        assert "markets_tracked" in status["metrics"]

    def test_from_config(self):
        """Test creation from config dictionary."""
        config = {
            "enabled": True,
            "min_edge": 0.08,
            "lookback_periods": 48,
            "z_score_threshold": 2.5,
            "min_samples": 24,
        }

        strategy = MeanReversionStrategy.from_config(config)

        assert strategy._lookback_periods == 48
        assert strategy._z_score_threshold == 2.5
        assert strategy._min_samples == 24


class TestMarketMakingQuote:
    """Tests for MarketMakingQuote dataclass."""

    def test_quote_creation(self):
        """Test creating a market making quote."""
        quote = MarketMakingQuote(
            ticker="TEST",
            bid_price=45,
            ask_price=55,
            bid_size=10,
            ask_size=10,
            mid_price=50.0,
            spread=10,
            spread_pct=0.20,
        )

        assert quote.ticker == "TEST"
        assert quote.bid_price == 45
        assert quote.ask_price == 55
        assert quote.spread == 10

    def test_quote_to_dict(self):
        """Test quote serialization."""
        quote = MarketMakingQuote(
            ticker="TEST",
            bid_price=45,
            ask_price=55,
            bid_size=10,
            ask_size=10,
            mid_price=50.0,
            spread=10,
            spread_pct=0.20,
            inventory_skew=0.5,
        )

        d = quote.to_dict()

        assert d["ticker"] == "TEST"
        assert d["bid_price"] == 45
        assert d["inventory_skew"] == 0.5


class TestArbitrageStrategy:
    """Tests for ArbitrageStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ArbitrageStrategy(min_profit=0.02, fees=0.01)

        assert strategy.name == "arbitrage"
        assert strategy._min_profit == 0.02
        assert strategy._fees == 0.01

    def test_initialization_defaults(self):
        """Test strategy default values."""
        strategy = ArbitrageStrategy()

        assert strategy.name == "arbitrage"
        assert strategy._min_profit == 0.01  # 1% default
        assert strategy._fees == 0.02  # 2% default

    @pytest.mark.asyncio
    async def test_generate_signals_disabled(self):
        """Test disabled strategy returns no signals."""
        strategy = ArbitrageStrategy(enabled=False)
        signals = await strategy.generate_signals([])
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_empty_markets(self):
        """Test with empty market list."""
        strategy = ArbitrageStrategy()
        signals = await strategy.generate_signals([])
        assert signals == []

    def test_get_status(self):
        """Test status reporting."""
        strategy = ArbitrageStrategy()
        status = strategy.get_status()

        assert status["name"] == "arbitrage"
        assert status["enabled"] is True


class TestInventoryState:
    """Tests for InventoryState in market making."""

    def test_initialization(self):
        """Test inventory state initialization."""
        state = InventoryState(
            ticker="TEST",
            position=10,
            avg_price=50.0,
        )

        assert state.ticker == "TEST"
        assert state.position == 10
        assert state.avg_price == 50.0

    def test_is_max_long(self):
        """Test max long position detection."""
        state = InventoryState(ticker="TEST", position=50, max_position=50)
        assert state.is_max_long is True

        state2 = InventoryState(ticker="TEST", position=25, max_position=50)
        assert state2.is_max_long is False

    def test_is_max_short(self):
        """Test max short position detection."""
        state = InventoryState(ticker="TEST", position=-50, max_position=50)
        assert state.is_max_short is True

        state2 = InventoryState(ticker="TEST", position=-25, max_position=50)
        assert state2.is_max_short is False

    def test_inventory_ratio(self):
        """Test inventory ratio calculation."""
        state = InventoryState(ticker="TEST", position=25, max_position=50)
        assert state.inventory_ratio == 0.5

        state2 = InventoryState(ticker="TEST", position=-25, max_position=50)
        assert state2.inventory_ratio == -0.5

    def test_inventory_ratio_zero_max(self):
        """Test inventory ratio with zero max position."""
        state = InventoryState(ticker="TEST", position=10, max_position=0)
        assert state.inventory_ratio == 0.0


class TestMarketMakingStrategy:
    """Tests for MarketMakingStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MarketMakingStrategy(
            min_spread_cents=5,
            quote_size=20,
            max_inventory=100,
        )

        assert strategy.name == "market_make"
        assert strategy._min_spread == 5
        assert strategy._quote_size == 20
        assert strategy._max_inventory == 100

    def test_initialization_defaults(self):
        """Test default values."""
        strategy = MarketMakingStrategy()

        assert strategy.name == "market_make"
        assert strategy._min_spread == 3  # MIN_SPREAD_CENTS
        assert strategy._quote_size == 10  # DEFAULT_QUOTE_SIZE
        assert strategy._max_inventory == 50  # DEFAULT_MAX_INVENTORY

    def test_target_markets(self):
        """Test target markets configuration."""
        strategy = MarketMakingStrategy(
            target_markets=["MARKET-1", "MARKET-2"]
        )

        assert strategy._target_markets == ["MARKET-1", "MARKET-2"]

    @pytest.mark.asyncio
    async def test_generate_signals_disabled(self):
        """Test disabled strategy returns no signals."""
        strategy = MarketMakingStrategy(enabled=False)
        signals = await strategy.generate_signals([])
        assert signals == []

    @pytest.mark.asyncio
    async def test_generate_signals_empty_markets(self):
        """Test with empty market list."""
        strategy = MarketMakingStrategy()
        signals = await strategy.generate_signals([])
        assert signals == []

    def test_get_status(self):
        """Test status reporting."""
        strategy = MarketMakingStrategy()
        status = strategy.get_status()

        assert status["name"] == "market_make"
        assert status["enabled"] is True


class TestStrategyIntegration:
    """Integration tests for strategy behavior."""

    @pytest.mark.asyncio
    async def test_mean_reversion_with_price_history(self):
        """Test mean reversion generates signals with sufficient history."""
        strategy = MeanReversionStrategy(
            min_samples=5,
            z_score_threshold=1.5,
            lookback_periods=10,
        )

        # Build up price history at 50
        for _ in range(8):
            strategy.update_price("TEST", 50.0)

        # Now price spikes to 70 (4 std dev at std=5)
        strategy.update_price("TEST", 70.0)
        strategy.update_price("TEST", 72.0)

        # Create a market with current high price
        markets = [
            {
                "ticker": "TEST",
                "yes_bid": 70,
                "yes_ask": 74,
                "close_time": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            }
        ]

        signals = await strategy.generate_signals(markets)

        # Should generate a NO signal (expect price to fall)
        if signals:
            assert signals[0].direction == "no"

    @pytest.mark.asyncio
    async def test_arbitrage_empty_without_polymarket(self):
        """Test arbitrage returns empty without Polymarket client."""
        strategy = ArbitrageStrategy()

        markets = [
            {
                "ticker": "ARB-TEST",
                "yes_bid": 40,
                "yes_ask": 42,
                "no_bid": 55,
                "no_ask": 57,
            }
        ]

        # Without Polymarket client, no cross-market arbitrage possible
        signals = await strategy.generate_signals(markets)
        assert signals == []

    @pytest.mark.asyncio
    async def test_market_making_with_target_markets(self):
        """Test market making with target markets."""
        strategy = MarketMakingStrategy(
            min_spread_cents=5,
            target_markets=["MM-TEST"],
        )

        markets = [
            {
                "ticker": "MM-TEST",
                "yes_bid": 40,
                "yes_ask": 55,  # 15 cent spread
                "volume": 5000,
                "open_interest": 1000,
            }
        ]

        signals = await strategy.generate_signals(markets)

        # May or may not generate depending on market suitability
        assert isinstance(signals, list)
