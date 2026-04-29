"""Tests for the voting ensemble module."""

import pytest
from datetime import datetime, timezone, timedelta

from src.voting.ensemble import VotingEnsemble, TradeIntent
from src.core.types import Signal


class TestVotingEnsemble:
    """Tests for VotingEnsemble class."""

    def test_initialization_with_defaults(self):
        """Test ensemble initializes with default settings."""
        voting = VotingEnsemble()

        assert voting._min_edge == 0.03
        assert voting._min_confidence == 0.50
        assert voting._min_agreement == 1
        assert voting._fee_adjustment == 0.01

    def test_initialization_with_custom_settings(self):
        """Test ensemble initializes with custom settings."""
        voting = VotingEnsemble(
            strategy_weights={"weather": 2.0, "arbitrage": 1.5},
            min_edge=0.05,
            min_confidence=0.60,
            min_agreement=2,
            fee_adjustment=0.02,
        )

        assert voting._min_edge == 0.05
        assert voting._min_confidence == 0.60
        assert voting._min_agreement == 2
        assert voting.get_strategy_weight("weather") == 2.0
        assert voting.get_strategy_weight("arbitrage") == 1.5

    def test_get_strategy_weight_default(self):
        """Test default weight for unknown strategy."""
        voting = VotingEnsemble()
        assert voting.get_strategy_weight("unknown") == 1.0

    def test_set_strategy_weight(self):
        """Test setting strategy weight."""
        voting = VotingEnsemble()
        voting.set_strategy_weight("test", 2.5)
        assert voting.get_strategy_weight("test") == 2.5

    def test_process_single_signal_passing(self):
        """Test single signal that passes thresholds."""
        voting = VotingEnsemble(
            min_edge=0.03,
            min_confidence=0.50,
            fee_adjustment=0.01,
        )

        signal = Signal.create(
            strategy_name="test",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.70,
            market_probability=0.50,
            confidence=0.80,
            max_position=10,
        )

        intent = voting.process_single_signal(signal)

        assert intent is not None
        assert intent.market_ticker == "TEST-123"
        assert intent.direction == "yes"
        assert intent.final_confidence == 0.80
        assert len(intent.contributing_signals) == 1

    def test_process_single_signal_edge_too_low(self):
        """Test signal rejected for low edge."""
        voting = VotingEnsemble(min_edge=0.10)

        signal = Signal.create(
            strategy_name="test",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.55,
            market_probability=0.50,
            confidence=0.80,
            max_position=10,
        )

        intent = voting.process_single_signal(signal)
        assert intent is None

    def test_process_single_signal_confidence_too_low(self):
        """Test signal rejected for low confidence."""
        voting = VotingEnsemble(min_confidence=0.70)

        signal = Signal.create(
            strategy_name="test",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.80,
            market_probability=0.50,
            confidence=0.50,
            max_position=10,
        )

        intent = voting.process_single_signal(signal)
        assert intent is None

    def test_process_single_signal_expired(self):
        """Test expired signal is rejected."""
        voting = VotingEnsemble()

        signal = Signal.create(
            strategy_name="test",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.70,
            market_probability=0.50,
            confidence=0.80,
            max_position=10,
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        intent = voting.process_single_signal(signal)
        assert intent is None

    def test_aggregate_signals_empty_list(self):
        """Test aggregation with empty signal list."""
        voting = VotingEnsemble()
        intents = voting.aggregate_signals([])
        assert intents == []

    def test_aggregate_signals_single_market(self):
        """Test aggregation with signals for one market."""
        voting = VotingEnsemble(min_agreement=1)

        signal = Signal.create(
            strategy_name="test",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.70,
            market_probability=0.50,
            confidence=0.80,
            max_position=10,
        )

        intents = voting.aggregate_signals([signal])
        assert len(intents) == 1
        assert intents[0].market_ticker == "TEST-123"

    def test_aggregate_signals_multiple_markets(self):
        """Test aggregation with signals for multiple markets."""
        voting = VotingEnsemble(min_agreement=1)

        signals = [
            Signal.create(
                strategy_name="test",
                market_ticker="TEST-1",
                direction="yes",
                target_probability=0.70,
                market_probability=0.50,
                confidence=0.80,
                max_position=10,
            ),
            Signal.create(
                strategy_name="test",
                market_ticker="TEST-2",
                direction="no",
                target_probability=0.70,
                market_probability=0.50,
                confidence=0.75,
                max_position=10,
            ),
        ]

        intents = voting.aggregate_signals(signals)
        assert len(intents) == 2
        tickers = {i.market_ticker for i in intents}
        assert tickers == {"TEST-1", "TEST-2"}

    def test_aggregate_signals_filters_expired(self):
        """Test that expired signals are filtered out."""
        voting = VotingEnsemble(min_agreement=1)

        signals = [
            Signal.create(
                strategy_name="test",
                market_ticker="TEST-1",
                direction="yes",
                target_probability=0.70,
                market_probability=0.50,
                confidence=0.80,
                max_position=10,
            ),
            Signal.create(
                strategy_name="test",
                market_ticker="TEST-2",
                direction="yes",
                target_probability=0.70,
                market_probability=0.50,
                confidence=0.80,
                max_position=10,
                expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
            ),
        ]

        intents = voting.aggregate_signals(signals)
        assert len(intents) == 1
        assert intents[0].market_ticker == "TEST-1"

    def test_get_status(self):
        """Test status reporting."""
        voting = VotingEnsemble(
            strategy_weights={"test": 2.0},
            min_edge=0.05,
        )

        status = voting.get_status()

        assert status["strategy_weights"] == {"test": 2.0}
        assert status["config"]["min_edge"] == 0.05
        assert "signals_processed" in status["stats"]
        assert "intents_generated" in status["stats"]

    def test_from_config(self):
        """Test creation from config dictionary."""
        config = {
            "strategies": {
                "weather": {"weight": 2.0, "enabled": True},
                "arbitrage": {"weight": 1.5, "enabled": True},
                "disabled": {"weight": 1.0, "enabled": False},
            },
            "min_edge": 0.04,
            "min_confidence": 0.55,
            "min_agreement": 2,
        }

        voting = VotingEnsemble.from_config(config)

        assert voting._min_edge == 0.04
        assert voting._min_confidence == 0.55
        assert voting._min_agreement == 2
        assert voting.get_strategy_weight("weather") == 2.0
        assert voting.get_strategy_weight("arbitrage") == 1.5
        assert voting.get_strategy_weight("disabled") == 1.0  # Default for disabled


class TestTradeIntent:
    """Tests for TradeIntent dataclass."""

    def test_signal_count(self):
        """Test signal count property."""
        signal = Signal.create(
            strategy_name="test",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.70,
            market_probability=0.50,
            confidence=0.80,
            max_position=10,
        )

        intent = TradeIntent(
            intent_id="test-intent",
            market_ticker="TEST-123",
            direction="yes",
            final_edge=0.15,
            final_confidence=0.80,
            contributing_signals=[signal],
            strategy_votes={"test": 0.15},
        )

        assert intent.signal_count == 1

    def test_primary_signal(self):
        """Test primary signal selection."""
        signal1 = Signal.create(
            strategy_name="test1",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.70,
            market_probability=0.50,
            confidence=0.60,
            max_position=10,
        )

        signal2 = Signal.create(
            strategy_name="test2",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.70,
            market_probability=0.50,
            confidence=0.90,
            max_position=10,
        )

        intent = TradeIntent(
            intent_id="test-intent",
            market_ticker="TEST-123",
            direction="yes",
            final_edge=0.15,
            final_confidence=0.75,
            contributing_signals=[signal1, signal2],
            strategy_votes={"test1": 0.12, "test2": 0.18},
        )

        assert intent.primary_signal.strategy_name == "test2"

    def test_to_dict(self):
        """Test dictionary conversion."""
        signal = Signal.create(
            strategy_name="test",
            market_ticker="TEST-123",
            direction="yes",
            target_probability=0.70,
            market_probability=0.50,
            confidence=0.80,
            max_position=10,
        )

        intent = TradeIntent(
            intent_id="test-intent",
            market_ticker="TEST-123",
            direction="yes",
            final_edge=0.15,
            final_confidence=0.80,
            contributing_signals=[signal],
            strategy_votes={"test": 0.15},
            category="weather",
        )

        d = intent.to_dict()

        assert d["intent_id"] == "test-intent"
        assert d["market_ticker"] == "TEST-123"
        assert d["direction"] == "yes"
        assert d["final_edge"] == 0.15
        assert d["category"] == "weather"
        assert d["signal_count"] == 1


class TestCategoryExtraction:
    """Tests for market category extraction."""

    def test_weather_category(self):
        """Test weather market detection."""
        voting = VotingEnsemble()

        assert voting._extract_category("KXHIGHNY-25JAN01-B79") == "weather"
        assert voting._extract_category("KXLOWCHI-25JAN01-T30") == "weather"
        assert voting._extract_category("KXRAINMIA-25JAN01-B0.1") == "weather"

    def test_economic_category(self):
        """Test economic market detection."""
        voting = VotingEnsemble()

        assert voting._extract_category("FED-RATE-JAN25") == "economic"
        assert voting._extract_category("FOMC-25JAN") == "economic"
        assert voting._extract_category("CPI-DEC24") == "economic"

    def test_sports_category(self):
        """Test sports market detection."""
        voting = VotingEnsemble()

        assert voting._extract_category("NFL-SUPERBOWL") == "sports"
        assert voting._extract_category("NBA-FINALS") == "sports"

    def test_politics_category(self):
        """Test politics market detection."""
        voting = VotingEnsemble()

        assert voting._extract_category("PRES-2024") == "politics"
        assert voting._extract_category("ELECTION-2024") == "politics"

    def test_unknown_category(self):
        """Test unknown market category."""
        voting = VotingEnsemble()

        assert voting._extract_category("RANDOM-MARKET-123") is None
