"""Pytest configuration and fixtures."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("KALSHI_API_KEY_ID", "test-api-key")
os.environ.setdefault("KALSHI_PRIVATE_KEY_PATH", "")
os.environ.setdefault("KALSHI_ENVIRONMENT", "demo")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path(temp_dir: Path) -> Path:
    """Get temporary database path."""
    return temp_dir / "test.db"


@pytest.fixture
def mock_settings(temp_db_path: Path) -> MagicMock:
    """Create mock settings for testing."""
    from kalshi_bot.config.settings import (
        APISettings,
        ArbitrageSettings,
        DashboardSettings,
        Environment,
        LoggingSettings,
        PortfolioSettings,
        RiskSettings,
        TradingSettings,
    )

    settings = MagicMock()
    settings.api_key_id = "test-api-key"
    settings.private_key_path = ""
    settings.environment = Environment.DEMO
    settings.database_path = str(temp_db_path)
    settings.api_base_url = "https://demo-api.kalshi.co/trade-api/v2"
    settings.ws_base_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"

    settings.trading = TradingSettings()
    settings.risk = RiskSettings()
    settings.portfolio = PortfolioSettings()
    settings.arbitrage = ArbitrageSettings()
    settings.api = APISettings()
    settings.dashboard = DashboardSettings()
    settings.logging = LoggingSettings()

    return settings


@pytest.fixture
def sample_order_book() -> dict[str, Any]:
    """Sample order book data."""
    return {
        "market_ticker": "TEST-MARKET",
        "yes_bids": [{"price": 45, "quantity": 100}, {"price": 44, "quantity": 50}],
        "yes_asks": [{"price": 47, "quantity": 80}, {"price": 48, "quantity": 60}],
        "no_bids": [{"price": 53, "quantity": 90}, {"price": 52, "quantity": 40}],
        "no_asks": [{"price": 55, "quantity": 70}, {"price": 56, "quantity": 50}],
    }


@pytest.fixture
def sample_market_data() -> dict[str, Any]:
    """Sample market data."""
    return {
        "ticker": "TEST-MARKET",
        "event_ticker": "TEST-EVENT",
        "title": "Test Market",
        "subtitle": "Will something happen?",
        "status": "open",
        "yes_bid": 45,
        "yes_ask": 47,
        "no_bid": 53,
        "no_ask": 55,
        "last_price": 46,
        "volume": 1000,
        "open_interest": 500,
    }


@pytest.fixture
def sample_position() -> dict[str, Any]:
    """Sample position data."""
    return {
        "market_ticker": "TEST-MARKET",
        "side": "yes",
        "quantity": 10,
        "average_price": 45.0,
        "market_exposure": 4.50,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.20,
    }


@pytest.fixture
def sample_trade() -> dict[str, Any]:
    """Sample trade data."""
    return {
        "trade_id": "trade-123",
        "order_id": "order-456",
        "market_ticker": "TEST-MARKET",
        "side": "yes",
        "price": 45,
        "quantity": 10,
        "fee": 0.14,
        "executed_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_arbitrage_opportunity() -> dict[str, Any]:
    """Sample arbitrage opportunity."""
    return {
        "opportunity_id": "arb-001",
        "arbitrage_type": "single_market",
        "markets": ["TEST-MARKET"],
        "expected_profit": 3.0,
        "expected_profit_pct": 0.03,
        "confidence": 0.95,
        "legs": [
            {"market": "TEST-MARKET", "side": "yes", "price": 45, "quantity": 10},
            {"market": "TEST-MARKET", "side": "no", "price": 52, "quantity": 10},
        ],
        "max_quantity": 50,
        "total_cost": 9.70,
        "fees": 0.28,
        "net_profit": 0.02,
    }


@pytest.fixture
def mock_api_client() -> AsyncMock:
    """Create mock API client."""
    client = AsyncMock()
    client.get_balance = AsyncMock(return_value={"balance": 10000.0})
    client.get_positions = AsyncMock(return_value=[])
    client.get_markets = AsyncMock(return_value=([], None))
    client.get_all_markets = AsyncMock(return_value=[])
    client.get_orderbook = AsyncMock(return_value={})
    client.create_order = AsyncMock(
        return_value={"order_id": "test-order-123", "status": "open"}
    )
    client.cancel_order = AsyncMock(return_value={"status": "cancelled"})
    return client


@pytest.fixture
def mock_websocket() -> AsyncMock:
    """Create mock WebSocket manager."""
    ws = AsyncMock()
    ws.connect = AsyncMock()
    ws.disconnect = AsyncMock()
    ws.subscribe = AsyncMock()
    ws.unsubscribe = AsyncMock()
    return ws


@pytest.fixture
async def database(temp_db_path: Path) -> AsyncGenerator[Any, None]:
    """Create test database."""
    from kalshi_bot.persistence.database import Database

    db = Database(str(temp_db_path))
    await db.initialize()
    yield db
    await db.close()


# ============================================================================
# New src/ module fixtures (Phase 2 architecture)
# ============================================================================

@pytest.fixture
def paper_broker():
    """Create a fresh paper broker for testing."""
    from src.execution.paper_broker import PaperBroker
    return PaperBroker(db=None, initial_balance=1000.0, slippage_bps=10)


@pytest.fixture
def mock_mode_manager() -> MagicMock:
    """Create a mock mode manager in PAPER mode."""
    from src.core.types import TradingMode

    manager = MagicMock()
    manager.current_mode = TradingMode.PAPER
    manager.is_paper = True
    manager.is_live = False
    manager.config = MagicMock()
    manager.config.max_position_dollars = 10000.0
    manager.config.max_daily_loss_dollars = 10000.0
    manager.config.mode = TradingMode.PAPER
    return manager


@pytest.fixture
def sample_signal():
    """Create a sample valid signal for testing."""
    from src.core.types import Signal

    return Signal.create(
        strategy_name="test_strategy",
        market_ticker="TEST-MARKET-123",
        direction="yes",
        target_probability=0.65,
        market_probability=0.50,
        confidence=0.75,
        max_position=50,
        metadata={"market_price_cents": 50},
    )
