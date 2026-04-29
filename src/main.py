#!/usr/bin/env python3
"""Main entry point for Kalshi Trading Bot v2.

This module initializes all components and runs the main trading loop.
The bot always starts in PAPER mode unless a valid signed configuration
exists for LIVE modes.

Usage:
    python -m src.main              # Run with default config
    python -m src.main --paper      # Force paper mode
    python -m src.main --config-dir /path/to/config
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.core.mode import verify_mode_on_startup, get_mode_manager, ModeManager
from src.core.types import TradingMode, Signal
from src.data.kalshi import KalshiClient, WebSocketManager
from src.data.odds_feed import OddsFeed
from src.execution.paper_broker import PaperBroker
from src.ledger.database import Database
from src.metrics import BrierCalculator, PerformanceTracker, CalibrationCurve, TradeResult
from src.observability.logging import setup_logging, get_logger, new_correlation_id
from src.risk.engine import RiskEngine
from src.strategies.base import Strategy
from src.strategies.weather import WeatherStrategy
from src.strategies.calibration import CalibrationStrategy
from src.voting import VotingEnsemble


logger = get_logger(__name__)


class TradingBot:
    """Main trading bot orchestrator."""

    # Trading loop configuration
    LOOP_INTERVAL_SECONDS = 60  # Run strategies every minute
    MARKET_FETCH_INTERVAL_SECONDS = 300  # Refresh market list every 5 minutes
    DAILY_RESET_HOUR = 0  # Reset daily metrics at midnight UTC

    def __init__(
        self,
        config_dir: str = "config",
        secrets_dir: str = "secrets",
        data_dir: str = "data",
    ) -> None:
        """
        Initialize trading bot.

        Args:
            config_dir: Directory containing YAML config files
            secrets_dir: Directory containing secrets (.env, kalshi.pem)
            data_dir: Directory for database and other data
        """
        self.config_dir = Path(config_dir)
        self.secrets_dir = Path(secrets_dir)
        self.data_dir = Path(data_dir)

        self._shutdown_event = asyncio.Event()
        self._config: dict[str, Any] = {}

        # Components (initialized in start())
        self._db: Database | None = None
        self._client: KalshiClient | None = None
        self._ws: WebSocketManager | None = None
        self._broker: PaperBroker | None = None
        self._risk_engine: RiskEngine | None = None
        self._mode_manager: ModeManager | None = None

        # Strategy and voting components
        self._strategies: list[Strategy] = []
        self._voting: VotingEnsemble | None = None

        # Metrics tracking
        self._brier_calc: BrierCalculator | None = None
        self._performance: PerformanceTracker | None = None
        self._calibration: CalibrationCurve | None = None

        # Runtime state
        self._markets_cache: list[dict[str, Any]] = []
        self._last_market_fetch: datetime | None = None
        self._last_daily_reset: datetime | None = None
        self._loop_count = 0

    def _load_config(self) -> dict[str, Any]:
        """Load all configuration files."""
        config: dict[str, Any] = {}

        config_files = ["mode.yaml", "strategies.yaml", "risk.yaml", "markets.yaml"]
        for filename in config_files:
            path = self.config_dir / filename
            if path.exists():
                with open(path) as f:
                    config[filename.replace(".yaml", "")] = yaml.safe_load(f) or {}
            else:
                logger.warning(f"Config file not found: {path}")
                config[filename.replace(".yaml", "")] = {}

        return config

    def _load_secrets(self) -> dict[str, str]:
        """Load secrets from .env file."""
        secrets: dict[str, str] = {}
        env_path = self.secrets_dir / ".env"

        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        secrets[key.strip()] = value.strip().strip("\"'")
                        os.environ[key.strip()] = value.strip().strip("\"'")

        return secrets

    async def start(self) -> None:
        """Start the trading bot."""
        cid = new_correlation_id()
        logger.info("Starting Kalshi Trading Bot v2", correlation_id=cid)

        # Load configuration
        self._config = self._load_config()
        secrets = self._load_secrets()

        # Verify mode (will default to PAPER if invalid)
        mode_config = verify_mode_on_startup()
        self._mode_manager = get_mode_manager()

        # Set up logging based on config
        log_config = self._config.get("logging", {})
        setup_logging(
            log_level=log_config.get("level", "INFO"),
            json_format=log_config.get("json_format", False),
            log_file=log_config.get("file", "logs/bot.log"),
        )

        # Initialize database
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db = Database(str(self.data_dir / "ledger.db"))
        await self._db.initialize()
        logger.info("Database initialized")

        # Initialize Kalshi client
        api_key_id = secrets.get("KALSHI_API_KEY_ID", os.getenv("KALSHI_API_KEY_ID", ""))
        private_key_path = str(self.secrets_dir / "kalshi.pem")

        if not api_key_id:
            logger.warning("KALSHI_API_KEY_ID not set - API calls will fail")

        if not Path(private_key_path).exists():
            logger.warning(f"Private key not found at {private_key_path}")
            private_key_path = ""

        # Determine API URL based on environment
        environment = secrets.get("KALSHI_ENVIRONMENT", os.getenv("KALSHI_ENVIRONMENT", "demo"))
        if environment == "production":
            api_base_url = "https://api.elections.kalshi.com/trade-api/v2"
            ws_base_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
        else:
            api_base_url = "https://demo-api.kalshi.co/trade-api/v2"
            ws_base_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"

        self._client = KalshiClient(
            api_base_url=api_base_url,
            api_key_id=api_key_id,
            private_key_path=private_key_path,
            requests_per_second=10.0,
            timeout=30,
        )

        # Connect to API if credentials are available
        if api_key_id and private_key_path:
            try:
                await self._client.connect()
                health = await self._client.health_check()
                if health:
                    logger.info(f"Connected to Kalshi API ({environment})")
                else:
                    logger.warning("API health check failed")
            except Exception as e:
                logger.error(f"Failed to connect to Kalshi API: {e}")

        # Initialize broker (always paper for now)
        risk_config = self._config.get("risk", {})
        sizing_config = risk_config.get("sizing", {})

        self._broker = PaperBroker(
            db=self._db,
            initial_balance=1000.0,
            slippage_bps=10,
        )
        logger.info("Paper broker initialized")

        # Initialize Risk Engine
        self._risk_engine = RiskEngine(
            broker=self._broker,
            db=self._db,
            mode_manager=self._mode_manager,
            kelly_fraction=sizing_config.get("kelly_fraction", 0.25),
            max_position_contracts=sizing_config.get("max_position_contracts", 100),
            max_daily_loss=risk_config.get("circuit_breakers", {}).get("max_daily_loss", 100.0),
            max_drawdown=risk_config.get("circuit_breakers", {}).get("max_drawdown", 0.20),
            max_consecutive_losses=risk_config.get("circuit_breakers", {}).get("max_consecutive_losses", 5),
        )
        logger.info("Risk Engine initialized")

        # Initialize Strategies
        await self._initialize_strategies()
        logger.info(f"Initialized {len(self._strategies)} strategies")

        # Initialize Voting Ensemble
        strategies_config = self._config.get("strategies", {})
        self._voting = VotingEnsemble.from_config(strategies_config)
        logger.info("Voting ensemble initialized")

        # Initialize Metrics Tracking
        self._brier_calc = BrierCalculator(calibration_threshold=0.22)
        self._performance = PerformanceTracker(initial_balance=self._broker.balance)
        self._calibration = CalibrationCurve(n_buckets=10, min_bucket_size=5)
        logger.info("Metrics tracking initialized")

        # Log startup summary
        logger.info(
            "Bot started",
            mode=mode_config.mode.value,
            broker="paper",
            balance=self._broker.balance,
            api_connected=self._client is not None and api_key_id != "",
            strategies=[s.name for s in self._strategies],
        )

    async def _initialize_strategies(self) -> None:
        """Initialize trading strategies from configuration."""
        strategies_config = self._config.get("strategies", {})

        # Weather strategy
        weather_config = strategies_config.get("weather", {})
        if weather_config.get("enabled", True):
            weather_strategy = WeatherStrategy(
                db=self._db,
                enabled=True,
                min_edge=weather_config.get("min_edge", 0.10),
                min_confidence=weather_config.get("min_confidence", 0.60),
                enabled_locations=weather_config.get("enabled_locations"),
            )
            self._strategies.append(weather_strategy)
            logger.info("Initialized weather strategy")

        # Calibration strategy (uses external odds from sportsbooks)
        calibration_config = strategies_config.get("calibration", {})
        if calibration_config.get("enabled", True):
            # Initialize odds feed
            odds_api_key = os.getenv("ODDS_API_KEY", "")
            odds_feed = OddsFeed(api_key=odds_api_key) if odds_api_key else None

            if odds_feed:
                await odds_feed.connect()

            calibration_strategy = CalibrationStrategy(
                odds_feed=odds_feed,
                db=self._db,
                enabled=True,
                min_edge=calibration_config.get("min_edge", 0.05),
                min_confidence=calibration_config.get("min_confidence", 0.60),
                enabled_sports=calibration_config.get("enabled_sports", ["NFL", "NBA"]),
            )
            self._strategies.append(calibration_strategy)
            logger.info(
                f"Initialized calibration strategy "
                f"(odds_feed={'enabled' if odds_feed else 'disabled'})"
            )

        # Phase 3 strategies (not yet implemented):
        # - arbitrage strategy (cross-market)
        # - market_make strategy (spread capture)
        # - mean_reversion strategy (fade overreactions)

    async def _fetch_markets(self) -> list[dict[str, Any]]:
        """Fetch tradeable markets from Kalshi API."""
        if self._client is None:
            logger.warning("API client not initialized, using cached markets")
            return self._markets_cache

        # Check if we need to refresh
        now = datetime.now(timezone.utc)
        if self._last_market_fetch is not None:
            elapsed = (now - self._last_market_fetch).total_seconds()
            if elapsed < self.MARKET_FETCH_INTERVAL_SECONDS and self._markets_cache:
                return self._markets_cache

        try:
            markets_config = self._config.get("markets", {})
            min_volume = markets_config.get("liquidity_filters", {}).get("min_volume", 0)
            min_oi = markets_config.get("liquidity_filters", {}).get("min_open_interest", 0)

            markets = await self._client.get_all_markets(
                status="open",
                min_volume=min_volume,
                min_open_interest=min_oi,
                require_orderbook=True,
                max_pages=10,  # Limit to avoid excessive API calls
            )

            # Convert to dict format for strategies
            self._markets_cache = [m.to_dict() for m in markets]
            self._last_market_fetch = now

            logger.info(f"Fetched {len(self._markets_cache)} markets")
            return self._markets_cache

        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return self._markets_cache

    async def _run_strategies(self, markets: list[dict[str, Any]]) -> list[Signal]:
        """Run all strategies and collect signals."""
        all_signals: list[Signal] = []

        for strategy in self._strategies:
            if not strategy.enabled:
                continue

            try:
                signals = await strategy.generate_signals(markets)
                all_signals.extend(signals)

                if signals:
                    logger.info(
                        f"Strategy {strategy.name} generated {len(signals)} signals"
                    )

            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")
                strategy.record_run(error=str(e))

        return all_signals

    async def _process_signals(self, signals: list[Signal]) -> None:
        """Process signals through voting and risk engine."""
        if not signals or self._voting is None or self._risk_engine is None:
            return

        # Phase 1: Single-strategy mode - process signals directly
        # Phase 2 will use: intents = self._voting.aggregate_signals(signals)
        for signal in signals:
            intent = self._voting.process_single_signal(signal)
            if intent is None:
                continue

            try:
                # Evaluate through risk engine
                decision = await self._risk_engine.evaluate_signal(intent.primary_signal)

                if decision.approved:
                    # Execute the trade
                    result = await self._risk_engine.execute_decision(
                        decision, intent.primary_signal
                    )

                    if result.success and self._brier_calc:
                        # Track prediction for Brier score
                        self._brier_calc.record_prediction(
                            prediction_id=signal.signal_id,
                            market_ticker=signal.market_ticker,
                            strategy_name=signal.strategy_name,
                            predicted_probability=signal.target_probability,
                            direction=signal.direction,
                        )

                    if result.success:
                        logger.info(
                            f"Executed trade: {signal.direction} {decision.approved_size} "
                            f"contracts of {signal.market_ticker}"
                        )

            except Exception as e:
                logger.error(f"Failed to process signal {signal.signal_id[:8]}: {e}")

    async def _check_daily_reset(self) -> None:
        """Reset daily metrics at midnight UTC."""
        now = datetime.now(timezone.utc)

        if self._last_daily_reset is None:
            self._last_daily_reset = now
            return

        # Check if we've crossed midnight
        if now.date() > self._last_daily_reset.date():
            if self._risk_engine:
                self._risk_engine.reset_daily_breakers()

            if self._performance:
                self._performance.record_daily_snapshot()

            self._last_daily_reset = now
            logger.info("Daily metrics reset completed")

    async def _check_stop_file(self) -> bool:
        """Check if STOP file exists (manual kill switch)."""
        stop_file = Path("STOP")
        if stop_file.exists():
            logger.warning("STOP file detected - halting trading")
            return True
        return False

    async def run(self) -> None:
        """Run the main trading loop."""
        logger.info("Entering main trading loop")

        try:
            while not self._shutdown_event.is_set():
                self._loop_count += 1
                loop_start = datetime.now(timezone.utc)

                # Check for manual stop
                if await self._check_stop_file():
                    break

                # Check for daily reset
                await self._check_daily_reset()

                try:
                    # 1. Fetch market data
                    markets = await self._fetch_markets()

                    if markets:
                        # 2. Run strategies to generate signals
                        signals = await self._run_strategies(markets)

                        # 3. Process signals through voting and risk engine
                        if signals:
                            await self._process_signals(signals)

                except Exception as e:
                    logger.error(f"Error in trading loop iteration: {e}")
                    # Continue running - don't crash on transient errors

                # Log heartbeat
                if self._broker and self._loop_count % 5 == 0:  # Every 5 loops
                    status = self._broker.get_status()
                    logger.info(
                        "Heartbeat",
                        loop=self._loop_count,
                        balance=f"${status['current_balance']:.2f}",
                        trades=status["trade_count"],
                        pnl=f"${status['total_pnl']:.2f}",
                        mode=self._mode_manager.current_mode.value if self._mode_manager else "unknown",
                    )

                # Wait for next iteration
                elapsed = (datetime.now(timezone.utc) - loop_start).total_seconds()
                sleep_time = max(0, self.LOOP_INTERVAL_SECONDS - elapsed)

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=sleep_time,
                    )
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue loop

        except asyncio.CancelledError:
            logger.info("Main loop cancelled")

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        logger.info("Shutting down...")

        self._shutdown_event.set()

        # Close components
        if self._broker:
            await self._broker.close()

        if self._client:
            await self._client.close()

        if self._db:
            await self._db.close()

        logger.info("Shutdown complete")

    def get_status(self) -> dict[str, Any]:
        """Get current bot status."""
        return {
            "mode": self._mode_manager.current_mode.value if self._mode_manager else "unknown",
            "broker": self._broker.get_status() if self._broker else None,
            "risk_engine": self._risk_engine.get_status() if self._risk_engine else None,
            "strategies": [
                {"name": s.name, "enabled": s.enabled, "status": s.get_status()}
                for s in self._strategies
            ],
            "voting": self._voting.get_status() if self._voting else None,
            "metrics": {
                "brier": self._brier_calc.get_status() if self._brier_calc else None,
                "performance": self._performance.get_status() if self._performance else None,
                "calibration": self._calibration.get_status() if self._calibration else None,
            },
            "runtime": {
                "loop_count": self._loop_count,
                "markets_cached": len(self._markets_cache),
                "last_market_fetch": (
                    self._last_market_fetch.isoformat()
                    if self._last_market_fetch
                    else None
                ),
            },
            "config": {
                "config_dir": str(self.config_dir),
                "secrets_dir": str(self.secrets_dir),
                "data_dir": str(self.data_dir),
            },
        }


async def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    # Set up basic logging first
    setup_logging(
        log_level=args.log_level,
        json_format=args.json,
        log_file="logs/bot.log" if not args.no_file_log else None,
    )

    bot = TradingBot(
        config_dir=args.config_dir,
        secrets_dir=args.secrets_dir,
        data_dir=args.data_dir,
    )

    # Set up signal handlers
    loop = asyncio.get_running_loop()

    def handle_signal(sig: signal.Signals) -> None:
        logger.info(f"Received signal {sig.name}")
        asyncio.create_task(bot.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    try:
        await bot.start()
        await bot.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        await bot.stop()
        return 0
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        await bot.stop()
        return 1


def cli() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Kalshi Trading Bot v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.main                    # Run with defaults
    python -m src.main --log-level DEBUG  # Verbose logging
    python -m src.main --json             # JSON log output
        """,
    )

    parser.add_argument(
        "--config-dir",
        default="config",
        help="Directory containing YAML config files (default: config)",
    )
    parser.add_argument(
        "--secrets-dir",
        default="secrets",
        help="Directory containing secrets (default: secrets)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for database and data files (default: data)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output logs as JSON",
    )
    parser.add_argument(
        "--no-file-log",
        action="store_true",
        help="Disable file logging",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))


if __name__ == "__main__":
    cli()
