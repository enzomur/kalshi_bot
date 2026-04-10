"""Main bot class orchestrating all components."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from kalshi_bot.api.client import KalshiAPIClient
from kalshi_bot.api.websocket import WebSocketManager
from kalshi_bot.arbitrage.detector import ArbitrageDetector
from kalshi_bot.config.settings import Settings
from kalshi_bot.core.types import ArbitrageOpportunity, OrderType, Side
from kalshi_bot.execution.executor import TradeExecutor
from kalshi_bot.execution.paper_trading import PaperTradingExecutor
from kalshi_bot.optimization.position_sizer import PositionSizer
from kalshi_bot.persistence.database import Database
from kalshi_bot.persistence.models import (
    AuditRepository,
    CircuitBreakerRepository,
    OpportunityRepository,
    OrderRepository,
    PortfolioSnapshotRepository,
    PositionRepository,
    TradeRepository,
)
from kalshi_bot.portfolio.manager import PortfolioManager
from kalshi_bot.portfolio.position_monitor import PositionMonitor
from kalshi_bot.portfolio.profit_lock import ProfitLock
from kalshi_bot.risk.circuit_breaker import CircuitBreaker
from kalshi_bot.risk.limits import PositionLimits
from kalshi_bot.risk.manager import RiskManager
from kalshi_bot.utils.logging import get_logger

# ML imports
from kalshi_bot.ml.data_collector import MarketSnapshotCollector
from kalshi_bot.ml.outcome_tracker import OutcomeTracker
from kalshi_bot.ml.inference.predictor import EdgePredictor
from kalshi_bot.ml.training.trainer import ModelTrainer
from kalshi_bot.ml.training.scheduler import TrainingScheduler
from kalshi_bot.ml.self_correction.monitor import PerformanceMonitor
from kalshi_bot.ml.self_correction.adjuster import PositionAdjuster
from kalshi_bot.ml.self_correction.disabler import StrategyDisabler

logger = get_logger(__name__)


class KalshiArbitrageBot:
    """
    Main bot class that orchestrates all trading components.

    The bot runs a continuous loop that:
    1. Syncs portfolio state with Kalshi
    2. Detects arbitrage opportunities
    3. Evaluates risk for each opportunity
    4. Sizes positions optimally
    5. Executes approved trades
    6. Updates state and persists data
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize the bot.

        Args:
            settings: Application settings
        """
        self.settings = settings

        self._running = False
        self._shutdown_event = asyncio.Event()

        self._db: Database | None = None
        self._api_client: KalshiAPIClient | None = None
        self._ws_manager: WebSocketManager | None = None

        self._audit_repo: AuditRepository | None = None
        self._trade_repo: TradeRepository | None = None
        self._order_repo: OrderRepository | None = None
        self._opportunity_repo: OpportunityRepository | None = None
        self._position_repo: PositionRepository | None = None
        self._snapshot_repo: PortfolioSnapshotRepository | None = None
        self._breaker_repo: CircuitBreakerRepository | None = None

        self._profit_lock: ProfitLock | None = None
        self._portfolio_manager: PortfolioManager | None = None
        self._position_monitor: PositionMonitor | None = None
        self._circuit_breaker: CircuitBreaker | None = None
        self._position_limits: PositionLimits | None = None
        self._risk_manager: RiskManager | None = None
        self._position_sizer: PositionSizer | None = None
        self._arbitrage_detector: ArbitrageDetector | None = None
        self._executor: TradeExecutor | None = None
        self._paper_executor: PaperTradingExecutor | None = None

        # ML components
        self._snapshot_collector: MarketSnapshotCollector | None = None
        self._outcome_tracker: OutcomeTracker | None = None
        self._edge_predictor: EdgePredictor | None = None
        self._model_trainer: ModelTrainer | None = None
        self._training_scheduler: TrainingScheduler | None = None
        self._performance_monitor: PerformanceMonitor | None = None
        self._position_adjuster: PositionAdjuster | None = None
        self._strategy_disabler: StrategyDisabler | None = None
        self._ml_enabled = False
        self._last_ml_snapshot: datetime | None = None
        self._snapshot_collection_task: asyncio.Task | None = None

        # Check paper trading mode
        self._paper_trading_mode = getattr(
            settings.trading, 'paper_trading_mode', True
        )

        self._scan_interval = 5.0
        self._snapshot_interval = 60.0
        self._last_snapshot: datetime | None = None
        self._trade_cooldown: dict[str, datetime] = {}  # market -> last trade time
        self._cooldown_seconds = 300  # 5 min cooldown per market

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing Kalshi Arbitrage Bot...")

        self._db = Database(self.settings.database_path)
        await self._db.initialize()

        self._audit_repo = AuditRepository(self._db)
        self._trade_repo = TradeRepository(self._db)
        self._order_repo = OrderRepository(self._db)
        self._opportunity_repo = OpportunityRepository(self._db)
        self._position_repo = PositionRepository(self._db)
        self._snapshot_repo = PortfolioSnapshotRepository(self._db)
        self._breaker_repo = CircuitBreakerRepository(self._db)

        self._api_client = KalshiAPIClient(self.settings)
        await self._api_client.connect()

        self._ws_manager = WebSocketManager(self.settings)

        self._profit_lock = ProfitLock(self.settings, self._db)

        self._portfolio_manager = PortfolioManager(
            self.settings,
            self._api_client,
            self._profit_lock,
            self._position_repo,
            self._snapshot_repo,
        )
        await self._portfolio_manager.initialize()

        self._position_monitor = PositionMonitor(self.settings)

        self._circuit_breaker = CircuitBreaker(self.settings, self._breaker_repo)
        self._circuit_breaker.set_peak_value(self._portfolio_manager.total_value)

        self._position_limits = PositionLimits(self.settings)

        self._risk_manager = RiskManager(
            self.settings,
            self._circuit_breaker,
            self._position_limits,
            self._audit_repo,
        )

        self._position_sizer = PositionSizer(self.settings)

        self._arbitrage_detector = ArbitrageDetector(
            self.settings,
            self._api_client,
            self._opportunity_repo,
        )

        self._executor = TradeExecutor(
            self.settings,
            self._api_client,
            self._order_repo,
            self._trade_repo,
        )

        # Initialize paper trading executor
        self._paper_executor = PaperTradingExecutor(
            self.settings,
            self._db,
        )
        # Use configured initial_principal for paper trading, not real balance
        paper_balance = self.settings.portfolio.initial_principal if self._paper_trading_mode else self._portfolio_manager.total_value
        self._paper_executor.set_initial_balance(paper_balance)

        # Set initial value for portfolio hard stop
        self._circuit_breaker.set_initial_value(self._portfolio_manager.total_value)

        if self._paper_trading_mode:
            logger.info("Paper trading mode ENABLED - no real trades will be executed")

        # Initialize ML components
        await self._initialize_ml_components()

        await self._audit_repo.log(
            "bot_initialized",
            {
                "settings": "loaded",
                "environment": self.settings.environment.value,
                "paper_trading_mode": self._paper_trading_mode,
                "ml_enabled": self._ml_enabled,
            },
            "info",
            component="bot",
        )

        logger.info("Bot initialization complete")

    async def _initialize_ml_components(self) -> None:
        """Initialize ML components for self-learning trading."""
        ml_settings = self.settings.ml
        self._ml_enabled = ml_settings.enabled

        # Always initialize data collection (even if ML trading disabled)
        self._snapshot_collector = MarketSnapshotCollector(
            db=self._db,
            api_client=self._api_client,
            interval_seconds=ml_settings.snapshot_interval_seconds,
            min_volume=ml_settings.min_volume_for_snapshot,
            min_open_interest=ml_settings.min_open_interest_for_snapshot,
        )

        self._outcome_tracker = OutcomeTracker(self._db)

        self._model_trainer = ModelTrainer(self._db)

        self._training_scheduler = TrainingScheduler(
            db=self._db,
            trainer=self._model_trainer,
        )

        if self._ml_enabled:
            self._edge_predictor = EdgePredictor(
                db=self._db,
                api_client=self._api_client,
                min_edge=ml_settings.min_edge_threshold,
                min_confidence=ml_settings.min_confidence,
            )

            self._performance_monitor = PerformanceMonitor(
                db=self._db,
                strategy_name="ml_edge",
            )
            self._performance_monitor.set_peak_value(self._portfolio_manager.total_value)

            self._position_adjuster = PositionAdjuster(
                db=self._db,
                monitor=self._performance_monitor,
            )

            self._strategy_disabler = StrategyDisabler(
                db=self._db,
                monitor=self._performance_monitor,
            )
            await self._strategy_disabler.load_state()

            logger.info("ML trading ENABLED")
        else:
            logger.info("ML trading DISABLED (data collection active)")

    async def start(self) -> None:
        """Start the bot's main trading loop."""
        if self._running:
            logger.warning("Bot is already running")
            return

        self._running = True
        self._shutdown_event.clear()

        logger.info("Starting Kalshi Arbitrage Bot...")

        await self._audit_repo.log(
            "bot_started",
            {"portfolio_value": self._portfolio_manager.total_value},
            "info",
            component="bot",
        )

        # Start background tasks
        background_tasks = []

        # Start snapshot collection in background
        if self._snapshot_collector:
            self._snapshot_collection_task = asyncio.create_task(
                self._snapshot_collector.start_collection_loop(self._shutdown_event)
            )
            background_tasks.append(self._snapshot_collection_task)

        # Start training scheduler if ML enabled
        if self._ml_enabled and self._training_scheduler:
            training_task = asyncio.create_task(
                self._training_scheduler.start(self._shutdown_event)
            )
            background_tasks.append(training_task)

        try:
            await self._main_loop()
        except asyncio.CancelledError:
            logger.info("Bot main loop cancelled")
        except Exception as e:
            logger.error(f"Bot main loop error: {e}")
            raise
        finally:
            self._running = False
            # Cancel background tasks
            for task in background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self._shutdown_event.set()

        if self._executor:
            await self._executor.cancel_all_pending()

        if self._audit_repo:
            await self._audit_repo.log(
                "bot_stopped",
                {},
                "info",
                component="bot",
            )

    async def shutdown(self) -> None:
        """Shutdown and cleanup all resources."""
        logger.info("Shutting down bot...")

        await self.stop()

        if self._ws_manager:
            await self._ws_manager.disconnect()

        if self._api_client:
            await self._api_client.close()

        if self._db:
            await self._db.close()

        logger.info("Bot shutdown complete")

    async def _main_loop(self) -> None:
        """Main trading loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._trading_cycle()

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self._scan_interval,
                    )
                    break
                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await self._risk_manager.record_api_error()
                await asyncio.sleep(self._scan_interval)

    async def _trading_cycle(self) -> None:
        """Execute one trading cycle."""
        await self._portfolio_manager.ensure_synced()

        await self._risk_manager.update_portfolio_value(
            self._portfolio_manager.total_value
        )

        await self._maybe_take_snapshot()

        # ML: Check for settlements and update prediction outcomes
        if self._outcome_tracker:
            await self._outcome_tracker.check_settlements()
            await self._outcome_tracker.update_prediction_outcomes()

        # ML: Update performance metrics
        if self._ml_enabled and self._performance_monitor:
            await self._performance_monitor.update_metrics(
                self._portfolio_manager.total_value
            )

        if not self._risk_manager.can_trade():
            logger.debug("Trading disabled by risk manager")
            return

        # ML: Check if strategy should be disabled
        if self._ml_enabled and self._strategy_disabler:
            state = await self._strategy_disabler.check_disable_conditions()
            if not state.is_enabled:
                logger.warning(f"ML strategy disabled: {state.disabled_reason}")

        await self._check_take_profit()

        # Get traditional arbitrage opportunities
        opportunities = await self._arbitrage_detector.detect_opportunities()

        # ML: Get ML-based opportunities if enabled and strategy not disabled
        if self._ml_enabled and self._edge_predictor:
            if self._strategy_disabler is None or self._strategy_disabler.is_enabled:
                ml_opportunities = await self._get_ml_opportunities()
                opportunities.extend(ml_opportunities)

        if not opportunities:
            logger.debug("No opportunities found")
            return

        for opportunity in opportunities:
            if self._shutdown_event.is_set():
                break

            await self._process_opportunity(opportunity)

    async def _get_ml_opportunities(self) -> list[ArbitrageOpportunity]:
        """Get ML-based trading opportunities."""
        if not self._edge_predictor:
            return []

        # Use paper trading balance if in paper mode, otherwise real balance
        if self._paper_trading_mode:
            available_capital = self._paper_executor.get_status().get("balance", 0)
        else:
            available_capital = self._portfolio_manager.available_for_trading

        # Apply position adjustment from self-correction
        if self._position_adjuster:
            multiplier = await self._position_adjuster.get_current_multiplier()
            available_capital *= multiplier
            if multiplier < 1.0:
                logger.debug(f"ML capital reduced to {multiplier:.0%} due to performance")

        if available_capital < 1.0:
            return []

        try:
            trading_opps = await self._edge_predictor.get_trading_opportunities(
                available_capital=available_capital,
                max_opportunities=5,
            )

            # Convert to ArbitrageOpportunity format
            return [opp.to_arbitrage_opportunity() for opp in trading_opps]

        except Exception as e:
            logger.error(f"Error getting ML opportunities: {e}")
            return []

    async def _process_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
    ) -> None:
        """
        Process a single arbitrage opportunity.

        Args:
            opportunity: The opportunity to process
        """
        # Check cooldown for all markets in this opportunity
        now = datetime.utcnow()
        for market in opportunity.markets:
            last_trade = self._trade_cooldown.get(market)
            if last_trade and (now - last_trade).total_seconds() < self._cooldown_seconds:
                logger.debug(f"Market {market} on cooldown, skipping")
                return

        risk_check = await self._risk_manager.pre_trade_check(
            opportunity,
            self._portfolio_manager.positions,
            self._portfolio_manager.available_for_trading,
            self._portfolio_manager.total_value,
        )

        if not risk_check.approved:
            logger.debug(
                f"Opportunity {opportunity.opportunity_id} rejected: {risk_check.reason}"
            )
            return

        max_quantity = risk_check.max_quantity or opportunity.max_quantity

        sizing = self._position_sizer.size_single_opportunity(
            opportunity,
            self._portfolio_manager.available_for_trading,
        )

        quantity = min(sizing.recommended_quantity, max_quantity)

        if quantity < 1:
            logger.debug(
                f"Opportunity {opportunity.opportunity_id}: insufficient quantity"
            )
            return

        # Use paper trading executor if enabled, otherwise real executor
        if self._paper_trading_mode:
            logger.info(
                f"[PAPER] Executing opportunity {opportunity.opportunity_id}: "
                f"{quantity} contracts, expected profit ${sizing.expected_profit:.4f}"
            )
            result = await self._paper_executor.execute_opportunity(opportunity, quantity)
        else:
            logger.info(
                f"Executing opportunity {opportunity.opportunity_id}: "
                f"{quantity} contracts, expected profit ${sizing.expected_profit:.4f}"
            )
            result = await self._executor.execute_opportunity(opportunity, quantity)

        # Update cooldown for all markets in this opportunity
        now = datetime.utcnow()
        for market in opportunity.markets:
            self._trade_cooldown[market] = now

        profit_loss = 0.0
        if result.success:
            profit_loss = (100 * result.filled_quantity / 100) - result.total_cost - result.total_fees

        await self._risk_manager.post_trade_update(
            result.success,
            profit_loss,
            opportunity.opportunity_id,
            result.error_message,
        )

        if self._opportunity_repo:
            status = "executed" if result.success else "failed"
            await self._opportunity_repo.update_status(
                opportunity.opportunity_id,
                status,
                {
                    "filled_quantity": result.filled_quantity,
                    "total_cost": result.total_cost,
                    "total_fees": result.total_fees,
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error_message,
                },
            )

        # Only sync with Kalshi if we made real trades
        if not self._paper_trading_mode:
            await self._portfolio_manager.sync_with_kalshi()

    async def _maybe_take_snapshot(self) -> None:
        """Take a portfolio snapshot if interval has passed."""
        now = datetime.utcnow()

        if self._last_snapshot is None:
            self._last_snapshot = now
            await self._portfolio_manager.take_snapshot()
            return

        elapsed = (now - self._last_snapshot).total_seconds()
        if elapsed >= self._snapshot_interval:
            self._last_snapshot = now
            await self._portfolio_manager.take_snapshot()

    async def _check_take_profit(self) -> None:
        """Check positions for take-profit conditions and execute sells."""
        if not self._position_monitor or not self._api_client:
            return

        positions = self._portfolio_manager.positions

        async for signal in self._position_monitor.check_positions(
            positions, self._api_client
        ):
            if self._shutdown_event.is_set():
                break

            await self._execute_take_profit(signal)

    async def _execute_take_profit(self, signal) -> None:
        """
        Execute a take-profit sell order.

        Args:
            signal: SellSignal from position monitor
        """
        position = signal.position

        logger.info(
            f"Executing take_profit sell: {position.market_ticker} "
            f"{position.side.value} x{position.quantity} @ {signal.current_price:.0f}c "
            f"(+{signal.pnl_pct:.1%})"
        )

        try:
            sell_side = Side.NO if position.side == Side.YES else Side.YES

            order = await self._executor.execute_single_order(
                market_ticker=position.market_ticker,
                side=sell_side,
                quantity=position.quantity,
                price=int(100 - signal.current_price),
                order_type=OrderType.LIMIT,
            )

            if self._audit_repo:
                await self._audit_repo.log(
                    "take_profit_executed",
                    {
                        "market_ticker": position.market_ticker,
                        "side": position.side.value,
                        "quantity": position.quantity,
                        "entry_price": position.average_price,
                        "exit_price": signal.current_price,
                        "pnl_pct": signal.pnl_pct,
                        "order_id": order.order_id,
                        "order_status": order.status.value,
                    },
                    "info",
                    component="position_monitor",
                )

            await self._portfolio_manager.sync_with_kalshi()

        except Exception as e:
            logger.error(f"Failed to execute take-profit for {position.market_ticker}: {e}")

            if self._audit_repo:
                await self._audit_repo.log(
                    "take_profit_failed",
                    {
                        "market_ticker": position.market_ticker,
                        "error": str(e),
                    },
                    "error",
                    component="position_monitor",
                )

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive bot status."""
        ml_status = None
        if self._ml_enabled:
            ml_status = {
                "enabled": self._ml_enabled,
                "predictor": self._edge_predictor.get_status() if self._edge_predictor else None,
                "performance_monitor": self._performance_monitor.get_status() if self._performance_monitor else None,
                "position_adjuster": self._position_adjuster.get_status() if self._position_adjuster else None,
                "strategy_disabler": self._strategy_disabler.get_status() if self._strategy_disabler else None,
            }

        return {
            "running": self._running,
            "paper_trading_mode": self._paper_trading_mode,
            "portfolio": self._portfolio_manager.get_status() if self._portfolio_manager else None,
            "risk": self._risk_manager.get_status() if self._risk_manager else None,
            "detector": self._arbitrage_detector.get_status() if self._arbitrage_detector else None,
            "executor": self._executor.get_status() if self._executor else None,
            "paper_executor": self._paper_executor.get_status() if self._paper_executor else None,
            "position_sizer": self._position_sizer.get_status() if self._position_sizer else None,
            "api_rate_limiter": self._api_client.get_rate_limiter_status() if self._api_client else None,
            "ml": ml_status,
            "snapshot_collector": self._snapshot_collector.get_status() if self._snapshot_collector else None,
            "outcome_tracker": self._outcome_tracker.get_status() if self._outcome_tracker else None,
        }

    @property
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self._running

    @property
    def portfolio_manager(self) -> PortfolioManager | None:
        """Get portfolio manager."""
        return self._portfolio_manager

    @property
    def risk_manager(self) -> RiskManager | None:
        """Get risk manager."""
        return self._risk_manager

    @property
    def arbitrage_detector(self) -> ArbitrageDetector | None:
        """Get arbitrage detector."""
        return self._arbitrage_detector

    @property
    def model_trainer(self) -> ModelTrainer | None:
        """Get model trainer."""
        return self._model_trainer

    @property
    def edge_predictor(self) -> EdgePredictor | None:
        """Get edge predictor."""
        return self._edge_predictor

    @property
    def strategy_disabler(self) -> StrategyDisabler | None:
        """Get strategy disabler."""
        return self._strategy_disabler

    @property
    def training_scheduler(self) -> TrainingScheduler | None:
        """Get training scheduler."""
        return self._training_scheduler
