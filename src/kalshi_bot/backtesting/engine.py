"""Main backtest engine that orchestrates the entire backtest process."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from kalshi_bot.arbitrage.strategies.base import ArbitrageStrategy
from kalshi_bot.arbitrage.strategies.multi_outcome import MultiOutcomeStrategy
from kalshi_bot.arbitrage.strategies.single_market import SingleMarketStrategy
from kalshi_bot.backtesting.data_loader import HistoricalDataLoader
from kalshi_bot.backtesting.metrics import BacktestMetrics, calculate_metrics
from kalshi_bot.backtesting.position_tracker import PositionTracker
from kalshi_bot.backtesting.simulator import TradeSimulator
from kalshi_bot.config.settings import Settings
from kalshi_bot.core.types import ArbitrageOpportunity, MarketData, OrderBook
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    start_date: datetime | None = None
    end_date: datetime | None = None
    initial_balance: float = 10000.0
    max_position_per_market: int = 100
    max_position_pct: float = 0.10
    min_profit_threshold: float = 0.01  # Minimum expected profit in dollars
    enable_single_market: bool = True
    enable_multi_outcome: bool = True
    max_opportunities_per_step: int = 5
    verbose: bool = False


@dataclass
class BacktestProgress:
    """Progress information during backtest."""

    total_steps: int = 0
    current_step: int = 0
    current_timestamp: datetime | None = None
    opportunities_found: int = 0
    trades_executed: int = 0
    positions_settled: int = 0


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    success: bool
    config: BacktestConfig
    metrics: BacktestMetrics
    progress: BacktestProgress
    error_message: str | None = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Backtest execution duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "metrics": self.metrics.to_dict(),
            "duration_seconds": self.duration_seconds,
            "opportunities_found": self.progress.opportunities_found,
            "trades_executed": self.progress.trades_executed,
            "positions_settled": self.progress.positions_settled,
            "error_message": self.error_message,
        }


class BacktestEngine:
    """
    Main backtest engine that orchestrates historical replay.

    Workflow:
    1. Load historical data from database
    2. Iterate through timestamps chronologically
    3. At each timestamp:
       - Load market state (prices, orderbooks)
       - Run arbitrage strategies to find opportunities
       - Simulate trade execution
       - Check for settlements and process them
    4. Calculate final metrics
    """

    def __init__(
        self,
        db: Database,
        settings: Settings | None = None,
    ) -> None:
        """
        Initialize the backtest engine.

        Args:
            db: Database connection
            settings: Optional settings (uses defaults if not provided)
        """
        self._db = db
        self._settings = settings
        self._data_loader = HistoricalDataLoader(db)
        self._simulator: TradeSimulator | None = None
        self._position_tracker: PositionTracker | None = None
        self._strategies: list[ArbitrageStrategy] = []
        self._progress = BacktestProgress()
        self._equity_snapshots: list[tuple[datetime, float]] = []

    async def run(
        self,
        config: BacktestConfig | None = None,
        progress_callback: callable | None = None,
    ) -> BacktestResult:
        """
        Run a complete backtest.

        Args:
            config: Backtest configuration
            progress_callback: Optional callback for progress updates

        Returns:
            BacktestResult with metrics and details
        """
        config = config or BacktestConfig()
        result = BacktestResult(
            success=False,
            config=config,
            metrics=BacktestMetrics(),
            progress=self._progress,
        )

        try:
            # Initialize components
            await self._initialize(config)

            # Get timestamps to process
            timestamps = await self._get_timestamps(config)
            if not timestamps:
                result.error_message = "No historical data found in date range"
                return result

            self._progress.total_steps = len(timestamps)
            logger.info(f"Starting backtest with {len(timestamps)} time steps")

            # Pre-load all settlements for efficiency
            all_settlements = await self._data_loader.get_all_settlements_before(
                timestamps[-1]
            )

            # Process each timestamp
            prev_timestamp = None
            for i, timestamp in enumerate(timestamps):
                self._progress.current_step = i + 1
                self._progress.current_timestamp = timestamp

                if progress_callback:
                    progress_callback(self._progress)

                # Process settlements since last timestamp
                if prev_timestamp:
                    await self._process_settlements(prev_timestamp, timestamp, all_settlements)

                # Load market state
                markets, orderbooks = await self._data_loader.load_market_state(timestamp)

                if not markets:
                    prev_timestamp = timestamp
                    continue

                # Find opportunities
                opportunities = self._find_opportunities(markets, orderbooks, config)
                self._progress.opportunities_found += len(opportunities)

                # Execute opportunities
                for opp in opportunities[:config.max_opportunities_per_step]:
                    sim_result = self._simulator.simulate_opportunity(opp, timestamp)
                    if sim_result.success:
                        self._progress.trades_executed += len(sim_result.trades)
                        self._position_tracker.record_trades(sim_result.trades, timestamp)

                # Record equity snapshot
                equity = self._calculate_current_equity()
                self._equity_snapshots.append((timestamp, equity))

                prev_timestamp = timestamp

                if config.verbose and i % 100 == 0:
                    logger.info(
                        f"Step {i+1}/{len(timestamps)}: "
                        f"balance=${self._simulator.balance:.2f}, "
                        f"positions={self._position_tracker.total_open_positions}"
                    )

            # Final settlement of remaining positions
            await self._settle_remaining_positions(timestamps[-1], all_settlements)

            # Calculate metrics
            result.metrics = calculate_metrics(
                self._simulator,
                self._position_tracker,
                self._equity_snapshots,
            )

            result.success = True
            result.end_time = datetime.utcnow()

            logger.info(
                f"Backtest complete: P&L=${result.metrics.total_pnl:.2f}, "
                f"trades={self._progress.trades_executed}"
            )

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            result.error_message = str(e)
            result.end_time = datetime.utcnow()

        return result

    async def _initialize(self, config: BacktestConfig) -> None:
        """Initialize backtest components."""
        self._simulator = TradeSimulator(
            initial_balance=config.initial_balance,
            max_position_per_market=config.max_position_per_market,
            max_position_pct=config.max_position_pct,
        )
        self._position_tracker = PositionTracker()
        self._progress = BacktestProgress()
        self._equity_snapshots = []

        # Initialize strategies
        self._strategies = []
        if config.enable_single_market:
            self._strategies.append(SingleMarketStrategy(self._settings))
        if config.enable_multi_outcome:
            self._strategies.append(MultiOutcomeStrategy(self._settings))

    async def _get_timestamps(self, config: BacktestConfig) -> list[datetime]:
        """Get timestamps to process based on config."""
        # Get available date range
        earliest, latest = await self._data_loader.get_date_range()
        if not earliest or not latest:
            return []

        start = config.start_date or earliest
        end = config.end_date or latest

        # Clamp to available data
        start = max(start, earliest)
        end = min(end, latest)

        return await self._data_loader.get_timestamps_in_range(start, end)

    def _find_opportunities(
        self,
        markets: list[MarketData],
        orderbooks: dict[str, OrderBook],
        config: BacktestConfig,
    ) -> list[ArbitrageOpportunity]:
        """Find arbitrage opportunities at current timestamp."""
        opportunities: list[ArbitrageOpportunity] = []

        for strategy in self._strategies:
            try:
                opps = strategy.find_opportunities(markets, orderbooks)
                # Filter by minimum profit threshold
                opps = [
                    o for o in opps
                    if o.net_profit >= config.min_profit_threshold
                ]
                opportunities.extend(opps)
            except Exception as e:
                logger.warning(f"Strategy {strategy.__class__.__name__} error: {e}")

        # Sort by expected profit
        opportunities.sort(key=lambda x: x.net_profit, reverse=True)

        return opportunities

    async def _process_settlements(
        self,
        prev_timestamp: datetime,
        current_timestamp: datetime,
        all_settlements: dict[str, dict],
    ) -> None:
        """Process settlements that occurred between timestamps."""
        # Check if any of our open positions have settled
        for ticker in list(self._position_tracker.open_positions.keys()):
            settlement = all_settlements.get(ticker)
            if settlement:
                settled_at = datetime.fromisoformat(settlement["confirmed_at"])
                if prev_timestamp < settled_at <= current_timestamp:
                    outcome = settlement["result"]
                    results = self._position_tracker.settle_market(
                        ticker, outcome, settled_at
                    )
                    self._progress.positions_settled += len(results)

                    # Add settlement payouts to balance
                    total_payout = sum(r.payout for r in results)
                    self._simulator.add_funds(total_payout)

    async def _settle_remaining_positions(
        self,
        final_timestamp: datetime,
        all_settlements: dict[str, dict],
    ) -> None:
        """Settle any remaining open positions at end of backtest."""
        for ticker in list(self._position_tracker.open_positions.keys()):
            settlement = all_settlements.get(ticker)
            if settlement:
                outcome = settlement["result"]
                settled_at = datetime.fromisoformat(settlement["confirmed_at"])
                results = self._position_tracker.settle_market(
                    ticker, outcome, settled_at
                )
                self._progress.positions_settled += len(results)

                total_payout = sum(r.payout for r in results)
                self._simulator.add_funds(total_payout)

    def _calculate_current_equity(self) -> float:
        """Calculate current equity (cash + unrealized value)."""
        # Cash balance
        equity = self._simulator.balance

        # Add position values at current prices (approximation using cost basis)
        equity += self._position_tracker.total_exposure

        return equity

    async def get_data_summary(self) -> dict[str, Any]:
        """Get summary of available historical data."""
        return await self._data_loader.get_snapshot_stats()

    def get_progress(self) -> BacktestProgress:
        """Get current backtest progress."""
        return self._progress
