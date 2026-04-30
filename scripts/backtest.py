#!/usr/bin/env python3
"""Backtest harness for strategy validation.

Replays historical market data through strategies to validate
performance before live trading.

Usage:
    python scripts/backtest.py --data historical_data.json
    python scripts/backtest.py --start 2024-01-01 --end 2024-03-01
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.types import Signal, Side
from src.strategies.base import Strategy
from src.strategies.weather import WeatherStrategy
from src.strategies.calibration import CalibrationStrategy
from src.voting.ensemble import VotingEnsemble
from src.metrics import (
    BrierCalculator,
    PerformanceTracker,
    CalibrationCurve,
    TradeResult,
)
from src.observability.logging import setup_logging, get_logger

logger = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    start_date: datetime
    end_date: datetime
    initial_balance: float = 10000.0
    strategies: list[str] = field(default_factory=lambda: ["weather", "calibration"])
    min_edge: float = 0.05
    kelly_fraction: float = 0.25
    max_position: int = 100
    fees_per_trade: float = 0.01  # 1%


@dataclass
class BacktestTrade:
    """A simulated trade from backtest."""

    signal: Signal
    entry_time: datetime
    entry_price: float
    quantity: int
    direction: str
    exit_time: datetime | None = None
    exit_price: float | None = None
    outcome: bool | None = None  # True = won, False = lost
    pnl: float = 0.0

    def settle(self, market_result: str, settlement_price: float) -> None:
        """Settle the trade based on market result."""
        self.exit_time = datetime.now(timezone.utc)
        self.exit_price = settlement_price

        # Determine if we won
        if self.direction == "yes":
            self.outcome = market_result == "yes"
            if self.outcome:
                self.pnl = (1.0 - self.entry_price) * self.quantity
            else:
                self.pnl = -self.entry_price * self.quantity
        else:
            self.outcome = market_result == "no"
            if self.outcome:
                self.pnl = (1.0 - self.entry_price) * self.quantity
            else:
                self.pnl = -self.entry_price * self.quantity


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    config: BacktestConfig
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    roi: float
    win_rate: float
    avg_edge: float
    brier_score: float | None
    sharpe_ratio: float | None
    max_drawdown: float
    trades: list[BacktestTrade]
    daily_returns: list[float]
    strategy_breakdown: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "initial_balance": self.config.initial_balance,
                "strategies": self.config.strategies,
            },
            "summary": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "total_pnl": self.total_pnl,
                "roi": self.roi,
                "win_rate": self.win_rate,
                "avg_edge": self.avg_edge,
                "brier_score": self.brier_score,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
            },
            "strategy_breakdown": self.strategy_breakdown,
        }

    def print_report(self) -> None:
        """Print formatted backtest report."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {self.config.start_date.date()} to {self.config.end_date.date()}")
        print(f"Initial Balance: ${self.config.initial_balance:,.2f}")
        print(f"Strategies: {', '.join(self.config.strategies)}")
        print("-" * 60)
        print(f"Total Trades:    {self.total_trades}")
        print(f"Winning Trades:  {self.winning_trades}")
        print(f"Losing Trades:   {self.losing_trades}")
        print(f"Win Rate:        {self.win_rate:.1%}")
        print("-" * 60)
        print(f"Total P&L:       ${self.total_pnl:,.2f}")
        print(f"ROI:             {self.roi:.1%}")
        print(f"Max Drawdown:    {self.max_drawdown:.1%}")

        if self.sharpe_ratio:
            print(f"Sharpe Ratio:    {self.sharpe_ratio:.2f}")
        if self.brier_score:
            print(f"Brier Score:     {self.brier_score:.4f}")

        print("-" * 60)
        print("By Strategy:")
        for strategy, stats in self.strategy_breakdown.items():
            print(f"  {strategy}:")
            print(f"    Trades: {stats['trades']}, Win Rate: {stats['win_rate']:.1%}, P&L: ${stats['pnl']:.2f}")

        print("=" * 60 + "\n")


class BacktestEngine:
    """Engine for running strategy backtests.

    Simulates market data replay and strategy execution.
    """

    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self._balance = config.initial_balance
        self._peak_balance = config.initial_balance
        self._trades: list[BacktestTrade] = []
        self._open_positions: dict[str, BacktestTrade] = {}

        # Metrics
        self._brier = BrierCalculator()
        self._performance = PerformanceTracker(initial_balance=config.initial_balance)
        self._calibration = CalibrationCurve()

        # Strategies (initialized lazily)
        self._strategies: list[Strategy] = []
        self._voting: VotingEnsemble | None = None

    async def initialize_strategies(self) -> None:
        """Initialize enabled strategies."""
        self._strategies = []

        if "weather" in self.config.strategies:
            self._strategies.append(
                WeatherStrategy(
                    enabled=True,
                    min_edge=self.config.min_edge,
                )
            )

        if "calibration" in self.config.strategies:
            self._strategies.append(
                CalibrationStrategy(
                    enabled=True,
                    min_edge=self.config.min_edge,
                )
            )

        self._voting = VotingEnsemble(
            min_edge=self.config.min_edge,
            min_agreement=1,  # Single-strategy mode for backtest
        )

        logger.info(f"Initialized {len(self._strategies)} strategies for backtest")

    async def run(
        self,
        historical_data: list[dict[str, Any]],
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            historical_data: List of market snapshots with timestamps

        Returns:
            BacktestResult with performance metrics
        """
        await self.initialize_strategies()

        # Sort data by timestamp
        sorted_data = sorted(
            historical_data,
            key=lambda x: x.get("timestamp", ""),
        )

        logger.info(f"Running backtest on {len(sorted_data)} market snapshots")

        daily_pnl: dict[str, float] = {}
        current_date = None

        for snapshot in sorted_data:
            timestamp_str = snapshot.get("timestamp")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            # Check date bounds
            if timestamp < self.config.start_date or timestamp > self.config.end_date:
                continue

            # Track daily P&L
            date_key = timestamp.date().isoformat()
            if current_date != date_key:
                current_date = date_key
                daily_pnl[date_key] = 0.0

            # Process market settlements
            self._process_settlements(snapshot)

            # Run strategies on current snapshot
            markets = snapshot.get("markets", [snapshot])
            await self._process_snapshot(markets, timestamp)

            # Update daily P&L
            daily_pnl[date_key] = sum(t.pnl for t in self._trades if t.outcome is not None)

        # Settle remaining positions at end
        self._settle_remaining_positions()

        # Calculate results
        return self._calculate_results(list(daily_pnl.values()))

    async def _process_snapshot(
        self,
        markets: list[dict[str, Any]],
        timestamp: datetime,
    ) -> None:
        """Process a single market snapshot."""
        all_signals: list[Signal] = []

        for strategy in self._strategies:
            try:
                signals = await strategy.generate_signals(markets)
                all_signals.extend(signals)
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} error: {e}")

        if not all_signals or self._voting is None:
            return

        # Process signals through voting
        for signal in all_signals:
            intent = self._voting.process_single_signal(signal)
            if intent is None:
                continue

            # Check if we already have a position
            if signal.market_ticker in self._open_positions:
                continue

            # Simulate trade execution
            self._execute_trade(signal, timestamp)

    def _execute_trade(self, signal: Signal, timestamp: datetime) -> None:
        """Simulate trade execution."""
        # Calculate position size using Kelly
        edge = signal.edge
        market_price = signal.metadata.get("market_price_cents", 50) / 100

        kelly_fraction = self.config.kelly_fraction
        full_kelly = (edge * (1 - market_price) - (1 - edge) * market_price) / (1 - market_price)
        position_fraction = max(0, full_kelly * kelly_fraction)

        position_dollars = self._balance * position_fraction
        quantity = min(
            int(position_dollars / market_price),
            self.config.max_position,
        )

        if quantity <= 0:
            return

        trade = BacktestTrade(
            signal=signal,
            entry_time=timestamp,
            entry_price=market_price,
            quantity=quantity,
            direction=signal.direction,
        )

        self._trades.append(trade)
        self._open_positions[signal.market_ticker] = trade

        # Update balance
        cost = market_price * quantity * (1 + self.config.fees_per_trade)
        self._balance -= cost

        # Track prediction for Brier
        self._brier.record_prediction(
            prediction_id=signal.signal_id,
            market_ticker=signal.market_ticker,
            strategy_name=signal.strategy_name,
            predicted_probability=signal.target_probability,
            direction=signal.direction,
        )

        logger.debug(
            f"Backtest trade: {signal.direction} {quantity} @ {market_price:.2f} "
            f"({signal.market_ticker})"
        )

    def _process_settlements(self, snapshot: dict[str, Any]) -> None:
        """Process any market settlements in the snapshot."""
        markets = snapshot.get("markets", [snapshot])

        for market in markets:
            ticker = market.get("ticker", "")
            result = market.get("result")

            if result and ticker in self._open_positions:
                trade = self._open_positions[ticker]
                settlement_price = 1.0 if result == trade.direction else 0.0
                trade.settle(result, settlement_price)

                # Update balance
                self._balance += trade.pnl

                # Track peak for drawdown
                if self._balance > self._peak_balance:
                    self._peak_balance = self._balance

                # Update Brier
                self._brier.resolve_prediction(
                    trade.signal.signal_id,
                    trade.outcome or False,
                )

                # Update calibration
                self._calibration.add_prediction(
                    trade.signal.target_probability,
                    trade.outcome or False,
                )

                # Record to performance tracker
                self._performance.record_trade(
                    TradeResult(
                        trade_id=trade.signal.signal_id,
                        market_ticker=ticker,
                        strategy_name=trade.signal.strategy_name,
                        direction=trade.direction,
                        entry_price=trade.entry_price * 100,
                        quantity=trade.quantity,
                        pnl=trade.pnl,
                        won=trade.outcome or False,
                        closed_at=trade.exit_time or datetime.now(timezone.utc),
                    )
                )

                del self._open_positions[ticker]

    def _settle_remaining_positions(self) -> None:
        """Settle any remaining open positions at 50% (unknown outcome)."""
        for ticker, trade in list(self._open_positions.items()):
            # Mark as loss (conservative)
            trade.outcome = False
            trade.exit_time = datetime.now(timezone.utc)
            trade.exit_price = 0.5
            trade.pnl = -trade.entry_price * trade.quantity * 0.5

            self._balance += trade.pnl

    def _calculate_results(self, daily_pnl: list[float]) -> BacktestResult:
        """Calculate final backtest results."""
        completed_trades = [t for t in self._trades if t.outcome is not None]
        winning = [t for t in completed_trades if t.outcome]
        losing = [t for t in completed_trades if not t.outcome]

        total_pnl = sum(t.pnl for t in completed_trades)
        roi = total_pnl / self.config.initial_balance if self.config.initial_balance > 0 else 0

        win_rate = len(winning) / len(completed_trades) if completed_trades else 0

        avg_edge = (
            sum(t.signal.edge for t in completed_trades) / len(completed_trades)
            if completed_trades
            else 0
        )

        # Calculate max drawdown
        max_drawdown = 0.0
        if self._peak_balance > 0:
            max_drawdown = (self._peak_balance - self._balance) / self._peak_balance

        # Get Brier score
        brier_result = self._brier.calculate_brier(min_predictions=5)
        brier_score = brier_result.brier_score if brier_result else None

        # Get Sharpe ratio
        sharpe = self._performance.calculate_sharpe_ratio(min_days=5)

        # Strategy breakdown
        strategy_breakdown: dict[str, dict[str, Any]] = {}
        for trade in completed_trades:
            strategy = trade.signal.strategy_name
            if strategy not in strategy_breakdown:
                strategy_breakdown[strategy] = {
                    "trades": 0,
                    "wins": 0,
                    "pnl": 0.0,
                }
            strategy_breakdown[strategy]["trades"] += 1
            if trade.outcome:
                strategy_breakdown[strategy]["wins"] += 1
            strategy_breakdown[strategy]["pnl"] += trade.pnl

        for strategy in strategy_breakdown:
            stats = strategy_breakdown[strategy]
            stats["win_rate"] = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0

        return BacktestResult(
            config=self.config,
            total_trades=len(completed_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=total_pnl,
            roi=roi,
            win_rate=win_rate,
            avg_edge=avg_edge,
            brier_score=brier_score,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            trades=completed_trades,
            daily_returns=daily_pnl,
            strategy_breakdown=strategy_breakdown,
        )


def load_historical_data(path: str) -> list[dict[str, Any]]:
    """Load historical data from JSON file."""
    with open(path) as f:
        return json.load(f)


def generate_sample_data(
    start: datetime,
    end: datetime,
    num_markets: int = 50,
) -> list[dict[str, Any]]:
    """Generate sample historical data for testing."""
    import random

    data = []
    current = start

    while current < end:
        snapshot = {
            "timestamp": current.isoformat(),
            "markets": [],
        }

        for i in range(num_markets):
            # Generate random market data
            yes_price = random.randint(20, 80)
            market = {
                "ticker": f"TEST-MARKET-{i:03d}",
                "yes_bid": yes_price - 2,
                "yes_ask": yes_price + 2,
                "no_bid": 100 - yes_price - 2,
                "no_ask": 100 - yes_price + 2,
                "last_price": yes_price,
                "volume": random.randint(100, 10000),
                "open_interest": random.randint(50, 5000),
            }

            # Randomly settle some markets
            if random.random() < 0.02:
                market["result"] = "yes" if random.random() < 0.5 else "no"

            snapshot["markets"].append(market)

        data.append(snapshot)
        current += timedelta(minutes=15)

    return data


async def main(args: argparse.Namespace) -> int:
    """Main entry point."""
    setup_logging(log_level=args.log_level)

    # Parse dates
    start_date = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_date = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_balance=args.balance,
        strategies=args.strategies.split(","),
        min_edge=args.min_edge,
        kelly_fraction=args.kelly_fraction,
    )

    # Load or generate data
    if args.data:
        logger.info(f"Loading historical data from {args.data}")
        historical_data = load_historical_data(args.data)
    else:
        logger.info("Generating sample data for backtest")
        historical_data = generate_sample_data(start_date, end_date)

    # Run backtest
    engine = BacktestEngine(config)
    result = await engine.run(historical_data)

    # Print results
    result.print_report()

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")

    return 0 if result.total_pnl >= 0 else 1


def cli() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Backtest trading strategies on historical data",
    )

    parser.add_argument(
        "--data",
        help="Path to historical data JSON file",
    )
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2024-03-01",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10000.0,
        help="Initial balance",
    )
    parser.add_argument(
        "--strategies",
        default="weather,calibration",
        help="Comma-separated list of strategies",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.05,
        help="Minimum edge threshold",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Kelly fraction for position sizing",
    )
    parser.add_argument(
        "--output",
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))


if __name__ == "__main__":
    cli()
