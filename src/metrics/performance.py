"""Performance tracking: win rate, Sharpe ratio, drawdown, ROI.

This module tracks trading performance metrics over time. It integrates with
the database to persist metrics and provides real-time performance summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradeResult:
    """Record of a completed trade."""

    trade_id: str
    market_ticker: str
    strategy_name: str
    direction: str  # "yes" or "no"
    entry_price: float  # In cents (0-100)
    quantity: int
    pnl: float  # Realized P&L in dollars
    won: bool
    closed_at: datetime
    hold_time_hours: float = 0.0

    @property
    def return_pct(self) -> float:
        """Calculate percentage return on the trade."""
        cost = (self.entry_price / 100) * self.quantity
        if cost == 0:
            return 0.0
        return self.pnl / cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "market_ticker": self.market_ticker,
            "strategy_name": self.strategy_name,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "won": self.won,
            "closed_at": self.closed_at.isoformat(),
            "hold_time_hours": self.hold_time_hours,
            "return_pct": self.return_pct,
        }


@dataclass
class DailyMetrics:
    """Daily performance snapshot."""

    date: datetime
    trades: int
    wins: int
    losses: int
    pnl: float
    starting_balance: float
    ending_balance: float

    @property
    def win_rate(self) -> float:
        """Daily win rate."""
        if self.trades == 0:
            return 0.0
        return self.wins / self.trades

    @property
    def daily_return(self) -> float:
        """Daily return percentage."""
        if self.starting_balance == 0:
            return 0.0
        return (self.ending_balance - self.starting_balance) / self.starting_balance

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "pnl": self.pnl,
            "win_rate": self.win_rate,
            "daily_return": self.daily_return,
        }


@dataclass
class PerformanceSummary:
    """Comprehensive performance summary."""

    # Overall metrics
    total_trades: int
    total_wins: int
    total_losses: int
    win_rate: float

    # P&L metrics
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    roi: float  # Return on initial investment

    # Risk metrics
    sharpe_ratio: float | None
    max_drawdown: float
    max_drawdown_dollars: float
    current_drawdown: float

    # Balance tracking
    initial_balance: float
    current_balance: float
    peak_balance: float

    # Per-strategy breakdown
    strategy_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Edge tracking
    avg_edge: float = 0.0
    edge_realization_rate: float = 0.0  # How much of expected edge we actually capture

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "roi": self.roi,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_dollars": self.max_drawdown_dollars,
            "current_drawdown": self.current_drawdown,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "strategy_metrics": self.strategy_metrics,
            "avg_edge": self.avg_edge,
            "edge_realization_rate": self.edge_realization_rate,
        }


class PerformanceTracker:
    """Tracks and calculates trading performance metrics.

    This tracker maintains a history of trades and calculates key performance
    indicators like Sharpe ratio, drawdown, win rate, and ROI.
    """

    # Annualization factor (assuming 365 trading days)
    TRADING_DAYS_PER_YEAR = 365

    def __init__(
        self,
        initial_balance: float = 1000.0,
        risk_free_rate: float = 0.05,  # 5% annual risk-free rate
    ) -> None:
        """
        Initialize the performance tracker.

        Args:
            initial_balance: Starting balance for ROI calculations
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self._initial_balance = initial_balance
        self._current_balance = initial_balance
        self._peak_balance = initial_balance
        self._risk_free_rate = risk_free_rate

        self._trades: list[TradeResult] = []
        self._daily_metrics: list[DailyMetrics] = []
        self._daily_returns: list[float] = []

        # Running totals
        self._total_pnl = 0.0
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._max_drawdown = 0.0
        self._max_drawdown_dollars = 0.0

        # Edge tracking
        self._expected_edges: list[float] = []
        self._realized_edges: list[float] = []

    def record_trade(self, trade: TradeResult) -> None:
        """
        Record a completed trade.

        Args:
            trade: TradeResult to record
        """
        self._trades.append(trade)
        self._realized_pnl += trade.pnl
        self._total_pnl = self._realized_pnl + self._unrealized_pnl

        # Update balance
        self._current_balance += trade.pnl

        # Update peak and drawdown
        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance

        if self._peak_balance > 0:
            current_dd = (self._peak_balance - self._current_balance) / self._peak_balance
            current_dd_dollars = self._peak_balance - self._current_balance
            if current_dd > self._max_drawdown:
                self._max_drawdown = current_dd
                self._max_drawdown_dollars = current_dd_dollars

        logger.debug(
            f"Recorded trade {trade.trade_id[:8]}: "
            f"{'WIN' if trade.won else 'LOSS'} ${trade.pnl:.2f}"
        )

    def record_edge(self, expected_edge: float, realized_edge: float) -> None:
        """
        Record expected vs realized edge for tracking edge capture rate.

        Args:
            expected_edge: The edge we expected from our model
            realized_edge: The actual edge we captured
        """
        self._expected_edges.append(expected_edge)
        self._realized_edges.append(realized_edge)

    def update_unrealized_pnl(self, unrealized: float) -> None:
        """Update unrealized P&L from open positions."""
        self._unrealized_pnl = unrealized
        self._total_pnl = self._realized_pnl + self._unrealized_pnl

    def record_daily_snapshot(
        self,
        date: datetime | None = None,
        ending_balance: float | None = None,
    ) -> DailyMetrics:
        """
        Record daily performance snapshot.

        Call this at end of each trading day to track daily metrics.

        Args:
            date: Date for the snapshot (default: today)
            ending_balance: Ending balance (default: current balance)

        Returns:
            DailyMetrics for the day
        """
        if date is None:
            date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if ending_balance is None:
            ending_balance = self._current_balance

        # Get starting balance from previous snapshot or initial
        if self._daily_metrics:
            starting_balance = self._daily_metrics[-1].ending_balance
        else:
            starting_balance = self._initial_balance

        # Count today's trades
        today_start = date
        today_end = date + timedelta(days=1)
        today_trades = [
            t for t in self._trades
            if today_start <= t.closed_at < today_end
        ]

        daily = DailyMetrics(
            date=date,
            trades=len(today_trades),
            wins=sum(1 for t in today_trades if t.won),
            losses=sum(1 for t in today_trades if not t.won),
            pnl=sum(t.pnl for t in today_trades),
            starting_balance=starting_balance,
            ending_balance=ending_balance,
        )

        self._daily_metrics.append(daily)

        if starting_balance > 0:
            daily_return = (ending_balance - starting_balance) / starting_balance
            self._daily_returns.append(daily_return)

        return daily

    def calculate_sharpe_ratio(self, min_days: int = 30) -> float | None:
        """
        Calculate Sharpe ratio from daily returns.

        Args:
            min_days: Minimum days of data required

        Returns:
            Annualized Sharpe ratio, or None if insufficient data
        """
        if len(self._daily_returns) < min_days:
            return None

        # Calculate mean and std of daily returns
        mean_return = sum(self._daily_returns) / len(self._daily_returns)
        variance = sum(
            (r - mean_return) ** 2 for r in self._daily_returns
        ) / len(self._daily_returns)
        std_return = math.sqrt(variance) if variance > 0 else 0.0

        if std_return == 0:
            return None

        # Daily risk-free rate
        daily_rf = self._risk_free_rate / self.TRADING_DAYS_PER_YEAR

        # Daily Sharpe
        daily_sharpe = (mean_return - daily_rf) / std_return

        # Annualize
        annualized_sharpe = daily_sharpe * math.sqrt(self.TRADING_DAYS_PER_YEAR)

        return annualized_sharpe

    def calculate_edge_realization(self) -> float:
        """
        Calculate what percentage of expected edge we actually captured.

        Returns:
            Edge realization rate (1.0 = captured 100% of expected edge)
        """
        if not self._expected_edges or sum(self._expected_edges) == 0:
            return 0.0

        total_expected = sum(self._expected_edges)
        total_realized = sum(self._realized_edges)

        return total_realized / total_expected

    def get_summary(self) -> PerformanceSummary:
        """
        Generate comprehensive performance summary.

        Returns:
            PerformanceSummary with all metrics
        """
        total_trades = len(self._trades)
        total_wins = sum(1 for t in self._trades if t.won)
        total_losses = total_trades - total_wins
        win_rate = total_wins / total_trades if total_trades > 0 else 0.0

        roi = (
            (self._current_balance - self._initial_balance) / self._initial_balance
            if self._initial_balance > 0
            else 0.0
        )

        current_drawdown = (
            (self._peak_balance - self._current_balance) / self._peak_balance
            if self._peak_balance > 0
            else 0.0
        )

        # Calculate per-strategy metrics
        strategy_metrics: dict[str, dict[str, Any]] = {}
        strategies = set(t.strategy_name for t in self._trades)
        for strategy in strategies:
            strat_trades = [t for t in self._trades if t.strategy_name == strategy]
            strat_wins = sum(1 for t in strat_trades if t.won)
            strat_pnl = sum(t.pnl for t in strat_trades)
            strategy_metrics[strategy] = {
                "trades": len(strat_trades),
                "wins": strat_wins,
                "losses": len(strat_trades) - strat_wins,
                "win_rate": strat_wins / len(strat_trades) if strat_trades else 0.0,
                "pnl": strat_pnl,
                "avg_return": sum(t.return_pct for t in strat_trades) / len(strat_trades) if strat_trades else 0.0,
            }

        # Average edge
        avg_edge = (
            sum(self._expected_edges) / len(self._expected_edges)
            if self._expected_edges
            else 0.0
        )

        return PerformanceSummary(
            total_trades=total_trades,
            total_wins=total_wins,
            total_losses=total_losses,
            win_rate=win_rate,
            total_pnl=self._total_pnl,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=self._unrealized_pnl,
            roi=roi,
            sharpe_ratio=self.calculate_sharpe_ratio(),
            max_drawdown=self._max_drawdown,
            max_drawdown_dollars=self._max_drawdown_dollars,
            current_drawdown=current_drawdown,
            initial_balance=self._initial_balance,
            current_balance=self._current_balance,
            peak_balance=self._peak_balance,
            strategy_metrics=strategy_metrics,
            avg_edge=avg_edge,
            edge_realization_rate=self.calculate_edge_realization(),
        )

    def get_strategy_summary(self, strategy_name: str) -> dict[str, Any]:
        """
        Get performance summary for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary with strategy-specific metrics
        """
        strat_trades = [t for t in self._trades if t.strategy_name == strategy_name]

        if not strat_trades:
            return {
                "strategy": strategy_name,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "pnl": 0.0,
            }

        wins = sum(1 for t in strat_trades if t.won)
        losses = len(strat_trades) - wins
        pnl = sum(t.pnl for t in strat_trades)
        avg_return = sum(t.return_pct for t in strat_trades) / len(strat_trades)

        return {
            "strategy": strategy_name,
            "trades": len(strat_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(strat_trades),
            "pnl": pnl,
            "avg_return": avg_return,
            "avg_hold_time_hours": sum(t.hold_time_hours for t in strat_trades) / len(strat_trades),
        }

    def get_daily_metrics(self, days: int = 30) -> list[DailyMetrics]:
        """Get recent daily metrics."""
        return self._daily_metrics[-days:]

    def get_status(self) -> dict[str, Any]:
        """Get current tracker status."""
        summary = self.get_summary()
        return {
            "trades": summary.total_trades,
            "win_rate": summary.win_rate,
            "pnl": summary.total_pnl,
            "roi": summary.roi,
            "sharpe": summary.sharpe_ratio,
            "max_drawdown": summary.max_drawdown,
            "current_drawdown": summary.current_drawdown,
        }
