"""Backtest performance metrics calculation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from kalshi_bot.backtesting.position_tracker import PositionTracker, SettlementResult
from kalshi_bot.backtesting.simulator import TradeSimulator
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EquityPoint:
    """A point on the equity curve."""

    timestamp: datetime
    equity: float
    drawdown: float
    drawdown_pct: float


@dataclass
class BacktestMetrics:
    """
    Comprehensive backtest performance metrics.

    Calculates standard trading metrics:
    - Returns: total, annualized, per-trade
    - Risk: Sharpe ratio, max drawdown, volatility
    - Win/loss: win rate, profit factor, average win/loss
    - Efficiency: number of trades, holding periods
    """

    # Basic results
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # Win/Loss metrics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: timedelta = field(default_factory=lambda: timedelta(0))
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    # Time metrics
    start_date: datetime | None = None
    end_date: datetime | None = None
    duration_days: int = 0
    avg_holding_period: timedelta = field(default_factory=lambda: timedelta(0))

    # Fee analysis
    total_fees: float = 0.0
    fees_pct_of_pnl: float = 0.0

    # Strategy breakdown
    strategy_pnl: dict[str, float] = field(default_factory=dict)
    strategy_trades: dict[str, int] = field(default_factory=dict)

    # Equity curve
    equity_curve: list[EquityPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "volatility": self.volatility,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "duration_days": self.duration_days,
            "total_fees": self.total_fees,
            "fees_pct_of_pnl": self.fees_pct_of_pnl,
            "strategy_pnl": self.strategy_pnl,
            "strategy_trades": self.strategy_trades,
        }


def calculate_metrics(
    simulator: TradeSimulator,
    position_tracker: PositionTracker,
    equity_snapshots: list[tuple[datetime, float]] | None = None,
    risk_free_rate: float = 0.05,
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics.

    Args:
        simulator: The trade simulator with execution data
        position_tracker: The position tracker with settlements
        equity_snapshots: Optional list of (timestamp, equity) for equity curve
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        BacktestMetrics with all calculated values
    """
    metrics = BacktestMetrics()

    # Basic balances
    metrics.initial_balance = simulator.initial_balance
    metrics.final_balance = simulator.balance + position_tracker.total_exposure
    metrics.total_pnl = position_tracker.get_total_pnl()
    metrics.total_fees = simulator.total_fees

    if metrics.initial_balance > 0:
        metrics.total_return_pct = (metrics.final_balance - metrics.initial_balance) / metrics.initial_balance

    # Fee analysis
    if abs(metrics.total_pnl) > 0.01:
        metrics.fees_pct_of_pnl = metrics.total_fees / abs(metrics.total_pnl)

    # Trade statistics
    settlements = position_tracker.settlements
    metrics.total_trades = len(settlements)

    if metrics.total_trades == 0:
        return metrics

    wins = [s for s in settlements if s.profit > 0]
    losses = [s for s in settlements if s.profit < 0]
    breakevens = [s for s in settlements if s.profit == 0]

    metrics.winning_trades = len(wins)
    metrics.losing_trades = len(losses)
    metrics.breakeven_trades = len(breakevens)

    # Win rate
    metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0.0

    # Average win/loss
    if wins:
        metrics.avg_win = sum(s.profit for s in wins) / len(wins)
        metrics.largest_win = max(s.profit for s in wins)

    if losses:
        metrics.avg_loss = sum(s.profit for s in losses) / len(losses)
        metrics.largest_loss = min(s.profit for s in losses)

    # Profit factor
    gross_profit = sum(s.profit for s in wins)
    gross_loss = abs(sum(s.profit for s in losses))
    if gross_loss > 0:
        metrics.profit_factor = gross_profit / gross_loss

    # Expectancy (average profit per trade)
    metrics.expectancy = metrics.total_pnl / metrics.total_trades if metrics.total_trades > 0 else 0.0

    # Time metrics
    if settlements:
        metrics.start_date = min(s.settled_at for s in settlements)
        metrics.end_date = max(s.settled_at for s in settlements)
        metrics.duration_days = (metrics.end_date - metrics.start_date).days

    # Calculate drawdown and risk metrics from equity snapshots
    if equity_snapshots and len(equity_snapshots) > 1:
        _calculate_equity_metrics(metrics, equity_snapshots, risk_free_rate)

    # Strategy breakdown
    _calculate_strategy_breakdown(metrics, settlements, simulator.trades)

    return metrics


def _calculate_equity_metrics(
    metrics: BacktestMetrics,
    equity_snapshots: list[tuple[datetime, float]],
    risk_free_rate: float,
) -> None:
    """Calculate equity curve metrics (drawdown, Sharpe, etc.)."""
    if len(equity_snapshots) < 2:
        return

    # Sort by timestamp
    sorted_snapshots = sorted(equity_snapshots, key=lambda x: x[0])

    # Build equity curve
    peak = sorted_snapshots[0][1]
    max_dd = 0.0
    max_dd_pct = 0.0
    dd_start: datetime | None = None
    max_dd_duration = timedelta(0)
    current_dd_duration = timedelta(0)

    equity_curve: list[EquityPoint] = []
    returns: list[float] = []

    prev_equity = sorted_snapshots[0][1]
    for timestamp, equity in sorted_snapshots:
        # Track returns
        if prev_equity > 0:
            ret = (equity - prev_equity) / prev_equity
            returns.append(ret)
        prev_equity = equity

        # Track drawdown
        if equity > peak:
            peak = equity
            dd_start = None
            current_dd_duration = timedelta(0)
        else:
            if dd_start is None:
                dd_start = timestamp
            current_dd_duration = timestamp - dd_start

        dd = peak - equity
        dd_pct = dd / peak if peak > 0 else 0.0

        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct

        if current_dd_duration > max_dd_duration:
            max_dd_duration = current_dd_duration

        equity_curve.append(EquityPoint(
            timestamp=timestamp,
            equity=equity,
            drawdown=dd,
            drawdown_pct=dd_pct,
        ))

    metrics.equity_curve = equity_curve
    metrics.max_drawdown = max_dd
    metrics.max_drawdown_pct = max_dd_pct
    metrics.max_drawdown_duration = max_dd_duration

    # Calculate risk metrics from returns
    if len(returns) > 1:
        # Volatility (annualized, assuming daily returns)
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance)
        metrics.volatility = std_dev * math.sqrt(252)  # Annualize

        # Sharpe ratio
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_returns = [r - daily_rf for r in returns]
        mean_excess = sum(excess_returns) / len(excess_returns)

        if std_dev > 0:
            metrics.sharpe_ratio = (mean_excess / std_dev) * math.sqrt(252)

        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_variance = sum(r ** 2 for r in negative_returns) / len(returns)
            downside_dev = math.sqrt(downside_variance)
            if downside_dev > 0:
                metrics.sortino_ratio = (mean_excess / downside_dev) * math.sqrt(252)

        # Calmar ratio
        if metrics.max_drawdown_pct > 0 and metrics.duration_days > 0:
            annualized_return = metrics.total_return_pct * (365 / metrics.duration_days)
            metrics.calmar_ratio = annualized_return / metrics.max_drawdown_pct


def _calculate_strategy_breakdown(
    metrics: BacktestMetrics,
    settlements: list[SettlementResult],
    trades: list,
) -> None:
    """Calculate per-strategy metrics."""
    # Group settlements by opportunity, then map to strategy
    opportunity_strategies: dict[str, str] = {}
    for trade in trades:
        opportunity_strategies[trade.opportunity_id] = trade.strategy

    strategy_pnl: dict[str, float] = {}
    strategy_trades: dict[str, int] = {}

    for settlement in settlements:
        strategy = opportunity_strategies.get(settlement.opportunity_id, "unknown")

        if strategy not in strategy_pnl:
            strategy_pnl[strategy] = 0.0
            strategy_trades[strategy] = 0

        strategy_pnl[strategy] += settlement.profit
        strategy_trades[strategy] += 1

    metrics.strategy_pnl = strategy_pnl
    metrics.strategy_trades = strategy_trades


def format_metrics_summary(metrics: BacktestMetrics) -> str:
    """Format metrics as a human-readable summary."""
    lines = [
        "=" * 60,
        "BACKTEST RESULTS",
        "=" * 60,
        "",
        "Performance",
        "-" * 40,
        f"  Initial Balance:    ${metrics.initial_balance:,.2f}",
        f"  Final Balance:      ${metrics.final_balance:,.2f}",
        f"  Total P&L:          ${metrics.total_pnl:,.2f}",
        f"  Total Return:       {metrics.total_return_pct:.2%}",
        f"  Total Fees:         ${metrics.total_fees:,.2f}",
        "",
        "Trade Statistics",
        "-" * 40,
        f"  Total Trades:       {metrics.total_trades}",
        f"  Winning Trades:     {metrics.winning_trades}",
        f"  Losing Trades:      {metrics.losing_trades}",
        f"  Win Rate:           {metrics.win_rate:.2%}",
        f"  Avg Win:            ${metrics.avg_win:,.2f}",
        f"  Avg Loss:           ${metrics.avg_loss:,.2f}",
        f"  Largest Win:        ${metrics.largest_win:,.2f}",
        f"  Largest Loss:       ${metrics.largest_loss:,.2f}",
        f"  Profit Factor:      {metrics.profit_factor:.2f}",
        f"  Expectancy:         ${metrics.expectancy:,.2f}",
        "",
        "Risk Metrics",
        "-" * 40,
        f"  Max Drawdown:       ${metrics.max_drawdown:,.2f} ({metrics.max_drawdown_pct:.2%})",
        f"  Sharpe Ratio:       {metrics.sharpe_ratio:.2f}",
        f"  Sortino Ratio:      {metrics.sortino_ratio:.2f}",
        f"  Calmar Ratio:       {metrics.calmar_ratio:.2f}",
        f"  Volatility (Ann):   {metrics.volatility:.2%}",
        "",
    ]

    if metrics.strategy_pnl:
        lines.extend([
            "Strategy Breakdown",
            "-" * 40,
        ])
        for strategy, pnl in sorted(metrics.strategy_pnl.items()):
            trades_count = metrics.strategy_trades.get(strategy, 0)
            lines.append(f"  {strategy}: ${pnl:,.2f} ({trades_count} trades)")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
