"""Runs backtests on signal candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from kalshi_bot.agents.signal_tester.signal_generator import SignalCandidate
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """A single trade in a backtest."""

    ticker: str
    signal_value: float
    entry_price: float
    predicted_direction: int  # 1 for YES, -1 for NO
    actual_outcome: int  # 1 for YES, 0 for NO
    pnl: float
    timestamp: datetime


@dataclass
class BacktestResult:
    """Results from backtesting a signal."""

    signal_id: str
    trades: list[BacktestTrade] = field(default_factory=list)
    start_date: datetime | None = None
    end_date: datetime | None = None

    @property
    def sample_size(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def average_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    @property
    def information_coefficient(self) -> float:
        """Calculate correlation between signal and outcome."""
        if len(self.trades) < 3:
            return 0.0

        signal_values = [t.signal_value * t.predicted_direction for t in self.trades]
        outcomes = [t.actual_outcome for t in self.trades]

        # Pearson correlation
        return float(np.corrcoef(signal_values, outcomes)[0, 1])

    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of returns."""
        if len(self.trades) < 2:
            return 0.0

        returns = [t.pnl for t in self.trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        # Annualized (assuming daily returns)
        return float(mean_return / std_return * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.trades:
            return 0.0

        cumulative = np.cumsum([t.pnl for t in self.trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative

        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "sample_size": self.sample_size,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "average_pnl": self.average_pnl,
            "information_coefficient": self.information_coefficient,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }


class SignalBacktestRunner:
    """
    Runs backtests on signal candidates using historical data.

    Uses settled markets and their snapshots to simulate trading
    based on signal triggers.
    """

    def __init__(
        self,
        db: "Database",
        backtest_days: int = 90,
    ) -> None:
        """
        Initialize the backtest runner.

        Args:
            db: Database connection
            backtest_days: Number of days to look back
        """
        self._db = db
        self._backtest_days = backtest_days

    async def run_backtest(
        self,
        signal: SignalCandidate,
        min_samples: int = 20,
    ) -> BacktestResult:
        """
        Run a backtest on a signal candidate.

        Args:
            signal: Signal to backtest
            min_samples: Minimum samples required for valid backtest

        Returns:
            BacktestResult with trade history and metrics
        """
        result = BacktestResult(signal_id=signal.signal_id)

        # Get settled markets with their snapshots
        cutoff_date = datetime.utcnow() - timedelta(days=self._backtest_days)

        settlements = await self._db.fetch_all(
            """
            SELECT ticker, outcome, confirmed_at
            FROM market_settlements
            WHERE confirmed_at >= ?
            ORDER BY confirmed_at DESC
            """,
            (cutoff_date.isoformat(),),
        )

        if not settlements:
            logger.debug(f"No settlements found for backtest of {signal.signal_id}")
            return result

        result.start_date = cutoff_date
        result.end_date = datetime.utcnow()

        for settlement in settlements:
            ticker = settlement["ticker"]
            outcome = 1 if settlement["outcome"] == "yes" else 0

            # Get snapshots before settlement
            snapshots = await self._db.fetch_all(
                """
                SELECT * FROM market_snapshots
                WHERE ticker = ?
                  AND snapshot_at < ?
                ORDER BY snapshot_at DESC
                LIMIT 10
                """,
                (ticker, settlement["confirmed_at"]),
            )

            if not snapshots:
                continue

            # Use most recent snapshot
            snapshot = snapshots[0]

            # Compute features from snapshot
            features = self._extract_features(snapshot)
            if not features:
                continue

            # Compute signal value
            try:
                from kalshi_bot.agents.signal_tester.signal_generator import SignalGenerator
                generator = SignalGenerator()
                signal_value = generator.compute_signal_value(signal, features)

                if signal_value is None:
                    continue

                # Check if signal is triggered
                if not (signal.min_threshold <= signal_value <= signal.max_threshold):
                    continue

                # Determine predicted direction
                if signal.expected_direction != 0:
                    predicted = signal.expected_direction
                else:
                    # For neutral signals, use signal value direction
                    predicted = 1 if signal_value > 0.5 else -1

                # Calculate P&L
                entry_price = features.get("current_price", 0.5)
                if predicted == 1:  # Bet YES
                    pnl = (1.0 - entry_price) if outcome == 1 else -entry_price
                else:  # Bet NO
                    pnl = entry_price if outcome == 0 else -(1.0 - entry_price)

                trade = BacktestTrade(
                    ticker=ticker,
                    signal_value=signal_value,
                    entry_price=entry_price,
                    predicted_direction=predicted,
                    actual_outcome=outcome,
                    pnl=pnl,
                    timestamp=datetime.fromisoformat(snapshot["snapshot_at"]),
                )
                result.trades.append(trade)

            except Exception as e:
                logger.debug(f"Error processing {ticker} for signal {signal.signal_id}: {e}")
                continue

        logger.info(
            f"Backtest {signal.signal_id}: {len(result.trades)} trades, "
            f"win rate {result.win_rate:.1%}, IC {result.information_coefficient:.3f}"
        )

        return result

    def _extract_features(self, snapshot: dict) -> dict[str, float]:
        """Extract features from a snapshot for signal computation."""
        try:
            # Get price
            price = snapshot.get("last_price")
            if price is None:
                yes_bid = snapshot.get("yes_bid")
                yes_ask = snapshot.get("yes_ask")
                if yes_bid is not None and yes_ask is not None:
                    price = (yes_bid + yes_ask) / 2
                else:
                    price = 50

            # Normalize price to 0-1
            current_price = price / 100.0

            # Get spread
            spread = snapshot.get("spread", 5)
            spread_normalized = min(1.0, spread / 20.0)

            # Get volume (log scale)
            volume = np.log1p(snapshot.get("volume", 0))
            open_interest = np.log1p(snapshot.get("open_interest", 0))

            # Time features
            snapshot_time = datetime.fromisoformat(snapshot["snapshot_at"])
            hour_of_day = snapshot_time.hour

            # Calculate hours to expiry if available
            hours_to_expiry = 0.5  # Default
            exp_time = snapshot.get("expiration_time") or snapshot.get("close_time")
            if exp_time:
                try:
                    exp_dt = datetime.fromisoformat(exp_time.replace("Z", "+00:00")).replace(tzinfo=None)
                    hours_to_expiry = max(0, (exp_dt - snapshot_time).total_seconds() / 3600)
                    hours_to_expiry = min(hours_to_expiry, 720) / 720.0
                except (ValueError, TypeError):
                    pass

            return {
                "current_price": current_price,
                "spread": spread_normalized,
                "volume": volume,
                "open_interest": open_interest,
                "hour_of_day": hour_of_day / 23.0,
                "hours_to_expiry": hours_to_expiry,
                "price_momentum_1h": 0.0,  # Would need historical data
                "price_momentum_6h": 0.0,
                "price_momentum_24h": 0.0,
                "price_volatility": 0.1,  # Default
                "volume_momentum": 0.0,
                # Weather features (default to 0 for non-weather)
                "nws_probability": 0.0,
                "nws_confidence": 0.0,
                "nws_temp_forecast": 0.0,
                "forecast_hours_out": 0.0,
                "forecast_recency": 0.0,
            }
        except Exception as e:
            logger.debug(f"Error extracting features: {e}")
            return {}

    async def run_all_backtests(
        self,
        signals: list[SignalCandidate],
    ) -> list[BacktestResult]:
        """Run backtests on multiple signals."""
        results = []
        for signal in signals:
            result = await self.run_backtest(signal)
            results.append(result)
        return results
