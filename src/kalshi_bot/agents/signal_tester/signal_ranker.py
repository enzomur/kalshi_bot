"""Ranks signal candidates by predictive power."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from kalshi_bot.agents.signal_tester.backtest_runner import BacktestResult
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class SignalRecommendation(str, Enum):
    """Recommendation for a signal based on backtest results."""

    APPROVE = "approve"        # Signal shows strong predictive power
    REJECT = "reject"          # Signal shows no/negative predictive power
    NEEDS_MORE_DATA = "needs_more_data"  # Not enough samples
    REVIEW = "review"          # Borderline, needs human review


@dataclass
class RankedSignal:
    """A signal with its ranking score and recommendation."""

    signal_id: str
    rank_score: float  # Combined score (higher is better)
    recommendation: SignalRecommendation
    win_rate: float
    information_coefficient: float
    sharpe_ratio: float
    p_value: float
    sample_size: int
    notes: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "rank_score": self.rank_score,
            "recommendation": self.recommendation.value,
            "win_rate": self.win_rate,
            "information_coefficient": self.information_coefficient,
            "sharpe_ratio": self.sharpe_ratio,
            "p_value": self.p_value,
            "sample_size": self.sample_size,
            "notes": self.notes,
        }


class SignalRanker:
    """
    Ranks signal candidates by their predictive power.

    Uses multiple metrics to evaluate signals:
    - Win rate: Basic accuracy
    - Information Coefficient: Correlation with outcomes
    - Sharpe Ratio: Risk-adjusted returns
    - Statistical significance: p-value from hypothesis test
    """

    def __init__(
        self,
        min_samples: int = 30,
        required_win_rate: float = 0.55,
        required_ic: float = 0.10,
        significance_level: float = 0.05,
    ) -> None:
        """
        Initialize the ranker.

        Args:
            min_samples: Minimum samples for valid ranking
            required_win_rate: Minimum win rate to approve
            required_ic: Minimum information coefficient to approve
            significance_level: P-value threshold for significance
        """
        self._min_samples = min_samples
        self._required_win_rate = required_win_rate
        self._required_ic = required_ic
        self._significance_level = significance_level

    def rank_signal(self, result: BacktestResult) -> RankedSignal:
        """
        Rank a single signal based on backtest results.

        Args:
            result: Backtest result

        Returns:
            RankedSignal with score and recommendation
        """
        # Check sample size
        if result.sample_size < self._min_samples:
            return RankedSignal(
                signal_id=result.signal_id,
                rank_score=0.0,
                recommendation=SignalRecommendation.NEEDS_MORE_DATA,
                win_rate=result.win_rate,
                information_coefficient=result.information_coefficient,
                sharpe_ratio=result.sharpe_ratio,
                p_value=1.0,
                sample_size=result.sample_size,
                notes=f"Only {result.sample_size} samples, need {self._min_samples}",
            )

        # Calculate p-value using binomial test
        wins = int(result.win_rate * result.sample_size)
        p_value = self._calculate_p_value(wins, result.sample_size)

        # Calculate composite rank score
        # Weights: win_rate (30%), IC (30%), Sharpe (20%), significance (20%)
        win_rate_score = max(0, (result.win_rate - 0.50) * 2)  # 0-1 scale
        ic_score = max(0, min(1, result.information_coefficient))
        sharpe_score = max(0, min(1, result.sharpe_ratio / 2))  # Normalize
        sig_score = 1.0 - p_value

        rank_score = (
            0.30 * win_rate_score +
            0.30 * ic_score +
            0.20 * sharpe_score +
            0.20 * sig_score
        )

        # Determine recommendation
        recommendation = self._get_recommendation(
            result.win_rate,
            result.information_coefficient,
            p_value,
        )

        notes = self._generate_notes(result, p_value, recommendation)

        return RankedSignal(
            signal_id=result.signal_id,
            rank_score=rank_score,
            recommendation=recommendation,
            win_rate=result.win_rate,
            information_coefficient=result.information_coefficient,
            sharpe_ratio=result.sharpe_ratio,
            p_value=p_value,
            sample_size=result.sample_size,
            notes=notes,
        )

    def rank_signals(
        self,
        results: list[BacktestResult],
    ) -> list[RankedSignal]:
        """
        Rank multiple signals.

        Args:
            results: List of backtest results

        Returns:
            List of RankedSignal, sorted by rank_score descending
        """
        ranked = [self.rank_signal(r) for r in results]
        ranked.sort(key=lambda x: x.rank_score, reverse=True)
        return ranked

    def get_approved_signals(
        self,
        ranked: list[RankedSignal],
    ) -> list[RankedSignal]:
        """Get only approved signals."""
        return [s for s in ranked if s.recommendation == SignalRecommendation.APPROVE]

    def _calculate_p_value(self, wins: int, total: int) -> float:
        """Calculate p-value using one-sided binomial test."""
        if total == 0:
            return 1.0

        # Test against null hypothesis of 50% win rate
        try:
            result = stats.binomtest(wins, total, p=0.5, alternative="greater")
            return float(result.pvalue)
        except Exception:
            # Fallback to normal approximation
            expected = total * 0.5
            std = np.sqrt(total * 0.5 * 0.5)
            if std == 0:
                return 1.0
            z = (wins - expected) / std
            return float(1 - stats.norm.cdf(z))

    def _get_recommendation(
        self,
        win_rate: float,
        ic: float,
        p_value: float,
    ) -> SignalRecommendation:
        """Determine recommendation based on metrics."""
        # Must be statistically significant
        if p_value > self._significance_level:
            return SignalRecommendation.REVIEW

        # Must meet both win rate and IC thresholds
        if win_rate >= self._required_win_rate and ic >= self._required_ic:
            return SignalRecommendation.APPROVE

        # Reject if clearly not predictive
        if win_rate < 0.50 or ic < 0:
            return SignalRecommendation.REJECT

        # Borderline cases need review
        return SignalRecommendation.REVIEW

    def _generate_notes(
        self,
        result: BacktestResult,
        p_value: float,
        recommendation: SignalRecommendation,
    ) -> str:
        """Generate human-readable notes about the signal."""
        notes = []

        if recommendation == SignalRecommendation.APPROVE:
            notes.append("Strong predictive power confirmed.")
        elif recommendation == SignalRecommendation.REJECT:
            notes.append("No significant predictive power.")

        if result.win_rate >= 0.60:
            notes.append(f"High win rate ({result.win_rate:.1%}).")
        elif result.win_rate < 0.50:
            notes.append(f"Below-chance win rate ({result.win_rate:.1%}).")

        if result.information_coefficient > 0.20:
            notes.append(f"Strong IC ({result.information_coefficient:.3f}).")
        elif result.information_coefficient < 0:
            notes.append(f"Negative IC ({result.information_coefficient:.3f}).")

        if p_value < 0.01:
            notes.append("Highly significant (p < 0.01).")
        elif p_value > 0.10:
            notes.append(f"Not significant (p = {p_value:.3f}).")

        if result.max_drawdown > 0.20:
            notes.append(f"High drawdown risk ({result.max_drawdown:.1%}).")

        return " ".join(notes)
