"""Brier score calculation for calibration measurement.

The Brier score measures the accuracy of probabilistic predictions.
Lower is better: 0 = perfect, 0.25 = random guessing (on 50/50), 1 = worst.

Formula: Brier = (1/N) * Σ(forecast_i - outcome_i)²

For trading, good calibration (Brier < 0.22) indicates our probability
estimates are reliable for sizing and edge calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionRecord:
    """A single prediction and its outcome."""

    prediction_id: str
    market_ticker: str
    strategy_name: str
    predicted_probability: float  # Our estimate (0-1)
    direction: str  # "yes" or "no"
    actual_outcome: bool | None = None  # True if we would have won
    settled_at: datetime | None = None

    @property
    def is_resolved(self) -> bool:
        """Check if this prediction has been resolved."""
        return self.actual_outcome is not None

    def brier_contribution(self) -> float | None:
        """
        Calculate this prediction's contribution to Brier score.

        Returns None if not yet resolved.
        """
        if self.actual_outcome is None:
            return None

        outcome = 1.0 if self.actual_outcome else 0.0
        return (self.predicted_probability - outcome) ** 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "market_ticker": self.market_ticker,
            "strategy_name": self.strategy_name,
            "predicted_probability": self.predicted_probability,
            "direction": self.direction,
            "actual_outcome": self.actual_outcome,
            "settled_at": self.settled_at.isoformat() if self.settled_at else None,
        }


@dataclass
class BrierResult:
    """Result of Brier score calculation."""

    brier_score: float
    n_predictions: int
    n_correct: int  # Predictions where outcome matched majority direction
    win_rate: float
    mean_probability: float
    is_calibrated: bool  # True if Brier < threshold

    # Per-strategy breakdown
    strategy_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "brier_score": self.brier_score,
            "n_predictions": self.n_predictions,
            "n_correct": self.n_correct,
            "win_rate": self.win_rate,
            "mean_probability": self.mean_probability,
            "is_calibrated": self.is_calibrated,
            "strategy_scores": self.strategy_scores,
        }


class BrierCalculator:
    """Calculates and tracks Brier scores for calibration monitoring.

    The Brier score is the mean squared error between predicted probabilities
    and actual outcomes. It measures both calibration and resolution.

    For trading:
    - Brier < 0.20: Excellent calibration
    - Brier < 0.22: Good calibration (our target)
    - Brier 0.22-0.25: Fair calibration
    - Brier > 0.25: Poor calibration (worse than random on 50/50)
    """

    DEFAULT_CALIBRATION_THRESHOLD = 0.22

    def __init__(
        self,
        calibration_threshold: float = DEFAULT_CALIBRATION_THRESHOLD,
    ) -> None:
        """
        Initialize the Brier calculator.

        Args:
            calibration_threshold: Brier score below which we consider calibrated
        """
        self._calibration_threshold = calibration_threshold
        self._predictions: list[PredictionRecord] = []

    def add_prediction(self, record: PredictionRecord) -> None:
        """Add a prediction record for tracking."""
        self._predictions.append(record)

    def record_prediction(
        self,
        prediction_id: str,
        market_ticker: str,
        strategy_name: str,
        predicted_probability: float,
        direction: str,
    ) -> PredictionRecord:
        """
        Record a new prediction.

        Args:
            prediction_id: Unique ID for this prediction
            market_ticker: Market being predicted
            strategy_name: Strategy that made the prediction
            predicted_probability: Our probability estimate (0-1)
            direction: "yes" or "no"

        Returns:
            The created PredictionRecord
        """
        record = PredictionRecord(
            prediction_id=prediction_id,
            market_ticker=market_ticker,
            strategy_name=strategy_name,
            predicted_probability=predicted_probability,
            direction=direction,
        )
        self._predictions.append(record)
        return record

    def resolve_prediction(
        self,
        prediction_id: str,
        actual_outcome: bool,
        settled_at: datetime | None = None,
    ) -> PredictionRecord | None:
        """
        Record the actual outcome for a prediction.

        Args:
            prediction_id: ID of the prediction to resolve
            actual_outcome: True if our direction was correct
            settled_at: When the market settled

        Returns:
            Updated PredictionRecord, or None if not found
        """
        for record in self._predictions:
            if record.prediction_id == prediction_id:
                record.actual_outcome = actual_outcome
                record.settled_at = settled_at or datetime.now(timezone.utc)
                return record
        return None

    def calculate_brier(
        self,
        strategy_name: str | None = None,
        min_predictions: int = 10,
    ) -> BrierResult | None:
        """
        Calculate Brier score from resolved predictions.

        Args:
            strategy_name: If provided, calculate only for this strategy
            min_predictions: Minimum resolved predictions required

        Returns:
            BrierResult if enough predictions, None otherwise
        """
        # Filter to resolved predictions
        resolved = [p for p in self._predictions if p.is_resolved]

        if strategy_name:
            resolved = [p for p in resolved if p.strategy_name == strategy_name]

        if len(resolved) < min_predictions:
            logger.debug(
                f"Not enough resolved predictions ({len(resolved)} < {min_predictions})"
            )
            return None

        # Calculate Brier score
        total_brier = 0.0
        n_correct = 0
        total_probability = 0.0

        for pred in resolved:
            contribution = pred.brier_contribution()
            if contribution is not None:
                total_brier += contribution
                total_probability += pred.predicted_probability
                if pred.actual_outcome:
                    n_correct += 1

        n = len(resolved)
        brier_score = total_brier / n
        win_rate = n_correct / n
        mean_probability = total_probability / n

        # Calculate per-strategy scores
        strategy_scores: dict[str, float] = {}
        strategies = set(p.strategy_name for p in resolved)
        for strat in strategies:
            strat_preds = [p for p in resolved if p.strategy_name == strat]
            if strat_preds:
                strat_brier = sum(
                    p.brier_contribution() or 0 for p in strat_preds
                ) / len(strat_preds)
                strategy_scores[strat] = strat_brier

        result = BrierResult(
            brier_score=brier_score,
            n_predictions=n,
            n_correct=n_correct,
            win_rate=win_rate,
            mean_probability=mean_probability,
            is_calibrated=brier_score < self._calibration_threshold,
            strategy_scores=strategy_scores,
        )

        logger.info(
            f"Brier score: {brier_score:.4f} "
            f"(n={n}, win_rate={win_rate:.1%}, calibrated={result.is_calibrated})"
        )

        return result

    def get_all_predictions(self) -> list[PredictionRecord]:
        """Get all tracked predictions."""
        return self._predictions.copy()

    def get_resolved_predictions(self) -> list[PredictionRecord]:
        """Get only resolved predictions."""
        return [p for p in self._predictions if p.is_resolved]

    def get_pending_predictions(self) -> list[PredictionRecord]:
        """Get predictions awaiting resolution."""
        return [p for p in self._predictions if not p.is_resolved]

    def get_status(self) -> dict[str, Any]:
        """Get calculator status."""
        resolved = len([p for p in self._predictions if p.is_resolved])
        pending = len(self._predictions) - resolved

        result = self.calculate_brier()

        return {
            "total_predictions": len(self._predictions),
            "resolved": resolved,
            "pending": pending,
            "brier_score": result.brier_score if result else None,
            "is_calibrated": result.is_calibrated if result else None,
            "calibration_threshold": self._calibration_threshold,
        }

    def clear(self) -> None:
        """Clear all predictions."""
        self._predictions.clear()


def calculate_brier_score(
    predicted_probabilities: list[float],
    actual_outcomes: list[bool],
) -> float:
    """
    Calculate Brier score for a list of predictions.

    This is a stateless utility function for one-off calculations.

    Args:
        predicted_probabilities: List of probability estimates (0-1)
        actual_outcomes: List of outcomes (True = win, False = loss)

    Returns:
        Brier score (0-1, lower is better)

    Raises:
        ValueError: If lists are empty or different lengths
    """
    if len(predicted_probabilities) != len(actual_outcomes):
        raise ValueError(
            f"Length mismatch: {len(predicted_probabilities)} predictions "
            f"vs {len(actual_outcomes)} outcomes"
        )

    if not predicted_probabilities:
        raise ValueError("Cannot calculate Brier score with no predictions")

    total = 0.0
    for prob, outcome in zip(predicted_probabilities, actual_outcomes, strict=True):
        outcome_value = 1.0 if outcome else 0.0
        total += (prob - outcome_value) ** 2

    return total / len(predicted_probabilities)
