"""Calibration curve analysis for probability predictions.

A calibration curve shows how well our predicted probabilities match actual
outcomes. Perfectly calibrated predictions fall along the diagonal:
- When we predict 70%, the event should occur ~70% of the time
- Deviations from the diagonal indicate miscalibration

This is critical for:
1. Validating our probability models
2. Identifying systematic over/under-confidence
3. Adjusting predictions if needed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationBucket:
    """A bucket in the calibration curve."""

    bucket_start: float  # e.g., 0.6 for 60-70% bucket
    bucket_end: float  # e.g., 0.7
    bucket_center: float  # e.g., 0.65
    n_predictions: int
    n_outcomes: int  # Number where outcome was True
    mean_predicted: float  # Average predicted probability
    actual_rate: float  # Actual outcome rate
    deviation: float  # actual_rate - mean_predicted

    @property
    def is_overconfident(self) -> bool:
        """True if we predict higher than actual rate."""
        return self.deviation < 0

    @property
    def is_underconfident(self) -> bool:
        """True if we predict lower than actual rate."""
        return self.deviation > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bucket_range": f"{self.bucket_start:.0%}-{self.bucket_end:.0%}",
            "bucket_center": self.bucket_center,
            "n_predictions": self.n_predictions,
            "n_outcomes": self.n_outcomes,
            "mean_predicted": self.mean_predicted,
            "actual_rate": self.actual_rate,
            "deviation": self.deviation,
            "is_overconfident": self.is_overconfident,
        }


@dataclass
class CalibrationAnalysis:
    """Complete calibration analysis results."""

    buckets: list[CalibrationBucket]
    total_predictions: int
    max_deviation: float  # Largest absolute deviation from diagonal
    mean_absolute_deviation: float  # Average deviation (calibration error)
    is_well_calibrated: bool  # True if max deviation < threshold (5pp)

    # Overall bias indicators
    overall_overconfident: bool  # More buckets overconfident than under
    overall_underconfident: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "buckets": [b.to_dict() for b in self.buckets],
            "total_predictions": self.total_predictions,
            "max_deviation": self.max_deviation,
            "mean_absolute_deviation": self.mean_absolute_deviation,
            "is_well_calibrated": self.is_well_calibrated,
            "overall_overconfident": self.overall_overconfident,
            "overall_underconfident": self.overall_underconfident,
        }

    def format_report(self) -> str:
        """Generate human-readable calibration report."""
        lines = [
            "=" * 60,
            "CALIBRATION REPORT",
            "=" * 60,
            f"Total predictions: {self.total_predictions}",
            f"Max deviation: {self.max_deviation:.1%}",
            f"Mean absolute deviation: {self.mean_absolute_deviation:.1%}",
            f"Well calibrated: {'YES' if self.is_well_calibrated else 'NO'}",
            "",
            "Bucket Analysis:",
            "-" * 60,
            f"{'Range':<12} {'N':>6} {'Predicted':>10} {'Actual':>10} {'Deviation':>10}",
            "-" * 60,
        ]

        for bucket in self.buckets:
            range_str = f"{bucket.bucket_start:.0%}-{bucket.bucket_end:.0%}"
            deviation_str = f"{bucket.deviation:+.1%}"
            if abs(bucket.deviation) > 0.05:
                deviation_str += " !"  # Flag significant deviations
            lines.append(
                f"{range_str:<12} {bucket.n_predictions:>6} "
                f"{bucket.mean_predicted:>10.1%} {bucket.actual_rate:>10.1%} "
                f"{deviation_str:>10}"
            )

        lines.extend([
            "-" * 60,
            "",
        ])

        if self.overall_overconfident:
            lines.append("TENDENCY: Overconfident (predictions too high)")
        elif self.overall_underconfident:
            lines.append("TENDENCY: Underconfident (predictions too low)")
        else:
            lines.append("TENDENCY: Balanced")

        lines.append("=" * 60)
        return "\n".join(lines)


class CalibrationCurve:
    """Builds and analyzes calibration curves from predictions.

    The calibration curve buckets predictions by probability range and
    compares the average prediction in each bucket to the actual outcome
    rate. Perfect calibration means the curve follows the diagonal.
    """

    # Default configuration
    DEFAULT_N_BUCKETS = 10  # 0-10%, 10-20%, ..., 90-100%
    DEFAULT_MIN_BUCKET_SIZE = 5  # Minimum predictions per bucket
    DEFAULT_WELL_CALIBRATED_THRESHOLD = 0.05  # 5pp max deviation

    def __init__(
        self,
        n_buckets: int = DEFAULT_N_BUCKETS,
        min_bucket_size: int = DEFAULT_MIN_BUCKET_SIZE,
        well_calibrated_threshold: float = DEFAULT_WELL_CALIBRATED_THRESHOLD,
    ) -> None:
        """
        Initialize the calibration curve analyzer.

        Args:
            n_buckets: Number of probability buckets (default 10)
            min_bucket_size: Minimum predictions needed per bucket
            well_calibrated_threshold: Max deviation for "well calibrated"
        """
        self._n_buckets = n_buckets
        self._min_bucket_size = min_bucket_size
        self._threshold = well_calibrated_threshold

        # Storage for predictions
        self._predictions: list[tuple[float, bool]] = []

    def add_prediction(self, predicted_prob: float, actual_outcome: bool) -> None:
        """
        Add a prediction-outcome pair.

        Args:
            predicted_prob: Predicted probability (0-1)
            actual_outcome: Whether the event occurred
        """
        if not 0 <= predicted_prob <= 1:
            raise ValueError(f"Probability must be in [0,1], got {predicted_prob}")
        self._predictions.append((predicted_prob, actual_outcome))

    def add_predictions(
        self,
        predicted_probs: list[float],
        actual_outcomes: list[bool],
    ) -> None:
        """
        Add multiple prediction-outcome pairs.

        Args:
            predicted_probs: List of predicted probabilities
            actual_outcomes: List of actual outcomes
        """
        if len(predicted_probs) != len(actual_outcomes):
            raise ValueError("Prediction and outcome lists must have same length")

        for prob, outcome in zip(predicted_probs, actual_outcomes, strict=True):
            self.add_prediction(prob, outcome)

    def analyze(self) -> CalibrationAnalysis | None:
        """
        Analyze calibration from stored predictions.

        Returns:
            CalibrationAnalysis if enough predictions, None otherwise
        """
        if len(self._predictions) < self._min_bucket_size:
            logger.debug(
                f"Not enough predictions for calibration analysis "
                f"({len(self._predictions)} < {self._min_bucket_size})"
            )
            return None

        # Build buckets
        bucket_width = 1.0 / self._n_buckets
        buckets: list[CalibrationBucket] = []
        total_deviation = 0.0
        max_deviation = 0.0
        overconfident_count = 0
        underconfident_count = 0

        for i in range(self._n_buckets):
            bucket_start = i * bucket_width
            bucket_end = (i + 1) * bucket_width
            bucket_center = (bucket_start + bucket_end) / 2

            # Get predictions in this bucket
            bucket_preds = [
                (prob, outcome)
                for prob, outcome in self._predictions
                if bucket_start <= prob < bucket_end
            ]

            # Handle the last bucket (include 1.0)
            if i == self._n_buckets - 1:
                bucket_preds.extend([
                    (prob, outcome)
                    for prob, outcome in self._predictions
                    if prob == 1.0
                ])

            if len(bucket_preds) < self._min_bucket_size:
                continue

            n_predictions = len(bucket_preds)
            n_outcomes = sum(1 for _, outcome in bucket_preds if outcome)
            mean_predicted = sum(prob for prob, _ in bucket_preds) / n_predictions
            actual_rate = n_outcomes / n_predictions
            deviation = actual_rate - mean_predicted

            bucket = CalibrationBucket(
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                bucket_center=bucket_center,
                n_predictions=n_predictions,
                n_outcomes=n_outcomes,
                mean_predicted=mean_predicted,
                actual_rate=actual_rate,
                deviation=deviation,
            )
            buckets.append(bucket)

            # Track statistics
            total_deviation += abs(deviation)
            if abs(deviation) > max_deviation:
                max_deviation = abs(deviation)
            if bucket.is_overconfident:
                overconfident_count += 1
            elif bucket.is_underconfident:
                underconfident_count += 1

        if not buckets:
            return None

        mean_absolute_deviation = total_deviation / len(buckets)

        analysis = CalibrationAnalysis(
            buckets=buckets,
            total_predictions=len(self._predictions),
            max_deviation=max_deviation,
            mean_absolute_deviation=mean_absolute_deviation,
            is_well_calibrated=max_deviation < self._threshold,
            overall_overconfident=overconfident_count > underconfident_count,
            overall_underconfident=underconfident_count > overconfident_count,
        )

        logger.info(
            f"Calibration analysis: max_deviation={max_deviation:.1%}, "
            f"MAD={mean_absolute_deviation:.1%}, "
            f"well_calibrated={analysis.is_well_calibrated}"
        )

        return analysis

    def get_ascii_curve(self, width: int = 50, height: int = 20) -> str:
        """
        Generate ASCII representation of calibration curve.

        Args:
            width: Width of the chart in characters
            height: Height of the chart in characters

        Returns:
            ASCII art string of the calibration curve
        """
        analysis = self.analyze()
        if not analysis:
            return "Insufficient data for calibration curve"

        # Initialize grid
        grid = [[" " for _ in range(width + 5)] for _ in range(height + 3)]

        # Draw axes
        for y in range(height):
            grid[y][4] = "|"
        for x in range(5, width + 5):
            grid[height][x] = "-"

        # Add axis labels
        grid[height][4] = "+"
        grid[height + 1][2:5] = list("0  ")
        grid[height + 1][width + 2:width + 5] = list("100")
        grid[0][0:4] = list("100%")
        grid[height - 1][0:4] = list("  0%")

        # Draw diagonal (perfect calibration)
        for i in range(min(width, height)):
            x = 5 + int(i * width / height)
            y = height - 1 - i
            if 0 <= y < height and 5 <= x < width + 5:
                grid[y][x] = "."

        # Plot actual calibration points
        for bucket in analysis.buckets:
            x = 5 + int(bucket.bucket_center * width)
            y = height - 1 - int(bucket.actual_rate * (height - 1))
            if 0 <= y < height and 5 <= x < width + 5:
                grid[y][x] = "*"

        # Add title and labels
        lines = ["CALIBRATION CURVE (* = actual, . = perfect)", ""]
        lines.extend("".join(row) for row in grid)
        lines.append("    Predicted Probability -->")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored predictions."""
        self._predictions.clear()

    def get_status(self) -> dict[str, Any]:
        """Get current status."""
        analysis = self.analyze()
        return {
            "n_predictions": len(self._predictions),
            "n_buckets_configured": self._n_buckets,
            "is_well_calibrated": analysis.is_well_calibrated if analysis else None,
            "max_deviation": analysis.max_deviation if analysis else None,
            "mean_absolute_deviation": (
                analysis.mean_absolute_deviation if analysis else None
            ),
        }


def quick_calibration_check(
    predicted_probs: list[float],
    actual_outcomes: list[bool],
    n_buckets: int = 10,
) -> CalibrationAnalysis | None:
    """
    Quick calibration analysis utility function.

    Args:
        predicted_probs: List of predicted probabilities
        actual_outcomes: List of actual outcomes
        n_buckets: Number of buckets for analysis

    Returns:
        CalibrationAnalysis if enough data, None otherwise
    """
    curve = CalibrationCurve(n_buckets=n_buckets)
    curve.add_predictions(predicted_probs, actual_outcomes)
    return curve.analyze()
