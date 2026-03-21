"""Training scheduler for periodic model retraining."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from kalshi_bot.ml.training.trainer import ModelTrainer
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


class TrainingScheduler:
    """
    Schedules automatic model retraining based on triggers.

    Triggers retraining when:
    - 100 new settlements since last training
    - Accuracy drops 10%+ from baseline
    - 7 days elapsed since last training

    After training, automatically activates the new model if it
    outperforms the current model.
    """

    # Trigger thresholds
    NEW_SETTLEMENTS_TRIGGER = 100
    ACCURACY_DROP_TRIGGER = 0.10  # 10% drop
    DAYS_ELAPSED_TRIGGER = 7
    CHECK_INTERVAL_SECONDS = 3600  # Check every hour

    def __init__(
        self,
        db: Database,
        trainer: ModelTrainer,
    ) -> None:
        """
        Initialize training scheduler.

        Args:
            db: Database connection
            trainer: Model trainer instance
        """
        self._db = db
        self._trainer = trainer

        self._running = False
        self._last_check: datetime | None = None
        self._last_training: datetime | None = None
        self._settlements_at_last_training = 0
        self._baseline_accuracy: float | None = None

    async def start(self, shutdown_event: asyncio.Event) -> None:
        """
        Start the training scheduler loop.

        Args:
            shutdown_event: Event to signal shutdown
        """
        self._running = True
        logger.info("Starting training scheduler")

        # Initialize state
        await self._initialize_state()

        while not shutdown_event.is_set():
            await self._check_triggers()

            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=self.CHECK_INTERVAL_SECONDS,
                )
                break
            except asyncio.TimeoutError:
                pass

        self._running = False
        logger.info("Training scheduler stopped")

    async def _initialize_state(self) -> None:
        """Initialize scheduler state from database."""
        # Get last training info
        last_model = await self._db.fetch_one(
            """
            SELECT trained_at, metrics, training_samples
            FROM ml_models
            WHERE is_active = TRUE
            ORDER BY trained_at DESC
            LIMIT 1
            """
        )

        if last_model and last_model["trained_at"]:
            self._last_training = self._parse_datetime(last_model["trained_at"])
            if last_model["metrics"]:
                import json
                metrics = json.loads(last_model["metrics"])
                self._baseline_accuracy = metrics.get("accuracy")

        # Get settlement count at last training
        settlement_count = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_settlements"
        )
        if settlement_count:
            # Estimate based on training samples (each market generates ~5 samples)
            training_samples = last_model["training_samples"] if last_model else 0
            self._settlements_at_last_training = max(0, settlement_count["count"] - training_samples // 5)

    async def _check_triggers(self) -> None:
        """Check if any retraining triggers are met."""
        self._last_check = datetime.utcnow()

        # Check if we have minimum data for training
        settlement_count = await self._get_settlement_count()
        if settlement_count < ModelTrainer.MIN_SETTLEMENTS:
            logger.debug(
                f"Insufficient settlements for training: "
                f"{settlement_count} < {ModelTrainer.MIN_SETTLEMENTS}"
            )
            return

        trigger_reason = await self._evaluate_triggers(settlement_count)

        if trigger_reason:
            logger.info(f"Training triggered: {trigger_reason}")
            await self._run_training()

    async def _evaluate_triggers(self, settlement_count: int) -> str | None:
        """
        Evaluate all training triggers.

        Args:
            settlement_count: Current number of settlements

        Returns:
            Trigger reason or None if no trigger met
        """
        # Trigger 1: New settlements
        new_settlements = settlement_count - self._settlements_at_last_training
        if new_settlements >= self.NEW_SETTLEMENTS_TRIGGER:
            return f"{new_settlements} new settlements since last training"

        # Trigger 2: Time elapsed
        if self._last_training:
            days_elapsed = (datetime.utcnow() - self._last_training).days
            if days_elapsed >= self.DAYS_ELAPSED_TRIGGER:
                return f"{days_elapsed} days since last training"

        # Trigger 3: Accuracy drop (check prediction accuracy)
        if self._baseline_accuracy:
            current_accuracy = await self._get_recent_accuracy()
            if current_accuracy is not None:
                drop = self._baseline_accuracy - current_accuracy
                if drop >= self.ACCURACY_DROP_TRIGGER:
                    return f"Accuracy dropped {drop:.1%} from baseline"

        return None

    async def _get_recent_accuracy(self, window_predictions: int = 50) -> float | None:
        """Get accuracy over recent predictions."""
        result = await self._db.fetch_one(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM ml_predictions
            WHERE actual_outcome IS NOT NULL
            ORDER BY predicted_at DESC
            LIMIT ?
            """,
            (window_predictions,),
        )

        if result and result["total"] > 0:
            return result["correct"] / result["total"]
        return None

    async def _run_training(self) -> None:
        """Execute model training and activation."""
        try:
            # Try logistic first, then gradient boost if we have enough data
            settlement_count = await self._get_settlement_count()

            if settlement_count >= 1000:
                # Use gradient boost for larger datasets
                result = await self._trainer.train_model("gradient_boost")
            else:
                result = await self._trainer.train_model("logistic")

            if not result.success:
                logger.error(f"Training failed: {result.error_message}")
                return

            # Compare with current model
            comparison = await self._trainer.compare_models(result.model_id)

            if comparison.should_replace:
                logger.info(f"Activating new model: {comparison.reason}")
                await self._trainer.activate_model(result.model_id)
                self._baseline_accuracy = result.metrics.accuracy if result.metrics else None
            else:
                logger.info(f"Keeping current model: {comparison.reason}")

            # Update state
            self._last_training = datetime.utcnow()
            self._settlements_at_last_training = settlement_count

        except Exception as e:
            logger.error(f"Training run failed: {e}")

    async def _get_settlement_count(self) -> int:
        """Get total settlement count."""
        result = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_settlements"
        )
        return result["count"] if result else 0

    async def force_training(self, model_type: str = "logistic") -> dict:
        """
        Force immediate model training.

        Args:
            model_type: Type of model to train

        Returns:
            Training result as dictionary
        """
        logger.info(f"Forced training requested for {model_type}")
        result = await self._trainer.train_model(model_type)

        if result.success:
            comparison = await self._trainer.compare_models(result.model_id)
            if comparison.should_replace:
                await self._trainer.activate_model(result.model_id)
                result_dict = result.to_dict()
                result_dict["activated"] = True
                result_dict["comparison"] = {
                    "improvement": comparison.improvement,
                    "reason": comparison.reason,
                }
                return result_dict

        return result.to_dict()

    def _parse_datetime(self, value) -> datetime | None:
        """Parse datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                clean = value.replace("Z", "+00:00")
                return datetime.fromisoformat(clean)
            except ValueError:
                return None
        return None

    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "running": self._running,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "last_training": self._last_training.isoformat() if self._last_training else None,
            "settlements_at_last_training": self._settlements_at_last_training,
            "baseline_accuracy": self._baseline_accuracy,
            "triggers": {
                "new_settlements_threshold": self.NEW_SETTLEMENTS_TRIGGER,
                "accuracy_drop_threshold": self.ACCURACY_DROP_TRIGGER,
                "days_elapsed_threshold": self.DAYS_ELAPSED_TRIGGER,
            },
        }
