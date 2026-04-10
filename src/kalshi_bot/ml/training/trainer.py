"""Model training orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from kalshi_bot.ml.feature_engineer import FeatureEngineer, FEATURE_NAMES
from kalshi_bot.ml.models import (
    BasePredictionModel,
    GradientBoostModel,
    LogisticRegressionModel,
    ModelMetrics,
)
from kalshi_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from kalshi_bot.persistence.database import Database

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Result of model training."""

    model_id: str
    model_type: str
    success: bool
    metrics: ModelMetrics | None
    cv_accuracy: float | None
    cv_std: float | None
    training_samples: int
    feature_count: int
    model_path: str | None
    error_message: str | None = None
    trained_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "success": self.success,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "cv_accuracy": self.cv_accuracy,
            "cv_std": self.cv_std,
            "training_samples": self.training_samples,
            "feature_count": self.feature_count,
            "model_path": self.model_path,
            "error_message": self.error_message,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }


@dataclass
class ModelComparison:
    """Comparison between two models."""

    new_model_id: str
    current_model_id: str
    new_accuracy: float
    current_accuracy: float
    improvement: float
    should_replace: bool
    reason: str


class ModelTrainer:
    """
    Orchestrates ML model training and evaluation.

    Responsibilities:
    - Prepare training data from settled markets
    - Train models with cross-validation
    - Compare new models against current active model
    - Save and register models in database
    """

    # Minimum requirements for training
    MIN_SETTLEMENTS = 100
    MIN_SNAPSHOTS_PER_MARKET = 1  # Lowered from 10 to use available data

    # Sampling parameters
    HOURS_BEFORE_SETTLEMENT = [1, 3, 6, 12, 24]

    def __init__(
        self,
        db: Database,
        models_dir: str = "data/models",
    ) -> None:
        """
        Initialize model trainer.

        Args:
            db: Database connection
            models_dir: Directory to save trained models
        """
        self._db = db
        self._models_dir = Path(models_dir)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._feature_engineer = FeatureEngineer(db)

    async def prepare_training_data(
        self,
        min_snapshots: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from settled markets.

        Args:
            min_snapshots: Minimum snapshots required per market

        Returns:
            Tuple of (X, y) arrays for training
        """
        min_snapshots = min_snapshots or self.MIN_SNAPSHOTS_PER_MARKET

        # Get settled markets with enough history
        settlements = await self._db.fetch_all(
            """
            SELECT ticker, outcome, confirmed_at
            FROM market_settlements
            WHERE snapshot_count >= ?
            ORDER BY confirmed_at
            """,
            (min_snapshots,),
        )

        logger.info(f"Found {len(settlements)} settlements with sufficient data")

        X_list = []
        y_list = []

        for settlement in settlements:
            ticker = settlement["ticker"]
            outcome = settlement["outcome"]
            confirmed_at = self._parse_datetime(settlement["confirmed_at"])

            if confirmed_at is None:
                continue

            label = 1 if outcome == "yes" else 0

            # Compute features at multiple time points
            for hours in self.HOURS_BEFORE_SETTLEMENT:
                as_of = confirmed_at - timedelta(hours=hours)
                features = await self._feature_engineer.compute_features(ticker, as_of)

                if features:
                    X_list.append(features.to_array())
                    y_list.append(label)

        if not X_list:
            raise ValueError("No training samples could be generated")

        X = np.array(X_list)
        y = np.array(y_list)

        n_yes = np.sum(y)
        n_no = len(y) - n_yes
        imbalance_ratio = n_no / n_yes if n_yes > 0 else float('inf')

        logger.info(f"Prepared {len(y)} training samples from {len(settlements)} markets")
        logger.info(f"Class distribution: YES={n_yes} ({n_yes/len(y)*100:.1f}%), NO={n_no} ({n_no/len(y)*100:.1f}%)")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.1f}:1 (NO:YES)")

        # Apply SMOTE to balance classes if heavily imbalanced
        if imbalance_ratio > 3.0 and n_yes >= 10:
            X, y = self._apply_smote(X, y)

        return X, y

    def _apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to balance the dataset.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            Resampled (X, y) tuple
        """
        try:
            from imblearn.over_sampling import SMOTE

            n_yes_before = np.sum(y)
            n_no_before = len(y) - n_yes_before

            # Use SMOTE to oversample minority class
            # Don't fully balance - aim for ~1:3 ratio (YES:NO)
            target_yes = min(n_no_before // 3, n_yes_before * 5)
            sampling_strategy = target_yes / n_no_before

            smote = SMOTE(
                sampling_strategy=min(sampling_strategy, 1.0),
                random_state=42,
                k_neighbors=min(5, n_yes_before - 1),
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)

            n_yes_after = np.sum(y_resampled)
            n_no_after = len(y_resampled) - n_yes_after

            logger.info(
                f"SMOTE resampling: YES {n_yes_before} -> {n_yes_after}, "
                f"NO {n_no_before} -> {n_no_after}"
            )
            logger.info(
                f"New ratio: {n_no_after/n_yes_after:.1f}:1 (NO:YES)"
            )

            return X_resampled, y_resampled

        except ImportError:
            logger.warning("imbalanced-learn not installed, skipping SMOTE")
            return X, y
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}, using original data")
            return X, y

    async def train_model(
        self,
        model_type: str = "logistic",
        run_cv: bool = True,
        n_folds: int = 5,
    ) -> TrainingResult:
        """
        Train a new model.

        Args:
            model_type: Type of model ('logistic' or 'gradient_boost')
            run_cv: Whether to run cross-validation
            n_folds: Number of CV folds

        Returns:
            TrainingResult with model info and metrics
        """
        model_id = f"{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"

        try:
            # Check if we have enough data
            settlement_count = await self._get_settlement_count()
            if settlement_count < self.MIN_SETTLEMENTS:
                return TrainingResult(
                    model_id=model_id,
                    model_type=model_type,
                    success=False,
                    metrics=None,
                    cv_accuracy=None,
                    cv_std=None,
                    training_samples=0,
                    feature_count=0,
                    model_path=None,
                    error_message=f"Insufficient data: {settlement_count} settlements < {self.MIN_SETTLEMENTS} required",
                )

            # Prepare training data
            X, y = await self.prepare_training_data()

            if len(y) < 100:
                return TrainingResult(
                    model_id=model_id,
                    model_type=model_type,
                    success=False,
                    metrics=None,
                    cv_accuracy=None,
                    cv_std=None,
                    training_samples=len(y),
                    feature_count=X.shape[1],
                    model_path=None,
                    error_message=f"Insufficient training samples: {len(y)} < 100 required",
                )

            # Create model
            if model_type == "logistic":
                model = LogisticRegressionModel(model_id=model_id)
            elif model_type == "gradient_boost":
                model = GradientBoostModel(model_id=model_id)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Run cross-validation
            cv_accuracy = None
            cv_std = None
            if run_cv:
                cv_results = model.cross_validate(X, y, n_folds, FEATURE_NAMES)
                cv_accuracy = cv_results.mean_score
                cv_std = cv_results.std_score
                logger.info(f"CV results: {cv_accuracy:.3f} (+/- {cv_std:.3f})")

            # Train final model on all data
            metrics = model.train(X, y, FEATURE_NAMES)

            # Save model
            model_path = self._models_dir / f"{model_id}.joblib"
            model.save(model_path)

            # Register in database
            await self._register_model(model, str(model_path), cv_accuracy, cv_std)

            return TrainingResult(
                model_id=model_id,
                model_type=model_type,
                success=True,
                metrics=metrics,
                cv_accuracy=cv_accuracy,
                cv_std=cv_std,
                training_samples=len(y),
                feature_count=X.shape[1],
                model_path=str(model_path),
                trained_at=model.trained_at,
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return TrainingResult(
                model_id=model_id,
                model_type=model_type,
                success=False,
                metrics=None,
                cv_accuracy=None,
                cv_std=None,
                training_samples=0,
                feature_count=0,
                model_path=None,
                error_message=str(e),
            )

    async def compare_models(
        self,
        new_model_id: str,
        current_model_id: str | None = None,
    ) -> ModelComparison:
        """
        Compare a new model against the current active model.

        Args:
            new_model_id: ID of newly trained model
            current_model_id: ID of current model (if None, get from DB)

        Returns:
            ModelComparison with recommendation
        """
        # Get new model metrics
        new_model = await self._db.fetch_one(
            "SELECT * FROM ml_models WHERE model_id = ?",
            (new_model_id,),
        )

        if not new_model:
            return ModelComparison(
                new_model_id=new_model_id,
                current_model_id=current_model_id or "",
                new_accuracy=0,
                current_accuracy=0,
                improvement=0,
                should_replace=False,
                reason="New model not found",
            )

        new_metrics = json.loads(new_model["metrics"])
        # Use F1 for YES class as primary metric (minority class performance)
        new_f1_yes = new_metrics.get("f1_yes", new_metrics.get("f1_score", 0))
        new_accuracy = new_metrics.get("accuracy", 0)

        # Get current active model if not specified
        if current_model_id is None:
            current = await self._db.fetch_one(
                """
                SELECT * FROM ml_models
                WHERE is_active = TRUE AND model_type = ?
                """,
                (new_model["model_type"],),
            )
            if current:
                current_model_id = current["model_id"]

        # If no current model, new model should be activated
        if not current_model_id:
            return ModelComparison(
                new_model_id=new_model_id,
                current_model_id="",
                new_accuracy=new_f1_yes,  # Report F1 as "accuracy" for compatibility
                current_accuracy=0,
                improvement=new_f1_yes,
                should_replace=True,
                reason="No current active model",
            )

        # Get current model metrics
        current = await self._db.fetch_one(
            "SELECT * FROM ml_models WHERE model_id = ?",
            (current_model_id,),
        )

        if not current:
            return ModelComparison(
                new_model_id=new_model_id,
                current_model_id=current_model_id,
                new_accuracy=new_f1_yes,
                current_accuracy=0,
                improvement=new_f1_yes,
                should_replace=True,
                reason="Current model not found",
            )

        current_metrics = json.loads(current["metrics"])
        current_f1_yes = current_metrics.get("f1_yes", current_metrics.get("f1_score", 0))
        current_accuracy = current_metrics.get("accuracy", 0)
        improvement = new_f1_yes - current_f1_yes

        # Decision criteria: replace if F1 improvement > 5% (minority class is harder)
        should_replace = improvement >= 0.05

        reason = (
            f"New model F1(YES)={new_f1_yes:.3f} vs current {current_f1_yes:.3f}"
            f" (improvement: {improvement:+.3f}), "
            f"Accuracy: {new_accuracy:.3f}"
        )

        return ModelComparison(
            new_model_id=new_model_id,
            current_model_id=current_model_id,
            new_accuracy=new_f1_yes,  # Report F1 as primary metric
            current_accuracy=current_f1_yes,
            improvement=improvement,
            should_replace=should_replace,
            reason=reason,
        )

    async def activate_model(self, model_id: str) -> bool:
        """
        Activate a model for prediction.

        Args:
            model_id: Model to activate

        Returns:
            True if successful
        """
        # Get model info
        model = await self._db.fetch_one(
            "SELECT model_type FROM ml_models WHERE model_id = ?",
            (model_id,),
        )

        if not model:
            logger.error(f"Model not found: {model_id}")
            return False

        # Deactivate current active model of same type
        await self._db.execute(
            """
            UPDATE ml_models
            SET is_active = FALSE, retired_at = CURRENT_TIMESTAMP
            WHERE model_type = ? AND is_active = TRUE
            """,
            (model["model_type"],),
        )

        # Activate new model
        await self._db.execute(
            """
            UPDATE ml_models
            SET is_active = TRUE, activated_at = CURRENT_TIMESTAMP, status = 'active'
            WHERE model_id = ?
            """,
            (model_id,),
        )

        logger.info(f"Activated model: {model_id}")
        return True

    async def get_active_model(
        self,
        model_type: str = "logistic",
    ) -> BasePredictionModel | None:
        """
        Get the currently active model.

        Args:
            model_type: Type of model to get

        Returns:
            Loaded model or None
        """
        model_record = await self._db.fetch_one(
            """
            SELECT * FROM ml_models
            WHERE is_active = TRUE AND model_type = ?
            """,
            (model_type,),
        )

        if not model_record:
            return None

        model_path = model_record["model_path"]
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return None

        if model_type == "logistic":
            model = LogisticRegressionModel()
        elif model_type == "gradient_boost":
            model = GradientBoostModel()
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None

        model.load(model_path)
        return model

    async def _get_settlement_count(self) -> int:
        """Get total number of settlements."""
        result = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_settlements"
        )
        return result["count"] if result else 0

    async def _register_model(
        self,
        model: BasePredictionModel,
        model_path: str,
        cv_accuracy: float | None,
        cv_std: float | None,
    ) -> None:
        """Register trained model in database."""
        await self._db.execute(
            """
            INSERT INTO ml_models (
                model_id, model_type, model_path,
                training_samples, feature_count,
                metrics, cv_results, feature_importance,
                status, trained_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'trained', ?)
            """,
            (
                model.model_id,
                model.model_type,
                model_path,
                model.metrics.n_samples if model.metrics else 0,
                len(FEATURE_NAMES),
                json.dumps(model.metrics.to_dict()) if model.metrics else "{}",
                json.dumps(model.cv_results.to_dict()) if model.cv_results else None,
                json.dumps(model.feature_importance),
                model.trained_at.isoformat() if model.trained_at else None,
            ),
        )

    def _parse_datetime(self, value) -> datetime | None:
        """Parse datetime from various formats, returning naive UTC datetime."""
        if value is None:
            return None
        if isinstance(value, datetime):
            # Strip timezone info if present
            if value.tzinfo is not None:
                return value.replace(tzinfo=None)
            return value
        if isinstance(value, str):
            try:
                clean = value.replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean)
                # Strip timezone info to get naive UTC datetime
                if dt.tzinfo is not None:
                    return dt.replace(tzinfo=None)
                return dt
            except ValueError:
                return None
        return None

    async def get_training_status(self) -> dict:
        """Get current training status and data availability."""
        settlement_count = await self._get_settlement_count()
        snapshot_count = await self._db.fetch_one(
            "SELECT COUNT(*) as count FROM market_snapshots"
        )

        # Get active models
        active_models = await self._db.fetch_all(
            "SELECT model_id, model_type, metrics, trained_at FROM ml_models WHERE is_active = TRUE"
        )

        # Get recent training history
        recent_training = await self._db.fetch_all(
            """
            SELECT model_id, model_type, status, metrics, trained_at
            FROM ml_models
            ORDER BY trained_at DESC
            LIMIT 5
            """
        )

        return {
            "ready_for_training": settlement_count >= self.MIN_SETTLEMENTS,
            "settlement_count": settlement_count,
            "min_settlements_required": self.MIN_SETTLEMENTS,
            "snapshot_count": snapshot_count["count"] if snapshot_count else 0,
            "active_models": [
                {
                    "model_id": m["model_id"],
                    "model_type": m["model_type"],
                    "metrics": json.loads(m["metrics"]) if m["metrics"] else None,
                    "trained_at": m["trained_at"],
                }
                for m in active_models
            ],
            "recent_training": [
                {
                    "model_id": m["model_id"],
                    "model_type": m["model_type"],
                    "status": m["status"],
                    "trained_at": m["trained_at"],
                }
                for m in recent_training
            ],
        }
