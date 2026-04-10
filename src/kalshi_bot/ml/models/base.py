"""Base class for prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    accuracy: float
    auc: float
    log_loss: float
    precision: float
    recall: float
    f1_score: float
    brier_score: float  # Calibration metric

    # Per-class metrics
    precision_yes: float = 0.0
    recall_yes: float = 0.0
    precision_no: float = 0.0
    recall_no: float = 0.0
    f1_yes: float = 0.0  # F1 for YES class (minority)
    f1_no: float = 0.0   # F1 for NO class (majority)

    # Balanced metrics (important for imbalanced data)
    balanced_accuracy: float = 0.0

    # Optimal threshold for imbalanced classification
    optimal_threshold: float = 0.5

    # Sample info
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "auc": self.auc,
            "log_loss": self.log_loss,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "brier_score": self.brier_score,
            "precision_yes": self.precision_yes,
            "recall_yes": self.recall_yes,
            "precision_no": self.precision_no,
            "recall_no": self.recall_no,
            "f1_yes": self.f1_yes,
            "f1_no": self.f1_no,
            "balanced_accuracy": self.balanced_accuracy,
            "optimal_threshold": self.optimal_threshold,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetrics:
        """Create from dictionary."""
        return cls(
            accuracy=data.get("accuracy", 0.0),
            auc=data.get("auc", 0.0),
            log_loss=data.get("log_loss", float("inf")),
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1_score=data.get("f1_score", 0.0),
            brier_score=data.get("brier_score", 1.0),
            precision_yes=data.get("precision_yes", 0.0),
            recall_yes=data.get("recall_yes", 0.0),
            precision_no=data.get("precision_no", 0.0),
            recall_no=data.get("recall_no", 0.0),
            f1_yes=data.get("f1_yes", 0.0),
            f1_no=data.get("f1_no", 0.0),
            balanced_accuracy=data.get("balanced_accuracy", 0.0),
            optimal_threshold=data.get("optimal_threshold", 0.5),
            n_samples=data.get("n_samples", 0),
            n_positive=data.get("n_positive", 0),
            n_negative=data.get("n_negative", 0),
        )


@dataclass
class CVResults:
    """Cross-validation results."""

    fold_scores: list[float]
    mean_score: float
    std_score: float
    fold_metrics: list[ModelMetrics] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fold_scores": self.fold_scores,
            "mean": self.mean_score,
            "std": self.std_score,
            "fold_metrics": [m.to_dict() for m in self.fold_metrics],
        }


class BasePredictionModel(ABC):
    """
    Abstract base class for ML prediction models.

    All prediction models should inherit from this class and implement
    the required methods for training, prediction, and persistence.
    """

    def __init__(self, model_id: str | None = None) -> None:
        """
        Initialize the model.

        Args:
            model_id: Unique identifier for this model instance
        """
        self.model_id = model_id or self._generate_model_id()
        self.trained_at: datetime | None = None
        self.metrics: ModelMetrics | None = None
        self.cv_results: CVResults | None = None
        self.feature_importance: dict[str, float] = {}
        self._is_trained = False

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> ModelMetrics:
        """
        Train the model on data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 1 for YES, 0 for NO
            feature_names: Optional list of feature names

        Returns:
            Training metrics
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of YES outcome.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of P(YES) probabilities (n_samples,)
        """
        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model file
        """
        pass

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to model file
        """
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier."""
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary outcomes.

        Args:
            X: Feature matrix
            threshold: Probability threshold for YES prediction

        Returns:
            Array of predictions (1 for YES, 0 for NO)
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def compute_edge(
        self,
        X: np.ndarray,
        market_prices: np.ndarray,
    ) -> np.ndarray:
        """
        Compute edge between model probability and market price.

        Args:
            X: Feature matrix
            market_prices: Current market prices (0-100 cents)

        Returns:
            Array of edges (positive = model thinks YES is underpriced)
        """
        model_proba = self.predict_proba(X)
        market_proba = market_prices / 100.0
        return model_proba - market_proba

    def _generate_model_id(self) -> str:
        """Generate unique model ID."""
        from uuid import uuid4
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid4())[:8]
        return f"{self.model_type}_{timestamp}_{short_uuid}"

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> ModelMetrics:
        """
        Compute comprehensive metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities

        Returns:
            ModelMetrics instance
        """
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            f1_score,
            log_loss,
            precision_recall_fscore_support,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Handle edge cases
        n_samples = len(y_true)
        n_positive = int(np.sum(y_true))
        n_negative = n_samples - n_positive

        # Find optimal threshold for F1 score on minority class
        optimal_threshold = self._find_optimal_threshold(y_true, y_proba)

        # Use optimal threshold for predictions
        y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

        # Basic metrics (using optimal threshold)
        accuracy = accuracy_score(y_true, y_pred_optimal)
        balanced_acc = balanced_accuracy_score(y_true, y_pred_optimal)

        # AUC - requires both classes
        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.5  # Default if only one class

        # Log loss
        try:
            ll = log_loss(y_true, y_proba)
        except ValueError:
            ll = float("inf")

        # Precision/Recall/F1 (using optimal threshold)
        precision = precision_score(y_true, y_pred_optimal, zero_division=0)
        recall = recall_score(y_true, y_pred_optimal, zero_division=0)
        f1 = f1_score(y_true, y_pred_optimal, zero_division=0)

        # Brier score (calibration)
        brier = brier_score_loss(y_true, y_proba)

        # Per-class metrics
        prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred_optimal, labels=[0, 1], zero_division=0
        )

        return ModelMetrics(
            accuracy=accuracy,
            auc=auc,
            log_loss=ll,
            precision=precision,
            recall=recall,
            f1_score=f1,
            brier_score=brier,
            precision_yes=prec_per_class[1],
            recall_yes=rec_per_class[1],
            precision_no=prec_per_class[0],
            recall_no=rec_per_class[0],
            f1_yes=f1_per_class[1],
            f1_no=f1_per_class[0],
            balanced_accuracy=balanced_acc,
            optimal_threshold=optimal_threshold,
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_negative,
        )

    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """
        Find optimal threshold that maximizes F1 score for minority class.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import f1_score

        best_threshold = 0.5
        best_f1 = 0.0

        # Search through thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return float(best_threshold)

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        feature_names: list[str] | None = None,
    ) -> CVResults:
        """
        Perform cross-validation.

        Uses F1 score for minority class as primary metric (not accuracy)
        because the dataset is heavily imbalanced.

        Args:
            X: Feature matrix
            y: Labels
            n_folds: Number of folds
            feature_names: Optional feature names

        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        fold_scores = []
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Create a fresh model instance for this fold
            fold_model = self.__class__(model_id=f"{self.model_id}_fold{fold_idx}")

            # Train and evaluate
            fold_model.train(X_train, y_train, feature_names)
            y_proba = fold_model.predict_proba(X_val)
            y_pred = fold_model.predict(X_val)

            metrics = self._compute_metrics(y_val, y_pred, y_proba)
            # Use F1 for YES class as primary score (minority class performance)
            fold_scores.append(metrics.f1_yes)
            fold_metrics.append(metrics)

        self.cv_results = CVResults(
            fold_scores=fold_scores,
            mean_score=float(np.mean(fold_scores)),
            std_score=float(np.std(fold_scores)),
            fold_metrics=fold_metrics,
        )

        return self.cv_results

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    def get_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "is_trained": self._is_trained,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "cv_results": self.cv_results.to_dict() if self.cv_results else None,
            "feature_importance": self.feature_importance,
        }
