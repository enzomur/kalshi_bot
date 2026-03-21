"""Logistic regression model for probability prediction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from kalshi_bot.ml.models.base import BasePredictionModel, ModelMetrics
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class LogisticRegressionModel(BasePredictionModel):
    """
    Logistic regression model for market outcome prediction.

    Good baseline model with:
    - Calibrated probabilities (important for edge calculation)
    - Interpretable coefficients
    - Fast training and inference
    - Works well with limited data
    """

    def __init__(
        self,
        model_id: str | None = None,
        regularization: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        """
        Initialize logistic regression model.

        Args:
            model_id: Unique model identifier
            regularization: Inverse regularization strength (C parameter)
            max_iter: Maximum iterations for solver
        """
        super().__init__(model_id)
        self._regularization = regularization
        self._max_iter = max_iter

        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = []

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "logistic"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> ModelMetrics:
        """
        Train the logistic regression model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 for YES, 0 for NO)
            feature_names: Optional feature names for interpretation

        Returns:
            Training metrics
        """
        logger.info(f"Training logistic regression model on {len(y)} samples")

        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train model
        self._model = LogisticRegression(
            C=self._regularization,
            max_iter=self._max_iter,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
        )
        self._model.fit(X_scaled, y)

        # Compute feature importance from coefficients
        self._compute_feature_importance()

        # Get predictions for metrics
        y_proba = self._model.predict_proba(X_scaled)[:, 1]
        y_pred = self._model.predict(X_scaled)

        # Compute metrics
        self.metrics = self._compute_metrics(y, y_pred, y_proba)
        self.trained_at = datetime.utcnow()
        self._is_trained = True

        logger.info(
            f"Model trained: accuracy={self.metrics.accuracy:.3f}, "
            f"AUC={self.metrics.auc:.3f}, brier={self.metrics.brier_score:.3f}"
        )

        return self.metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of YES outcome.

        Args:
            X: Feature matrix

        Returns:
            P(YES) probabilities
        """
        if not self._is_trained or self._model is None or self._scaler is None:
            raise RuntimeError("Model must be trained before prediction")

        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "model_id": self.model_id,
            "trained_at": self.trained_at,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "cv_results": self.cv_results.to_dict() if self.cv_results else None,
            "feature_importance": self.feature_importance,
            "regularization": self._regularization,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to model file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        self._model = model_data["model"]
        self._scaler = model_data["scaler"]
        self._feature_names = model_data["feature_names"]
        self.model_id = model_data["model_id"]
        self.trained_at = model_data["trained_at"]
        self.feature_importance = model_data.get("feature_importance", {})
        self._regularization = model_data.get("regularization", 1.0)

        if model_data.get("metrics"):
            from kalshi_bot.ml.models.base import ModelMetrics
            self.metrics = ModelMetrics.from_dict(model_data["metrics"])

        self._is_trained = True
        logger.info(f"Model loaded from {path}")

    def _compute_feature_importance(self) -> None:
        """Compute feature importance from model coefficients."""
        if self._model is None:
            return

        # Use absolute coefficient values as importance
        coefficients = np.abs(self._model.coef_[0])
        total = np.sum(coefficients)

        if total > 0:
            normalized = coefficients / total
        else:
            normalized = np.zeros_like(coefficients)

        self.feature_importance = {
            name: float(imp)
            for name, imp in zip(self._feature_names, normalized)
        }

    def get_coefficients(self) -> dict[str, float]:
        """Get model coefficients (signed importance)."""
        if self._model is None:
            return {}

        return {
            name: float(coef)
            for name, coef in zip(self._feature_names, self._model.coef_[0])
        }

    def get_calibration_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve data.

        Args:
            X: Feature matrix
            y: True labels
            n_bins: Number of bins

        Returns:
            Tuple of (mean_predicted, fraction_positive) arrays
        """
        from sklearn.calibration import calibration_curve

        y_proba = self.predict_proba(X)
        fraction_pos, mean_predicted = calibration_curve(y, y_proba, n_bins=n_bins)
        return mean_predicted, fraction_pos
