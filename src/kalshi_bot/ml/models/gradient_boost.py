"""Gradient boosting model using LightGBM."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from kalshi_bot.ml.models.base import BasePredictionModel, ModelMetrics
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class GradientBoostModel(BasePredictionModel):
    """
    LightGBM gradient boosting model for market outcome prediction.

    Advantages over logistic regression:
    - Captures non-linear relationships
    - Handles feature interactions automatically
    - Better performance with more training data
    - Built-in feature importance

    Recommended for use after collecting 1000+ settled markets.
    """

    def __init__(
        self,
        model_id: str | None = None,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
    ) -> None:
        """
        Initialize gradient boosting model.

        Args:
            model_id: Unique model identifier
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            min_samples_leaf: Minimum samples per leaf
        """
        super().__init__(model_id)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._min_samples_leaf = min_samples_leaf

        self._model = None
        self._feature_names: list[str] = []
        self._optimal_threshold: float = 0.5

    @property
    def model_type(self) -> str:
        """Return model type identifier."""
        return "gradient_boost"

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> ModelMetrics:
        """
        Train the gradient boosting model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (1 for YES, 0 for NO)
            feature_names: Optional feature names for interpretation

        Returns:
            Training metrics
        """
        try:
            import lightgbm as lgb
            # Try to actually use LightGBM to verify it works
            _ = lgb.Dataset(X[:10], label=y[:10])
        except (ImportError, OSError) as e:
            logger.warning(f"LightGBM unavailable ({e}). Using sklearn GradientBoosting.")
            return self._train_sklearn(X, y, feature_names)

        logger.info(f"Training LightGBM model on {len(y)} samples")

        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Calculate class weights for imbalanced data
        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Create dataset
        train_data = lgb.Dataset(
            X, label=y,
            feature_name=self._feature_names,
        )

        # Training parameters
        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "auc"],
            "boosting_type": "gbdt",
            "num_leaves": 2 ** self._max_depth - 1,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "min_data_in_leaf": self._min_samples_leaf,
            "scale_pos_weight": scale_pos_weight,
            "verbose": -1,
            "seed": 42,
        }

        # Train model
        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=self._n_estimators,
        )

        # Compute feature importance
        self._compute_feature_importance()

        # Get predictions for metrics
        y_proba = self._model.predict(X)
        y_pred = (y_proba >= 0.5).astype(int)

        # Compute metrics (this finds optimal threshold automatically)
        self.metrics = self._compute_metrics(y, y_pred, y_proba)
        self._optimal_threshold = self.metrics.optimal_threshold
        self.trained_at = datetime.utcnow()
        self._is_trained = True

        logger.info(
            f"Model trained: F1(YES)={self.metrics.f1_yes:.3f}, "
            f"balanced_acc={self.metrics.balanced_accuracy:.3f}, "
            f"AUC={self.metrics.auc:.3f}, optimal_threshold={self._optimal_threshold:.2f}"
        )

        return self.metrics

    def _train_sklearn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> ModelMetrics:
        """Fallback to sklearn GradientBoosting if LightGBM unavailable."""
        from sklearn.ensemble import GradientBoostingClassifier

        logger.info(f"Training sklearn GradientBoosting on {len(y)} samples")

        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        self._model = GradientBoostingClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            min_samples_leaf=self._min_samples_leaf,
            random_state=42,
        )
        self._model.fit(X, y)

        # Compute feature importance
        self.feature_importance = {
            name: float(imp)
            for name, imp in zip(self._feature_names, self._model.feature_importances_)
        }

        # Get predictions
        y_proba = self._model.predict_proba(X)[:, 1]
        y_pred = self._model.predict(X)

        # Compute metrics (this finds optimal threshold automatically)
        self.metrics = self._compute_metrics(y, y_pred, y_proba)
        self._optimal_threshold = self.metrics.optimal_threshold
        self.trained_at = datetime.utcnow()
        self._is_trained = True

        logger.info(
            f"Model trained: F1(YES)={self.metrics.f1_yes:.3f}, "
            f"balanced_acc={self.metrics.balanced_accuracy:.3f}, "
            f"AUC={self.metrics.auc:.3f}, optimal_threshold={self._optimal_threshold:.2f}"
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
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before prediction")

        try:
            import lightgbm as lgb
            if isinstance(self._model, lgb.Booster):
                return self._model.predict(X)
        except (ImportError, OSError):
            pass

        # sklearn fallback
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Predict binary outcomes using optimal threshold.

        Args:
            X: Feature matrix
            threshold: Override threshold (uses optimal if None)

        Returns:
            Array of predictions (1 for YES, 0 for NO)
        """
        if threshold is None:
            threshold = self._optimal_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @property
    def optimal_threshold(self) -> float:
        """Get the optimal classification threshold."""
        return self._optimal_threshold

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
            "feature_names": self._feature_names,
            "model_id": self.model_id,
            "trained_at": self.trained_at,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "cv_results": self.cv_results.to_dict() if self.cv_results else None,
            "feature_importance": self.feature_importance,
            "optimal_threshold": self._optimal_threshold,
            "params": {
                "n_estimators": self._n_estimators,
                "max_depth": self._max_depth,
                "learning_rate": self._learning_rate,
                "min_samples_leaf": self._min_samples_leaf,
            },
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
        self._feature_names = model_data["feature_names"]
        self.model_id = model_data["model_id"]
        self.trained_at = model_data["trained_at"]
        self.feature_importance = model_data.get("feature_importance", {})
        self._optimal_threshold = model_data.get("optimal_threshold", 0.5)

        params = model_data.get("params", {})
        self._n_estimators = params.get("n_estimators", 100)
        self._max_depth = params.get("max_depth", 5)
        self._learning_rate = params.get("learning_rate", 0.1)
        self._min_samples_leaf = params.get("min_samples_leaf", 20)

        if model_data.get("metrics"):
            from kalshi_bot.ml.models.base import ModelMetrics
            self.metrics = ModelMetrics.from_dict(model_data["metrics"])

        self._is_trained = True
        logger.info(f"Model loaded from {path}")

    def _compute_feature_importance(self) -> None:
        """Compute feature importance from model."""
        if self._model is None:
            return

        try:
            import lightgbm as lgb
            if isinstance(self._model, lgb.Booster):
                importance = self._model.feature_importance(importance_type="gain")
                total = np.sum(importance)
                if total > 0:
                    normalized = importance / total
                else:
                    normalized = np.zeros_like(importance)

                self.feature_importance = {
                    name: float(imp)
                    for name, imp in zip(self._feature_names, normalized)
                }
                return
        except (ImportError, OSError, AttributeError):
            pass

        # sklearn fallback
        if hasattr(self._model, "feature_importances_"):
            self.feature_importance = {
                name: float(imp)
                for name, imp in zip(self._feature_names, self._model.feature_importances_)
            }

    def get_tree_info(self) -> dict:
        """Get information about the tree structure."""
        if self._model is None:
            return {}

        try:
            import lightgbm as lgb
            if isinstance(self._model, lgb.Booster):
                return {
                    "num_trees": self._model.num_trees(),
                    "feature_names": self._feature_names,
                }
        except (ImportError, OSError, AttributeError):
            pass

        if hasattr(self._model, "n_estimators"):
            return {
                "num_trees": self._model.n_estimators,
                "feature_names": self._feature_names,
            }

        return {}
