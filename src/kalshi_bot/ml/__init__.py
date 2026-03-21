"""Machine Learning module for self-learning trading bot."""

from kalshi_bot.ml.data_collector import MarketSnapshotCollector
from kalshi_bot.ml.outcome_tracker import OutcomeTracker
from kalshi_bot.ml.feature_engineer import FeatureEngineer, MarketFeatures, FEATURE_NAMES
from kalshi_bot.ml.historical_backfill import HistoricalBackfiller, BackfillResult

__all__ = [
    # Data collection
    "MarketSnapshotCollector",
    "OutcomeTracker",
    # Feature engineering
    "FeatureEngineer",
    "MarketFeatures",
    "FEATURE_NAMES",
    # Historical backfill
    "HistoricalBackfiller",
    "BackfillResult",
]
