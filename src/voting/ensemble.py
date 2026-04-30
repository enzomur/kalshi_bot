"""Ensemble voting system for combining strategy signals.

The voting system aggregates signals from multiple strategies and produces
TradeIntent objects for the Risk Engine. Strategies are weighted, and signals
must meet consensus thresholds before being forwarded.

Key rules:
- Weighted vote: final_edge = Σ(strategy.weight × signal.edge × signal.confidence)
- Agreement filter: >=2 strategies must agree on direction (configurable)
- Minimum edge threshold: 3pp after fees (configurable)
- Single-strategy mode: when only one strategy is active, passes through directly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from src.core.types import Signal
from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradeIntent:
    """A consolidated trading intent from the voting system.

    This is the output of the voting process - multiple signals have been
    aggregated and consensus reached. The Risk Engine uses this to make
    final sizing and execution decisions.
    """

    intent_id: str
    market_ticker: str
    direction: str  # "yes" or "no"
    final_edge: float  # Weighted, aggregated edge
    final_confidence: float  # Weighted average confidence
    contributing_signals: list[Signal]
    strategy_votes: dict[str, float]  # strategy_name -> weighted vote
    category: str | None = None  # Market category for correlation limits
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def signal_count(self) -> int:
        """Number of strategies that contributed to this intent."""
        return len(self.contributing_signals)

    @property
    def primary_signal(self) -> Signal:
        """Get the primary (highest confidence) contributing signal."""
        return max(self.contributing_signals, key=lambda s: s.confidence)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent_id": self.intent_id,
            "market_ticker": self.market_ticker,
            "direction": self.direction,
            "final_edge": self.final_edge,
            "final_confidence": self.final_confidence,
            "signal_count": self.signal_count,
            "strategy_votes": self.strategy_votes,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "signal_ids": [s.signal_id for s in self.contributing_signals],
        }


@dataclass
class StrategyConfig:
    """Configuration for a strategy in the ensemble."""

    name: str
    weight: float = 1.0
    enabled: bool = True


class VotingEnsemble:
    """Aggregates signals from multiple strategies using weighted voting.

    The ensemble collects signals, groups them by market, applies weighted
    voting, and produces TradeIntent objects for markets that meet consensus
    thresholds.
    """

    # Default configuration
    DEFAULT_MIN_EDGE = 0.03  # 3% minimum edge after aggregation
    DEFAULT_MIN_CONFIDENCE = 0.50  # 50% minimum confidence
    DEFAULT_MIN_AGREEMENT = 1  # Minimum strategies agreeing (1 = single-strategy mode)
    DEFAULT_FEE_ADJUSTMENT = 0.01  # 1% fee adjustment (Kalshi fees)

    def __init__(
        self,
        strategy_weights: dict[str, float] | None = None,
        min_edge: float = DEFAULT_MIN_EDGE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        min_agreement: int = DEFAULT_MIN_AGREEMENT,
        fee_adjustment: float = DEFAULT_FEE_ADJUSTMENT,
    ) -> None:
        """
        Initialize the voting ensemble.

        Args:
            strategy_weights: Mapping of strategy name -> weight (default 1.0)
            min_edge: Minimum weighted edge to create intent (after fees)
            min_confidence: Minimum weighted confidence to create intent
            min_agreement: Minimum number of strategies agreeing on direction
            fee_adjustment: Percentage to subtract from edge for fees
        """
        self._strategy_weights = strategy_weights or {}
        self._min_edge = min_edge
        self._min_confidence = min_confidence
        self._min_agreement = min_agreement
        self._fee_adjustment = fee_adjustment

        # Track statistics
        self._signals_processed = 0
        self._intents_generated = 0

    def get_strategy_weight(self, strategy_name: str) -> float:
        """Get the weight for a strategy (default 1.0)."""
        return self._strategy_weights.get(strategy_name, 1.0)

    def set_strategy_weight(self, strategy_name: str, weight: float) -> None:
        """Set the weight for a strategy."""
        self._strategy_weights[strategy_name] = weight

    def aggregate_signals(self, signals: list[Signal]) -> list[TradeIntent]:
        """
        Aggregate signals from multiple strategies into trade intents.

        This is the main voting logic:
        1. Group signals by market ticker
        2. For each market, separate YES and NO signals
        3. Calculate weighted edge for each direction
        4. Check agreement threshold
        5. Check minimum edge threshold (after fees)
        6. Create TradeIntent if thresholds met

        Args:
            signals: List of Signal objects from strategies

        Returns:
            List of TradeIntent objects that met voting thresholds
        """
        if not signals:
            return []

        self._signals_processed += len(signals)

        # Filter out expired signals
        active_signals = [s for s in signals if not s.is_expired]
        if len(active_signals) < len(signals):
            expired_count = len(signals) - len(active_signals)
            logger.debug(f"Filtered {expired_count} expired signals")

        if not active_signals:
            return []

        # Group signals by market
        by_market: dict[str, list[Signal]] = {}
        for signal in active_signals:
            if signal.market_ticker not in by_market:
                by_market[signal.market_ticker] = []
            by_market[signal.market_ticker].append(signal)

        # Process each market
        intents: list[TradeIntent] = []
        for market_ticker, market_signals in by_market.items():
            intent = self._process_market(market_ticker, market_signals)
            if intent:
                intents.append(intent)
                self._intents_generated += 1

        if intents:
            logger.info(
                f"Voting produced {len(intents)} intents from {len(signals)} signals"
            )

        return intents

    def _process_market(
        self, market_ticker: str, signals: list[Signal]
    ) -> TradeIntent | None:
        """
        Process signals for a single market and produce intent if thresholds met.

        Args:
            market_ticker: The market being evaluated
            signals: All signals for this market

        Returns:
            TradeIntent if thresholds met, None otherwise
        """
        # Separate by direction
        yes_signals = [s for s in signals if s.direction == "yes"]
        no_signals = [s for s in signals if s.direction == "no"]

        # Calculate weighted edges for each direction
        yes_vote = self._calculate_weighted_vote(yes_signals)
        no_vote = self._calculate_weighted_vote(no_signals)

        # Determine winning direction
        if yes_vote["weighted_edge"] >= no_vote["weighted_edge"]:
            winning_direction = "yes"
            winning_vote = yes_vote
            winning_signals = yes_signals
        else:
            winning_direction = "no"
            winning_vote = no_vote
            winning_signals = no_signals

        # Check agreement threshold
        agreement_count = len(winning_signals)
        if agreement_count < self._min_agreement:
            logger.debug(
                f"{market_ticker}: insufficient agreement "
                f"({agreement_count} < {self._min_agreement})"
            )
            return None

        # Adjust edge for fees
        edge_after_fees = winning_vote["weighted_edge"] - self._fee_adjustment

        # Check edge threshold
        if edge_after_fees < self._min_edge:
            logger.debug(
                f"{market_ticker}: edge too low after fees "
                f"({edge_after_fees:.2%} < {self._min_edge:.2%})"
            )
            return None

        # Check confidence threshold
        if winning_vote["weighted_confidence"] < self._min_confidence:
            logger.debug(
                f"{market_ticker}: confidence too low "
                f"({winning_vote['weighted_confidence']:.2%} < {self._min_confidence:.2%})"
            )
            return None

        # Create trade intent
        intent = TradeIntent(
            intent_id=str(uuid.uuid4()),
            market_ticker=market_ticker,
            direction=winning_direction,
            final_edge=edge_after_fees,
            final_confidence=winning_vote["weighted_confidence"],
            contributing_signals=winning_signals,
            strategy_votes=winning_vote["strategy_votes"],
            category=self._extract_category(market_ticker),
        )

        logger.info(
            f"Vote passed for {market_ticker}: {winning_direction} "
            f"(edge={edge_after_fees:.2%}, confidence={winning_vote['weighted_confidence']:.2%}, "
            f"agreement={agreement_count})"
        )

        return intent

    def _calculate_weighted_vote(self, signals: list[Signal]) -> dict[str, Any]:
        """
        Calculate weighted vote statistics for a set of signals.

        Weighted edge formula:
            final_edge = Σ(weight_i × edge_i × confidence_i) / Σ(weight_i × confidence_i)

        This gives more weight to higher-confidence signals from higher-weighted strategies.

        Args:
            signals: Signals to aggregate (all same direction)

        Returns:
            Dictionary with weighted_edge, weighted_confidence, strategy_votes
        """
        if not signals:
            return {
                "weighted_edge": 0.0,
                "weighted_confidence": 0.0,
                "strategy_votes": {},
            }

        total_weight = 0.0
        weighted_edge_sum = 0.0
        weighted_confidence_sum = 0.0
        strategy_votes: dict[str, float] = {}

        for signal in signals:
            weight = self.get_strategy_weight(signal.strategy_name)
            vote_power = weight * signal.confidence

            weighted_edge_sum += weight * signal.edge * signal.confidence
            weighted_confidence_sum += vote_power
            total_weight += vote_power

            # Track per-strategy votes
            strategy_votes[signal.strategy_name] = weight * signal.edge * signal.confidence

        if total_weight == 0:
            return {
                "weighted_edge": 0.0,
                "weighted_confidence": 0.0,
                "strategy_votes": strategy_votes,
            }

        weighted_edge = weighted_edge_sum / total_weight
        weighted_confidence = weighted_confidence_sum / len(signals)

        return {
            "weighted_edge": weighted_edge,
            "weighted_confidence": weighted_confidence,
            "strategy_votes": strategy_votes,
        }

    def _extract_category(self, market_ticker: str) -> str | None:
        """
        Extract market category from ticker for correlation tracking.

        Categories are used by the Risk Engine to enforce category exposure limits.
        """
        ticker_upper = market_ticker.upper()

        # Weather markets
        if any(prefix in ticker_upper for prefix in ["KXHIGH", "KXLOW", "KXRAIN"]):
            return "weather"

        # Fed/economic markets
        if any(prefix in ticker_upper for prefix in ["FED", "FOMC", "CPI", "GDP"]):
            return "economic"

        # Sports markets (various prefixes)
        if any(prefix in ticker_upper for prefix in ["NFL", "NBA", "MLB", "NHL", "SOCCER"]):
            return "sports"

        # Politics
        if any(prefix in ticker_upper for prefix in ["PRES", "ELECT", "VOTE"]):
            return "politics"

        return None

    def process_single_signal(self, signal: Signal) -> TradeIntent | None:
        """
        Process a single signal directly (single-strategy mode).

        This bypasses multi-strategy voting when only one strategy is active.
        Still applies edge and confidence thresholds.

        Args:
            signal: Single signal to process

        Returns:
            TradeIntent if thresholds met, None otherwise
        """
        if signal.is_expired:
            logger.debug(f"Signal {signal.signal_id[:8]} expired")
            return None

        self._signals_processed += 1

        weight = self.get_strategy_weight(signal.strategy_name)
        weighted_edge = weight * signal.edge * signal.confidence
        edge_after_fees = weighted_edge - self._fee_adjustment

        if edge_after_fees < self._min_edge:
            logger.debug(
                f"Single signal edge too low: {edge_after_fees:.2%} < {self._min_edge:.2%}"
            )
            return None

        if signal.confidence < self._min_confidence:
            logger.debug(
                f"Single signal confidence too low: {signal.confidence:.2%}"
            )
            return None

        intent = TradeIntent(
            intent_id=str(uuid.uuid4()),
            market_ticker=signal.market_ticker,
            direction=signal.direction,
            final_edge=edge_after_fees,
            final_confidence=signal.confidence,
            contributing_signals=[signal],
            strategy_votes={signal.strategy_name: weighted_edge},
            category=self._extract_category(signal.market_ticker),
        )

        self._intents_generated += 1

        logger.info(
            f"Single-strategy intent for {signal.market_ticker}: {signal.direction} "
            f"(edge={edge_after_fees:.2%}, confidence={signal.confidence:.2%})"
        )

        return intent

    def get_status(self) -> dict[str, Any]:
        """Get voting system status."""
        return {
            "strategy_weights": self._strategy_weights,
            "config": {
                "min_edge": self._min_edge,
                "min_confidence": self._min_confidence,
                "min_agreement": self._min_agreement,
                "fee_adjustment": self._fee_adjustment,
            },
            "stats": {
                "signals_processed": self._signals_processed,
                "intents_generated": self._intents_generated,
            },
        }

    def set_multi_strategy_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable multi-strategy mode.

        In multi-strategy mode, at least 2 strategies must agree on direction.
        In single-strategy mode, 1 strategy is sufficient.

        Args:
            enabled: True for multi-strategy (min_agreement=2), False for single (min_agreement=1)
        """
        self._min_agreement = 2 if enabled else 1
        logger.info(f"Voting mode: {'multi-strategy' if enabled else 'single-strategy'} (min_agreement={self._min_agreement})")

    @property
    def is_multi_strategy_mode(self) -> bool:
        """Check if multi-strategy mode is enabled."""
        return self._min_agreement >= 2

    def get_agreement_summary(self, signals: list[Signal]) -> dict[str, Any]:
        """
        Get summary of strategy agreement for debugging.

        Args:
            signals: List of signals to analyze

        Returns:
            Summary of which strategies agree/disagree
        """
        by_market: dict[str, dict[str, list[str]]] = {}

        for signal in signals:
            ticker = signal.market_ticker
            if ticker not in by_market:
                by_market[ticker] = {"yes": [], "no": []}
            by_market[ticker][signal.direction].append(signal.strategy_name)

        return {
            ticker: {
                "yes_strategies": data["yes"],
                "no_strategies": data["no"],
                "agreement": len(data["yes"]) >= self._min_agreement or len(data["no"]) >= self._min_agreement,
            }
            for ticker, data in by_market.items()
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> VotingEnsemble:
        """
        Create a VotingEnsemble from configuration dictionary.

        Expected config structure:
        {
            "strategies": {
                "strategy_name": {"weight": 1.0, "enabled": true},
                ...
            },
            "min_edge": 0.03,
            "min_confidence": 0.50,
            "min_agreement": 2,
            "fee_adjustment": 0.01
        }

        Args:
            config: Configuration dictionary

        Returns:
            Configured VotingEnsemble instance
        """
        # Extract strategy weights
        strategy_weights: dict[str, float] = {}
        strategies_config = config.get("strategies", {})
        for name, strategy_config in strategies_config.items():
            if strategy_config.get("enabled", True):
                strategy_weights[name] = strategy_config.get("weight", 1.0)

        # Get voting config
        voting_config = config.get("voting", {})

        return cls(
            strategy_weights=strategy_weights,
            min_edge=voting_config.get("min_edge", config.get("min_edge", cls.DEFAULT_MIN_EDGE)),
            min_confidence=voting_config.get("min_confidence", config.get("min_confidence", cls.DEFAULT_MIN_CONFIDENCE)),
            min_agreement=voting_config.get("min_strategies", config.get("min_agreement", cls.DEFAULT_MIN_AGREEMENT)),
            fee_adjustment=voting_config.get("fee_adjustment", config.get("fee_adjustment", cls.DEFAULT_FEE_ADJUSTMENT)),
        )
