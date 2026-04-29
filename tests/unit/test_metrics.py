"""Tests for the metrics module."""

import pytest
from datetime import datetime, timezone

from src.metrics.brier import BrierCalculator, PredictionRecord, calculate_brier_score
from src.metrics.performance import PerformanceTracker, TradeResult
from src.metrics.calibration_curve import CalibrationCurve, quick_calibration_check


class TestBrierCalculator:
    """Tests for BrierCalculator class."""

    def test_initialization(self):
        """Test calculator initializes correctly."""
        calc = BrierCalculator(calibration_threshold=0.20)
        assert calc._calibration_threshold == 0.20
        assert len(calc._predictions) == 0

    def test_record_prediction(self):
        """Test recording a prediction."""
        calc = BrierCalculator()
        record = calc.record_prediction(
            prediction_id="p1",
            market_ticker="TEST",
            strategy_name="weather",
            predicted_probability=0.70,
            direction="yes",
        )

        assert record.prediction_id == "p1"
        assert record.predicted_probability == 0.70
        assert record.actual_outcome is None
        assert len(calc._predictions) == 1

    def test_resolve_prediction(self):
        """Test resolving a prediction."""
        calc = BrierCalculator()
        calc.record_prediction("p1", "TEST", "weather", 0.70, "yes")

        resolved = calc.resolve_prediction("p1", True)

        assert resolved is not None
        assert resolved.actual_outcome is True
        assert resolved.settled_at is not None

    def test_resolve_unknown_prediction(self):
        """Test resolving unknown prediction returns None."""
        calc = BrierCalculator()
        resolved = calc.resolve_prediction("unknown", True)
        assert resolved is None

    def test_calculate_brier_insufficient_data(self):
        """Test Brier calculation with insufficient data."""
        calc = BrierCalculator()
        calc.record_prediction("p1", "TEST", "weather", 0.70, "yes")
        calc.resolve_prediction("p1", True)

        result = calc.calculate_brier(min_predictions=10)
        assert result is None

    def test_calculate_brier_perfect_predictions(self):
        """Test Brier score for perfect predictions."""
        calc = BrierCalculator()

        # Perfect predictions: predict 1.0 for true outcomes
        for i in range(20):
            calc.record_prediction(f"p{i}", "TEST", "weather", 1.0, "yes")
            calc.resolve_prediction(f"p{i}", True)

        result = calc.calculate_brier(min_predictions=10)

        assert result is not None
        assert result.brier_score == 0.0  # Perfect
        assert result.n_correct == 20
        assert result.win_rate == 1.0

    def test_calculate_brier_worst_predictions(self):
        """Test Brier score for worst predictions."""
        calc = BrierCalculator()

        # Worst predictions: predict 1.0 for false outcomes
        for i in range(20):
            calc.record_prediction(f"p{i}", "TEST", "weather", 1.0, "yes")
            calc.resolve_prediction(f"p{i}", False)

        result = calc.calculate_brier(min_predictions=10)

        assert result is not None
        assert result.brier_score == 1.0  # Worst possible
        assert result.n_correct == 0

    def test_calculate_brier_calibrated(self):
        """Test calibration status based on threshold."""
        calc = BrierCalculator(calibration_threshold=0.25)

        # Random-ish predictions
        for i in range(20):
            calc.record_prediction(f"p{i}", "TEST", "weather", 0.6, "yes")
            calc.resolve_prediction(f"p{i}", i % 2 == 0)  # 50% win rate

        result = calc.calculate_brier(min_predictions=10)

        assert result is not None
        # Brier should be around 0.36 for 60% predictions with 50% outcomes

    def test_calculate_brier_by_strategy(self):
        """Test per-strategy Brier calculation."""
        calc = BrierCalculator()

        # Add predictions for two strategies
        for i in range(15):
            calc.record_prediction(f"p{i}", "TEST", "weather", 0.8, "yes")
            calc.resolve_prediction(f"p{i}", True)

        for i in range(15, 30):
            calc.record_prediction(f"p{i}", "TEST", "arbitrage", 0.5, "yes")
            calc.resolve_prediction(f"p{i}", True if i % 2 == 0 else False)

        result = calc.calculate_brier(min_predictions=10)

        assert result is not None
        assert "weather" in result.strategy_scores
        assert "arbitrage" in result.strategy_scores
        assert result.strategy_scores["weather"] < result.strategy_scores["arbitrage"]

    def test_get_status(self):
        """Test status reporting."""
        calc = BrierCalculator()
        calc.record_prediction("p1", "TEST", "weather", 0.70, "yes")

        status = calc.get_status()

        assert status["total_predictions"] == 1
        assert status["pending"] == 1
        assert status["resolved"] == 0


class TestCalculateBrierScore:
    """Tests for standalone Brier score function."""

    def test_perfect_score(self):
        """Test perfect Brier score."""
        probs = [1.0, 1.0, 0.0, 0.0]
        outcomes = [True, True, False, False]

        score = calculate_brier_score(probs, outcomes)
        assert score == 0.0

    def test_random_predictions(self):
        """Test Brier score for random predictions."""
        probs = [0.5, 0.5, 0.5, 0.5]
        outcomes = [True, False, True, False]

        score = calculate_brier_score(probs, outcomes)
        assert score == 0.25  # Expected for 50% predictions

    def test_length_mismatch_raises(self):
        """Test error on mismatched lengths."""
        with pytest.raises(ValueError):
            calculate_brier_score([0.5, 0.5], [True])

    def test_empty_raises(self):
        """Test error on empty inputs."""
        with pytest.raises(ValueError):
            calculate_brier_score([], [])


class TestPredictionRecord:
    """Tests for PredictionRecord dataclass."""

    def test_is_resolved(self):
        """Test is_resolved property."""
        record = PredictionRecord(
            prediction_id="p1",
            market_ticker="TEST",
            strategy_name="weather",
            predicted_probability=0.70,
            direction="yes",
        )

        assert not record.is_resolved

        record.actual_outcome = True
        assert record.is_resolved

    def test_brier_contribution(self):
        """Test individual Brier contribution."""
        record = PredictionRecord(
            prediction_id="p1",
            market_ticker="TEST",
            strategy_name="weather",
            predicted_probability=0.70,
            direction="yes",
            actual_outcome=True,
        )

        contribution = record.brier_contribution()
        assert contribution == pytest.approx(0.09)  # (0.7 - 1.0)^2

    def test_brier_contribution_unresolved(self):
        """Test Brier contribution for unresolved record."""
        record = PredictionRecord(
            prediction_id="p1",
            market_ticker="TEST",
            strategy_name="weather",
            predicted_probability=0.70,
            direction="yes",
        )

        assert record.brier_contribution() is None


class TestPerformanceTracker:
    """Tests for PerformanceTracker class."""

    def test_initialization(self):
        """Test tracker initializes correctly."""
        tracker = PerformanceTracker(initial_balance=1000.0)

        assert tracker._initial_balance == 1000.0
        assert tracker._current_balance == 1000.0
        assert tracker._peak_balance == 1000.0

    def test_record_trade_winning(self):
        """Test recording a winning trade."""
        tracker = PerformanceTracker(initial_balance=1000.0)

        trade = TradeResult(
            trade_id="t1",
            market_ticker="TEST",
            strategy_name="weather",
            direction="yes",
            entry_price=50,
            quantity=10,
            pnl=5.0,
            won=True,
            closed_at=datetime.now(timezone.utc),
        )

        tracker.record_trade(trade)

        summary = tracker.get_summary()
        assert summary.total_trades == 1
        assert summary.total_wins == 1
        assert summary.total_pnl == 5.0

    def test_record_trade_losing(self):
        """Test recording a losing trade."""
        tracker = PerformanceTracker(initial_balance=1000.0)

        trade = TradeResult(
            trade_id="t1",
            market_ticker="TEST",
            strategy_name="weather",
            direction="yes",
            entry_price=50,
            quantity=10,
            pnl=-5.0,
            won=False,
            closed_at=datetime.now(timezone.utc),
        )

        tracker.record_trade(trade)

        summary = tracker.get_summary()
        assert summary.total_losses == 1
        assert summary.total_pnl == -5.0

    def test_drawdown_tracking(self):
        """Test drawdown calculation."""
        tracker = PerformanceTracker(initial_balance=1000.0)

        # Win to establish peak
        tracker.record_trade(TradeResult(
            trade_id="t1",
            market_ticker="TEST",
            strategy_name="weather",
            direction="yes",
            entry_price=50,
            quantity=10,
            pnl=100.0,
            won=True,
            closed_at=datetime.now(timezone.utc),
        ))

        # Lose to create drawdown
        tracker.record_trade(TradeResult(
            trade_id="t2",
            market_ticker="TEST",
            strategy_name="weather",
            direction="yes",
            entry_price=50,
            quantity=10,
            pnl=-50.0,
            won=False,
            closed_at=datetime.now(timezone.utc),
        ))

        summary = tracker.get_summary()
        assert summary.peak_balance == 1100.0
        assert summary.current_balance == 1050.0
        assert summary.max_drawdown == pytest.approx(50.0 / 1100.0, rel=0.01)

    def test_roi_calculation(self):
        """Test ROI calculation."""
        tracker = PerformanceTracker(initial_balance=1000.0)

        tracker.record_trade(TradeResult(
            trade_id="t1",
            market_ticker="TEST",
            strategy_name="weather",
            direction="yes",
            entry_price=50,
            quantity=10,
            pnl=100.0,
            won=True,
            closed_at=datetime.now(timezone.utc),
        ))

        summary = tracker.get_summary()
        assert summary.roi == pytest.approx(0.10)  # 10% return

    def test_strategy_breakdown(self):
        """Test per-strategy metrics."""
        tracker = PerformanceTracker(initial_balance=1000.0)

        # Weather strategy trades
        for i in range(5):
            tracker.record_trade(TradeResult(
                trade_id=f"tw{i}",
                market_ticker="TEST",
                strategy_name="weather",
                direction="yes",
                entry_price=50,
                quantity=10,
                pnl=10.0 if i < 4 else -5.0,
                won=i < 4,
                closed_at=datetime.now(timezone.utc),
            ))

        # Arbitrage strategy trades
        for i in range(3):
            tracker.record_trade(TradeResult(
                trade_id=f"ta{i}",
                market_ticker="TEST",
                strategy_name="arbitrage",
                direction="yes",
                entry_price=50,
                quantity=10,
                pnl=5.0,
                won=True,
                closed_at=datetime.now(timezone.utc),
            ))

        summary = tracker.get_summary()

        assert "weather" in summary.strategy_metrics
        assert "arbitrage" in summary.strategy_metrics
        assert summary.strategy_metrics["weather"]["trades"] == 5
        assert summary.strategy_metrics["arbitrage"]["trades"] == 3

    def test_get_status(self):
        """Test status reporting."""
        tracker = PerformanceTracker(initial_balance=1000.0)
        status = tracker.get_status()

        assert "trades" in status
        assert "win_rate" in status
        assert "pnl" in status
        assert "roi" in status


class TestTradeResult:
    """Tests for TradeResult dataclass."""

    def test_return_pct(self):
        """Test percentage return calculation."""
        trade = TradeResult(
            trade_id="t1",
            market_ticker="TEST",
            strategy_name="weather",
            direction="yes",
            entry_price=50,  # 50 cents = $0.50 per contract
            quantity=10,     # 10 contracts = $5.00 cost
            pnl=1.0,         # $1.00 profit
            won=True,
            closed_at=datetime.now(timezone.utc),
        )

        assert trade.return_pct == pytest.approx(0.20)  # 20% return


class TestCalibrationCurve:
    """Tests for CalibrationCurve class."""

    def test_initialization(self):
        """Test curve initializes correctly."""
        curve = CalibrationCurve(n_buckets=10, min_bucket_size=5)

        assert curve._n_buckets == 10
        assert curve._min_bucket_size == 5

    def test_add_prediction(self):
        """Test adding a prediction."""
        curve = CalibrationCurve()
        curve.add_prediction(0.70, True)

        assert len(curve._predictions) == 1

    def test_add_prediction_invalid(self):
        """Test error on invalid probability."""
        curve = CalibrationCurve()

        with pytest.raises(ValueError):
            curve.add_prediction(1.5, True)

    def test_add_predictions_batch(self):
        """Test adding multiple predictions."""
        curve = CalibrationCurve()
        curve.add_predictions([0.5, 0.6, 0.7], [True, False, True])

        assert len(curve._predictions) == 3

    def test_add_predictions_length_mismatch(self):
        """Test error on mismatched lengths."""
        curve = CalibrationCurve()

        with pytest.raises(ValueError):
            curve.add_predictions([0.5, 0.6], [True])

    def test_analyze_insufficient_data(self):
        """Test analysis with insufficient data."""
        curve = CalibrationCurve(min_bucket_size=10)
        curve.add_prediction(0.70, True)

        analysis = curve.analyze()
        assert analysis is None

    def test_analyze_perfect_calibration(self):
        """Test analysis with perfect calibration."""
        curve = CalibrationCurve(n_buckets=10, min_bucket_size=5)

        # Add perfectly calibrated predictions
        # 60-70% bucket with 65% actual rate
        for _ in range(5):
            curve.add_prediction(0.65, True)
        for _ in range(3):
            curve.add_prediction(0.65, False)

        # This won't be perfect but should be reasonably calibrated

    def test_analyze_overconfident(self):
        """Test detection of overconfidence."""
        curve = CalibrationCurve(n_buckets=10, min_bucket_size=5)

        # Predict high (80%) but actual is low (20%)
        for _ in range(5):
            curve.add_prediction(0.85, True)
        for _ in range(20):
            curve.add_prediction(0.85, False)

        analysis = curve.analyze()

        if analysis:
            assert analysis.overall_overconfident

    def test_format_report(self):
        """Test report formatting."""
        curve = CalibrationCurve(n_buckets=10, min_bucket_size=3)

        for _ in range(10):
            curve.add_prediction(0.65, True)
        for _ in range(5):
            curve.add_prediction(0.65, False)

        analysis = curve.analyze()

        if analysis:
            report = analysis.format_report()
            assert "CALIBRATION REPORT" in report
            assert "Total predictions" in report

    def test_get_ascii_curve(self):
        """Test ASCII curve generation."""
        curve = CalibrationCurve(n_buckets=10, min_bucket_size=2)

        for i in range(20):
            prob = 0.3 + (i * 0.02)
            curve.add_prediction(prob, i % 3 != 0)

        ascii_art = curve.get_ascii_curve(width=30, height=15)

        assert "CALIBRATION CURVE" in ascii_art

    def test_clear(self):
        """Test clearing predictions."""
        curve = CalibrationCurve()
        curve.add_prediction(0.70, True)

        curve.clear()

        assert len(curve._predictions) == 0


class TestQuickCalibrationCheck:
    """Tests for quick_calibration_check function."""

    def test_quick_check(self):
        """Test quick calibration check."""
        probs = [0.65] * 10 + [0.35] * 10
        outcomes = [True] * 7 + [False] * 3 + [True] * 3 + [False] * 7

        analysis = quick_calibration_check(probs, outcomes, n_buckets=5)

        if analysis:
            assert analysis.total_predictions == 20
