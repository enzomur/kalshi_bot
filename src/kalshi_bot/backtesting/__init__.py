"""Backtesting module for replaying historical data and simulating trades."""

from kalshi_bot.backtesting.data_loader import HistoricalDataLoader
from kalshi_bot.backtesting.engine import BacktestEngine
from kalshi_bot.backtesting.metrics import BacktestMetrics
from kalshi_bot.backtesting.position_tracker import PositionTracker
from kalshi_bot.backtesting.report import ReportGenerator
from kalshi_bot.backtesting.simulator import TradeSimulator

__all__ = [
    "BacktestEngine",
    "HistoricalDataLoader",
    "TradeSimulator",
    "PositionTracker",
    "BacktestMetrics",
    "ReportGenerator",
]
