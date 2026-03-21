"""Arbitrage detection module."""

from kalshi_bot.arbitrage.detector import ArbitrageDetector
from kalshi_bot.arbitrage.strategies.base import ArbitrageStrategy
from kalshi_bot.arbitrage.strategies.cross_market import CrossMarketStrategy
from kalshi_bot.arbitrage.strategies.multi_outcome import MultiOutcomeStrategy
from kalshi_bot.arbitrage.strategies.single_market import SingleMarketStrategy

__all__ = [
    "ArbitrageDetector",
    "ArbitrageStrategy",
    "SingleMarketStrategy",
    "MultiOutcomeStrategy",
    "CrossMarketStrategy",
]
