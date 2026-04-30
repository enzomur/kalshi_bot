"""Trading strategies that emit Signal objects.

IMPORTANT: Strategies MUST NOT import from execution/.
They emit Signals only - the Risk Engine is the sole gatekeeper to execution.
"""

from src.strategies.base import Strategy, StrategyStatus
from src.strategies.weather import WeatherStrategy
from src.strategies.calibration import CalibrationStrategy
from src.strategies.arbitrage import ArbitrageStrategy
from src.strategies.market_make import MarketMakingStrategy
from src.strategies.mean_reversion import MeanReversionStrategy

__all__ = [
    "Strategy",
    "StrategyStatus",
    "WeatherStrategy",
    "CalibrationStrategy",
    "ArbitrageStrategy",
    "MarketMakingStrategy",
    "MeanReversionStrategy",
]
