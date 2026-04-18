"""Economic Data Agent - trades economic markets using FRED API data."""

from kalshi_bot.agents.economic.agent import EconomicDataAgent
from kalshi_bot.agents.economic.calendar import EconomicCalendar
from kalshi_bot.agents.economic.fred_client import FREDClient
from kalshi_bot.agents.economic.indicator_model import IndicatorModel

__all__ = ["EconomicDataAgent", "EconomicCalendar", "FREDClient", "IndicatorModel"]
