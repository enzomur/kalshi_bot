"""Data layer for external API access."""

from src.data.odds_feed import OddsFeed, Sport, GameOdds
from src.data.polymarket_client import (
    PolymarketClient,
    PolymarketMarket,
    ArbitrageOpportunity,
)

__all__ = [
    # Odds API
    "OddsFeed",
    "Sport",
    "GameOdds",
    # Polymarket
    "PolymarketClient",
    "PolymarketMarket",
    "ArbitrageOpportunity",
]
