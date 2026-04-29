"""Sportsbook odds feed from the-odds-api.com.

This module fetches consensus odds from major sportsbooks to provide
external probability estimates for sports markets. These can be compared
against Kalshi prices to find edge.

API Documentation: https://the-odds-api.com/liveapi/guides/v4/

Free tier: 500 requests/month
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any
from enum import Enum

import httpx

from src.observability.logging import get_logger

logger = get_logger(__name__)


class Sport(str, Enum):
    """Supported sports from the-odds-api."""

    # American sports
    NFL = "americanfootball_nfl"
    NCAA_FOOTBALL = "americanfootball_ncaaf"
    NBA = "basketball_nba"
    NCAA_BASKETBALL = "basketball_ncaab"
    MLB = "baseball_mlb"
    NHL = "icehockey_nhl"
    MLS = "soccer_usa_mls"

    # International
    EPL = "soccer_epl"  # English Premier League
    UEFA_CHAMPIONS = "soccer_uefa_champs_league"

    # Other
    UFC = "mma_mixed_martial_arts"
    BOXING = "boxing_boxing"


class OddsFormat(str, Enum):
    """Odds format options."""

    AMERICAN = "american"  # +150, -200
    DECIMAL = "decimal"  # 2.50, 1.50
    PROBABILITY = "probability"  # Internal conversion


@dataclass
class GameOdds:
    """Odds for a single game/event."""

    sport: str
    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime

    # Consensus probabilities (averaged across books)
    home_win_prob: float | None = None
    away_win_prob: float | None = None
    draw_prob: float | None = None  # For soccer

    # Spread/handicap
    spread_home: float | None = None
    spread_away: float | None = None
    spread_home_prob: float | None = None
    spread_away_prob: float | None = None

    # Totals (over/under)
    total_line: float | None = None
    over_prob: float | None = None
    under_prob: float | None = None

    # Metadata
    num_bookmakers: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sport": self.sport,
            "event_id": self.event_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "commence_time": self.commence_time.isoformat(),
            "home_win_prob": self.home_win_prob,
            "away_win_prob": self.away_win_prob,
            "draw_prob": self.draw_prob,
            "spread_home": self.spread_home,
            "spread_home_prob": self.spread_home_prob,
            "total_line": self.total_line,
            "over_prob": self.over_prob,
            "under_prob": self.under_prob,
            "num_bookmakers": self.num_bookmakers,
        }


def american_to_probability(american_odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        american_odds: American format odds (+150 or -200)

    Returns:
        Implied probability (0-1)
    """
    if american_odds > 0:
        # Underdog: +150 means $100 bet wins $150
        return 100 / (american_odds + 100)
    else:
        # Favorite: -200 means $200 bet wins $100
        return abs(american_odds) / (abs(american_odds) + 100)


def decimal_to_probability(decimal_odds: float) -> float:
    """
    Convert decimal odds to implied probability.

    Args:
        decimal_odds: Decimal format odds (2.50, 1.50)

    Returns:
        Implied probability (0-1)
    """
    if decimal_odds <= 1:
        return 1.0
    return 1 / decimal_odds


def remove_vig(probs: list[float]) -> list[float]:
    """
    Remove vigorish (bookmaker margin) from probabilities.

    Bookmaker odds typically sum to >100%. This normalizes them.

    Args:
        probs: List of implied probabilities

    Returns:
        Normalized probabilities summing to 1.0
    """
    total = sum(probs)
    if total == 0:
        return probs
    return [p / total for p in probs]


class OddsFeed:
    """Client for the-odds-api.com.

    Fetches odds from major sportsbooks and calculates consensus probabilities.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    # Bookmakers to use for consensus (major US books)
    PREFERRED_BOOKMAKERS = [
        "fanduel",
        "draftkings",
        "betmgm",
        "caesars",
        "pointsbetus",
        "bovada",
        "betonlineag",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        cache_ttl_seconds: int = 300,  # 5 minute cache
    ) -> None:
        """
        Initialize odds feed.

        Args:
            api_key: the-odds-api.com API key (or from ODDS_API_KEY env var)
            cache_ttl_seconds: How long to cache odds
        """
        self._api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[datetime, list[GameOdds]]] = {}
        self._client: httpx.AsyncClient | None = None

        # Track API usage
        self._requests_remaining: int | None = None
        self._requests_used: int | None = None

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OddsFeed:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def get_odds(
        self,
        sport: Sport | str,
        markets: str = "h2h,spreads,totals",
        force_refresh: bool = False,
    ) -> list[GameOdds]:
        """
        Get odds for a sport.

        Args:
            sport: Sport to fetch (Sport enum or string)
            markets: Comma-separated market types (h2h, spreads, totals)
            force_refresh: Bypass cache

        Returns:
            List of GameOdds for upcoming events
        """
        if not self._api_key:
            logger.warning("No ODDS_API_KEY configured - odds feed disabled")
            return []

        sport_key = sport.value if isinstance(sport, Sport) else sport

        # Check cache
        cache_key = f"{sport_key}:{markets}"
        if not force_refresh and cache_key in self._cache:
            cached_time, cached_odds = self._cache[cache_key]
            age = (datetime.now(timezone.utc) - cached_time).total_seconds()
            if age < self._cache_ttl:
                return cached_odds

        # Fetch from API
        try:
            await self.connect()
            if self._client is None:
                return []

            url = f"{self.BASE_URL}/sports/{sport_key}/odds"
            params = {
                "apiKey": self._api_key,
                "regions": "us",
                "markets": markets,
                "oddsFormat": "american",
            }

            response = await self._client.get(url, params=params)

            # Track API usage
            self._requests_remaining = int(
                response.headers.get("x-requests-remaining", 0)
            )
            self._requests_used = int(response.headers.get("x-requests-used", 0))

            if response.status_code == 401:
                logger.error("Invalid ODDS_API_KEY")
                return []

            if response.status_code == 429:
                logger.warning("Odds API rate limit exceeded")
                return []

            if response.status_code != 200:
                logger.error(f"Odds API error: {response.status_code}")
                return []

            data = response.json()
            odds = self._parse_odds(data, sport_key)

            # Update cache
            self._cache[cache_key] = (datetime.now(timezone.utc), odds)

            logger.info(
                f"Fetched {len(odds)} events for {sport_key} "
                f"(API requests remaining: {self._requests_remaining})"
            )

            return odds

        except Exception as e:
            logger.error(f"Failed to fetch odds: {e}")
            return []

    def _parse_odds(self, data: list[dict], sport: str) -> list[GameOdds]:
        """Parse API response into GameOdds objects."""
        results = []

        for event in data:
            try:
                game = GameOdds(
                    sport=sport,
                    event_id=event.get("id", ""),
                    home_team=event.get("home_team", ""),
                    away_team=event.get("away_team", ""),
                    commence_time=datetime.fromisoformat(
                        event.get("commence_time", "").replace("Z", "+00:00")
                    ),
                )

                bookmakers = event.get("bookmakers", [])
                game.num_bookmakers = len(bookmakers)

                # Calculate consensus from preferred bookmakers
                h2h_probs = self._extract_h2h(bookmakers, game.home_team, game.away_team)
                spread_data = self._extract_spreads(bookmakers, game.home_team)
                totals_data = self._extract_totals(bookmakers)

                # Set H2H probabilities (vig-adjusted)
                if h2h_probs["home"]:
                    avg_home = sum(h2h_probs["home"]) / len(h2h_probs["home"])
                    avg_away = sum(h2h_probs["away"]) / len(h2h_probs["away"])
                    avg_draw = (
                        sum(h2h_probs["draw"]) / len(h2h_probs["draw"])
                        if h2h_probs["draw"]
                        else 0
                    )

                    # Remove vig
                    if avg_draw > 0:
                        normalized = remove_vig([avg_home, avg_away, avg_draw])
                        game.home_win_prob = normalized[0]
                        game.away_win_prob = normalized[1]
                        game.draw_prob = normalized[2]
                    else:
                        normalized = remove_vig([avg_home, avg_away])
                        game.home_win_prob = normalized[0]
                        game.away_win_prob = normalized[1]

                # Set spread data
                if spread_data["home_spread"]:
                    game.spread_home = sum(spread_data["home_spread"]) / len(
                        spread_data["home_spread"]
                    )
                    game.spread_away = -game.spread_home if game.spread_home else None

                    if spread_data["home_prob"]:
                        avg_home = sum(spread_data["home_prob"]) / len(
                            spread_data["home_prob"]
                        )
                        avg_away = sum(spread_data["away_prob"]) / len(
                            spread_data["away_prob"]
                        )
                        normalized = remove_vig([avg_home, avg_away])
                        game.spread_home_prob = normalized[0]
                        game.spread_away_prob = normalized[1]

                # Set totals data
                if totals_data["line"]:
                    game.total_line = sum(totals_data["line"]) / len(totals_data["line"])

                    if totals_data["over_prob"]:
                        avg_over = sum(totals_data["over_prob"]) / len(
                            totals_data["over_prob"]
                        )
                        avg_under = sum(totals_data["under_prob"]) / len(
                            totals_data["under_prob"]
                        )
                        normalized = remove_vig([avg_over, avg_under])
                        game.over_prob = normalized[0]
                        game.under_prob = normalized[1]

                results.append(game)

            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")
                continue

        return results

    def _extract_h2h(
        self,
        bookmakers: list[dict],
        home_team: str,
        away_team: str,
    ) -> dict[str, list[float]]:
        """Extract head-to-head moneyline odds."""
        probs: dict[str, list[float]] = {"home": [], "away": [], "draw": []}

        for book in bookmakers:
            if book.get("key") not in self.PREFERRED_BOOKMAKERS:
                continue

            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)

                    if price == 0:
                        continue

                    prob = american_to_probability(price)

                    if name == home_team:
                        probs["home"].append(prob)
                    elif name == away_team:
                        probs["away"].append(prob)
                    elif name.lower() == "draw":
                        probs["draw"].append(prob)

        return probs

    def _extract_spreads(
        self,
        bookmakers: list[dict],
        home_team: str,
    ) -> dict[str, list[float]]:
        """Extract point spread odds."""
        data: dict[str, list[float]] = {
            "home_spread": [],
            "home_prob": [],
            "away_prob": [],
        }

        for book in bookmakers:
            if book.get("key") not in self.PREFERRED_BOOKMAKERS:
                continue

            for market in book.get("markets", []):
                if market.get("key") != "spreads":
                    continue

                home_point = None
                home_price = None
                away_price = None

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    point = outcome.get("point", 0)
                    price = outcome.get("price", 0)

                    if name == home_team:
                        home_point = point
                        home_price = price
                    else:
                        away_price = price

                if home_point is not None and home_price and away_price:
                    data["home_spread"].append(home_point)
                    data["home_prob"].append(american_to_probability(home_price))
                    data["away_prob"].append(american_to_probability(away_price))

        return data

    def _extract_totals(self, bookmakers: list[dict]) -> dict[str, list[float]]:
        """Extract over/under totals odds."""
        data: dict[str, list[float]] = {
            "line": [],
            "over_prob": [],
            "under_prob": [],
        }

        for book in bookmakers:
            if book.get("key") not in self.PREFERRED_BOOKMAKERS:
                continue

            for market in book.get("markets", []):
                if market.get("key") != "totals":
                    continue

                line = None
                over_price = None
                under_price = None

                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "").lower()
                    point = outcome.get("point", 0)
                    price = outcome.get("price", 0)

                    if name == "over":
                        line = point
                        over_price = price
                    elif name == "under":
                        under_price = price

                if line is not None and over_price and under_price:
                    data["line"].append(line)
                    data["over_prob"].append(american_to_probability(over_price))
                    data["under_prob"].append(american_to_probability(under_price))

        return data

    def get_api_usage(self) -> dict[str, Any]:
        """Get API usage statistics."""
        return {
            "requests_remaining": self._requests_remaining,
            "requests_used": self._requests_used,
            "cache_entries": len(self._cache),
        }

    async def get_upcoming_games(
        self,
        sport: Sport | str,
        hours_ahead: int = 24,
    ) -> list[GameOdds]:
        """
        Get odds for games starting in the next N hours.

        Args:
            sport: Sport to fetch
            hours_ahead: Only return games within this window

        Returns:
            Filtered list of GameOdds
        """
        all_odds = await self.get_odds(sport)
        cutoff = datetime.now(timezone.utc) + timedelta(hours=hours_ahead)

        return [
            game
            for game in all_odds
            if game.commence_time <= cutoff and game.commence_time > datetime.now(timezone.utc)
        ]

    def find_matching_game(
        self,
        odds_list: list[GameOdds],
        team_name: str,
    ) -> GameOdds | None:
        """
        Find a game involving a specific team.

        Args:
            odds_list: List of GameOdds to search
            team_name: Team name to find (partial match)

        Returns:
            Matching GameOdds or None
        """
        team_lower = team_name.lower()

        for game in odds_list:
            if (
                team_lower in game.home_team.lower()
                or team_lower in game.away_team.lower()
            ):
                return game

        return None
