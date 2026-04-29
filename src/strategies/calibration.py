"""Calibration Strategy - Uses external probability sources to find edge.

This strategy compares Kalshi market prices against external model probabilities
from sources like:
- Sportsbook consensus odds (via the-odds-api)
- Weather forecasts (NWS)
- Polling aggregates (for political markets)

When the external model disagrees with market price by more than a threshold,
we emit a signal to trade in the direction of the external model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

from src.core.types import Signal
from src.data.odds_feed import OddsFeed, Sport, GameOdds
from src.strategies.base import Strategy
from src.observability.logging import get_logger

logger = get_logger(__name__)


# Team name mappings from Kalshi tickers to standard names
TEAM_MAPPINGS: dict[str, str] = {
    # NFL
    "KC": "Kansas City Chiefs",
    "SF": "San Francisco 49ers",
    "BUF": "Buffalo Bills",
    "PHI": "Philadelphia Eagles",
    "DAL": "Dallas Cowboys",
    "MIA": "Miami Dolphins",
    "BAL": "Baltimore Ravens",
    "CIN": "Cincinnati Bengals",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    # NBA
    "LAL": "Los Angeles Lakers",
    "BOS": "Boston Celtics",
    "GSW": "Golden State Warriors",
    "MIL": "Milwaukee Bucks",
    "DEN": "Denver Nuggets",
    "PHX": "Phoenix Suns",
    # MLB
    "NYY": "New York Yankees",
    "LAD": "Los Angeles Dodgers",
    "HOU": "Houston Astros",
    "ATL": "Atlanta Braves",
}

# Kalshi sports market patterns
NFL_PATTERN = re.compile(r"NFL-([A-Z]+)-([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})")
NBA_PATTERN = re.compile(r"NBA-([A-Z]+)-([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})")
MLB_PATTERN = re.compile(r"MLB-([A-Z]+)-([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})")


@dataclass
class ParsedSportsMarket:
    """Parsed sports market ticker."""

    ticker: str
    sport: Sport
    home_team: str
    away_team: str
    game_date: datetime
    market_type: str  # "moneyline", "spread", "total"
    line: float | None = None  # For spreads/totals


class CalibrationStrategy(Strategy):
    """
    Strategy that uses external probability sources for calibration.

    Compares market prices to external model probabilities and signals
    when there's significant disagreement (edge).
    """

    # Default thresholds
    DEFAULT_MIN_EDGE = 0.05  # 5% minimum edge
    DEFAULT_MIN_CONFIDENCE = 0.60

    # Only trade markets in reasonable price range
    MIN_PRICE_CENTS = 15
    MAX_PRICE_CENTS = 85

    # Maximum hours before game to trade
    MAX_HOURS_TO_EVENT = 48

    def __init__(
        self,
        odds_feed: OddsFeed | None = None,
        db=None,
        enabled: bool = True,
        min_edge: float = DEFAULT_MIN_EDGE,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        enabled_sports: list[str] | None = None,
    ) -> None:
        """
        Initialize calibration strategy.

        Args:
            odds_feed: OddsFeed client for sportsbook odds
            db: Optional database connection
            enabled: Whether strategy is active
            min_edge: Minimum edge to generate signals
            min_confidence: Minimum confidence threshold
            enabled_sports: Sports to trade (default: NFL, NBA)
        """
        super().__init__(
            db=db,
            enabled=enabled,
            min_edge=min_edge,
            min_confidence=min_confidence,
        )
        self._odds_feed = odds_feed
        self._enabled_sports = enabled_sports or ["NFL", "NBA"]

        # Cache for odds data
        self._odds_cache: dict[str, list[GameOdds]] = {}
        self._cache_time: datetime | None = None
        self._cache_ttl = 300  # 5 minutes

    @property
    def name(self) -> str:
        return "calibration"

    async def generate_signals(
        self,
        markets: list[dict[str, Any]],
    ) -> list[Signal]:
        """
        Analyze markets and generate signals based on external odds.

        Args:
            markets: Market data from Kalshi API

        Returns:
            List of Signal objects
        """
        if not self._enabled:
            return []

        if self._odds_feed is None:
            logger.warning("No odds feed configured for calibration strategy")
            return []

        signals = []
        sports_markets = 0
        analyzed = 0

        # Refresh odds cache if needed
        await self._refresh_odds_cache()

        for market in markets:
            ticker = market.get("ticker", "")

            # Parse ticker to check if it's a sports market
            parsed = self._parse_sports_ticker(ticker)
            if parsed is None:
                continue

            sports_markets += 1

            # Check if sport is enabled
            sport_name = parsed.sport.name
            if sport_name not in self._enabled_sports:
                continue

            # Check time to event
            hours_to_event = (
                parsed.game_date - datetime.now(timezone.utc)
            ).total_seconds() / 3600
            if hours_to_event <= 0 or hours_to_event > self.MAX_HOURS_TO_EVENT:
                continue

            analyzed += 1

            # Get market price
            market_price = self._get_market_price(market)
            if market_price is None:
                continue

            # Check price bounds
            if market_price < self.MIN_PRICE_CENTS or market_price > self.MAX_PRICE_CENTS:
                continue

            # Find matching game in odds cache
            game_odds = self._find_matching_game(parsed)
            if game_odds is None:
                continue

            # Calculate external probability
            ext_prob, confidence = self._get_external_probability(parsed, game_odds)
            if ext_prob is None:
                continue

            # Check for edge
            market_prob = market_price / 100.0
            edge = abs(ext_prob - market_prob)

            if edge < self._min_edge:
                continue

            # Determine direction
            if ext_prob > market_prob:
                direction = "yes"
                target_prob = ext_prob
            else:
                direction = "no"
                target_prob = 1 - ext_prob

            signal = self.create_signal(
                market_ticker=ticker,
                direction=direction,
                target_probability=target_prob,
                market_probability=market_prob if direction == "yes" else (1 - market_prob),
                confidence=confidence,
                max_position=50,  # Conservative sizing
                metadata={
                    "external_probability": ext_prob,
                    "source": "sportsbook_consensus",
                    "sport": sport_name,
                    "home_team": game_odds.home_team,
                    "away_team": game_odds.away_team,
                    "num_bookmakers": game_odds.num_bookmakers,
                    "hours_to_event": hours_to_event,
                },
                expires_in_hours=min(hours_to_event, 2.0),
            )

            if signal is not None:
                signals.append(signal)
                logger.info(
                    f"Calibration signal: {ticker} {direction} "
                    f"(ext={ext_prob:.1%}, mkt={market_prob:.1%}, edge={edge:.1%})"
                )

        self.update_metrics(
            sports_markets_found=sports_markets,
            markets_analyzed=analyzed,
            signals_generated=len(signals),
        )
        self.record_run()

        return signals

    async def _refresh_odds_cache(self) -> None:
        """Refresh odds cache if stale."""
        now = datetime.now(timezone.utc)

        if self._cache_time is not None:
            age = (now - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                return

        if self._odds_feed is None:
            return

        try:
            # Fetch odds for each enabled sport
            sport_map = {
                "NFL": Sport.NFL,
                "NBA": Sport.NBA,
                "MLB": Sport.MLB,
                "NHL": Sport.NHL,
                "NCAA_FOOTBALL": Sport.NCAA_FOOTBALL,
                "NCAA_BASKETBALL": Sport.NCAA_BASKETBALL,
            }

            for sport_name in self._enabled_sports:
                sport = sport_map.get(sport_name)
                if sport:
                    odds = await self._odds_feed.get_odds(sport)
                    self._odds_cache[sport_name] = odds
                    logger.debug(f"Cached {len(odds)} games for {sport_name}")

            self._cache_time = now

        except Exception as e:
            logger.error(f"Failed to refresh odds cache: {e}")

    def _parse_sports_ticker(self, ticker: str) -> ParsedSportsMarket | None:
        """Parse a sports market ticker."""
        # Try NFL
        match = NFL_PATTERN.match(ticker)
        if match:
            return self._parse_game_ticker(ticker, match, Sport.NFL)

        # Try NBA
        match = NBA_PATTERN.match(ticker)
        if match:
            return self._parse_game_ticker(ticker, match, Sport.NBA)

        # Try MLB
        match = MLB_PATTERN.match(ticker)
        if match:
            return self._parse_game_ticker(ticker, match, Sport.MLB)

        return None

    def _parse_game_ticker(
        self,
        ticker: str,
        match: re.Match,
        sport: Sport,
    ) -> ParsedSportsMarket | None:
        """Parse a game ticker match."""
        try:
            home_code = match.group(1)
            away_code = match.group(2)
            year = 2000 + int(match.group(3))
            month_str = match.group(4)
            day = int(match.group(5))

            months = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
            }
            month = months.get(month_str)
            if month is None:
                return None

            game_date = datetime(year, month, day, 23, 59, tzinfo=timezone.utc)

            home_team = TEAM_MAPPINGS.get(home_code, home_code)
            away_team = TEAM_MAPPINGS.get(away_code, away_code)

            return ParsedSportsMarket(
                ticker=ticker,
                sport=sport,
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                market_type="moneyline",
            )

        except (ValueError, IndexError):
            return None

    def _get_market_price(self, market: dict[str, Any]) -> int | None:
        """Get market price in cents."""
        price = market.get("last_price")
        if price is not None:
            return int(price)

        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        if yes_bid is not None and yes_ask is not None:
            return (int(yes_bid) + int(yes_ask)) // 2

        return None

    def _find_matching_game(self, parsed: ParsedSportsMarket) -> GameOdds | None:
        """Find matching game in odds cache."""
        sport_name = parsed.sport.name
        odds_list = self._odds_cache.get(sport_name, [])

        if not odds_list:
            return None

        # Try to match by team names
        for game in odds_list:
            # Check if game is on the same date
            if game.commence_time.date() != parsed.game_date.date():
                continue

            # Check if teams match
            home_match = (
                parsed.home_team.lower() in game.home_team.lower()
                or game.home_team.lower() in parsed.home_team.lower()
            )
            away_match = (
                parsed.away_team.lower() in game.away_team.lower()
                or game.away_team.lower() in parsed.away_team.lower()
            )

            if home_match or away_match:
                return game

        return None

    def _get_external_probability(
        self,
        parsed: ParsedSportsMarket,
        game: GameOdds,
    ) -> tuple[float | None, float]:
        """
        Get external probability for the market.

        Returns (probability, confidence) tuple.
        """
        if parsed.market_type == "moneyline":
            # For moneyline, we want home team win probability
            prob = game.home_win_prob
            if prob is None:
                return None, 0.0

            # Confidence based on number of bookmakers
            confidence = min(0.90, 0.50 + (game.num_bookmakers * 0.05))

            return prob, confidence

        elif parsed.market_type == "spread":
            prob = game.spread_home_prob
            if prob is None:
                return None, 0.0

            confidence = min(0.85, 0.50 + (game.num_bookmakers * 0.05))
            return prob, confidence

        elif parsed.market_type == "total":
            prob = game.over_prob
            if prob is None:
                return None, 0.0

            confidence = min(0.80, 0.45 + (game.num_bookmakers * 0.05))
            return prob, confidence

        return None, 0.0

    def get_odds_cache_status(self) -> dict[str, Any]:
        """Get status of odds cache."""
        return {
            "sports_cached": list(self._odds_cache.keys()),
            "games_per_sport": {
                sport: len(games) for sport, games in self._odds_cache.items()
            },
            "cache_time": self._cache_time.isoformat() if self._cache_time else None,
            "cache_ttl": self._cache_ttl,
        }
