"""Exports market opportunities for manual Claude Code analysis."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from kalshi_bot.core.types import MarketData
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class OpportunityExporter:
    """
    Exports market opportunities for manual Claude Code analysis.

    Instead of automated Claude API calls, this generates reports that
    you can share with Claude Code for in-session analysis.
    """

    def __init__(
        self,
        min_volume: int = 100,
        max_spread_cents: int = 10,
        max_days_to_expiry: int = 30,
        price_range: tuple[int, int] = (20, 80),
    ) -> None:
        """
        Initialize exporter with filtering criteria.

        Args:
            min_volume: Minimum contract volume
            max_spread_cents: Maximum bid/ask spread in cents
            max_days_to_expiry: Maximum days until expiration
            price_range: (min, max) YES price in cents for edge potential
        """
        self.min_volume = min_volume
        self.max_spread_cents = max_spread_cents
        self.max_days_to_expiry = max_days_to_expiry
        self.price_range = price_range

    def filter_candidates(self, markets: list[MarketData]) -> list[MarketData]:
        """
        Filter markets to find good analysis candidates.

        Filters for:
        - High volume (> min_volume contracts)
        - Reasonable spreads (< max_spread_cents) - skipped if no spread data
        - Near-term expiry (< max_days_to_expiry)
        - Price between price_range (most edge potential)

        Args:
            markets: List of all markets

        Returns:
            Filtered list of candidate markets
        """
        now = datetime.utcnow()
        max_expiry = now + timedelta(days=self.max_days_to_expiry)
        candidates = []

        for market in markets:
            # Check volume
            if market.volume < self.min_volume:
                continue

            # Check spread (only if spread data is available)
            spread = market.spread
            if spread is not None and spread > self.max_spread_cents:
                continue

            # Check expiration (skip if no expiration or within range)
            if market.expiration_time and market.expiration_time > max_expiry:
                continue

            # Check price range - use yes_ask if available, otherwise yes_bid
            price = market.yes_ask or market.yes_bid
            if price is None:
                continue
            if not (self.price_range[0] <= price <= self.price_range[1]):
                continue

            # Check market is open/active
            if market.status not in ("open", "active"):
                continue

            candidates.append(market)

        # Sort by volume descending
        candidates.sort(key=lambda m: m.volume, reverse=True)

        return candidates

    def export_for_analysis(
        self,
        markets: list[MarketData],
        max_markets: int = 20,
    ) -> str:
        """
        Generate a markdown report of interesting markets.

        Args:
            markets: List of all markets
            max_markets: Maximum number of markets to include

        Returns:
            Markdown-formatted report for Claude Code analysis
        """
        candidates = self.filter_candidates(markets)[:max_markets]

        lines = [
            "# Market Opportunities for Analysis",
            "",
            f"**Generated:** {datetime.utcnow().isoformat()}Z",
            f"**Filters Applied:**",
            f"- Min Volume: {self.min_volume} contracts",
            f"- Max Spread: {self.max_spread_cents} cents",
            f"- Max Days to Expiry: {self.max_days_to_expiry}",
            f"- Price Range: {self.price_range[0]}-{self.price_range[1]} cents",
            "",
            f"**Found:** {len(candidates)} candidates",
            "",
            "---",
            "",
        ]

        if not candidates:
            lines.append("No markets match the filtering criteria.")
            return "\n".join(lines)

        for i, market in enumerate(candidates, 1):
            days_to_expiry = "N/A"
            if market.expiration_time:
                delta = market.expiration_time - datetime.utcnow()
                days_to_expiry = f"{delta.days}d {delta.seconds // 3600}h"

            lines.extend([
                f"## {i}. {market.title}",
                "",
                f"| Property | Value |",
                f"|----------|-------|",
                f"| **Ticker** | `{market.ticker}` |",
                f"| **Event** | `{market.event_ticker}` |",
                f"| **YES Price** | {market.yes_ask} cents (ask) / {market.yes_bid} cents (bid) |",
                f"| **NO Price** | {market.no_ask} cents (ask) / {market.no_bid} cents (bid) |",
                f"| **Spread** | {market.spread} cents |",
                f"| **Volume** | {market.volume:,} contracts |",
                f"| **Open Interest** | {market.open_interest:,} |",
                f"| **Expiry** | {days_to_expiry} |",
                "",
                "**Analysis Request:** Please provide your probability estimate, confidence level, and recommendation (BUY YES / BUY NO / SKIP).",
                "",
                "---",
                "",
            ])

        lines.extend([
            "## How to Use This Report",
            "",
            "1. Review each market above",
            "2. For each, I'll provide:",
            "   - My probability estimate (0-100%)",
            "   - Confidence level (low/medium/high)",
            "   - Edge calculation (my estimate vs market price)",
            "   - Recommendation (BUY YES / BUY NO / SKIP)",
            "   - Reasoning",
            "",
            "3. You decide which trades to execute",
            "",
            "**Example Analysis Format:**",
            "```",
            '## Analysis: "Will BTC exceed $100k by March 31?"',
            "",
            "Market Price: 42 cents YES",
            "My Estimate: 55% probability",
            "Edge: +13% (bullish)",
            "Confidence: Medium",
            "",
            "Reasoning:",
            "- BTC currently at $95k, 5% away from target",
            "- 13 days remaining",
            "- Historical volatility supports this move",
            "- But macro uncertainty is high",
            "",
            "Recommendation: SKIP (edge exists but confidence too low)",
            "```",
        ])

        return "\n".join(lines)

    def export_to_file(
        self,
        markets: list[MarketData],
        output_path: str | Path = "opportunities.md",
        max_markets: int = 20,
    ) -> Path:
        """
        Export opportunities report to a markdown file.

        Args:
            markets: List of all markets
            output_path: Path to output file
            max_markets: Maximum number of markets to include

        Returns:
            Path to the created file
        """
        output_path = Path(output_path)
        candidates = self.filter_candidates(markets)
        actual_count = min(len(candidates), max_markets)
        content = self.export_for_analysis(markets, max_markets)
        output_path.write_text(content)

        logger.info(f"Exported {actual_count} market opportunities to {output_path}")
        return output_path

    def get_candidates(self, markets: list[MarketData]) -> list[dict[str, Any]]:
        """
        Get candidate markets as dictionaries for API responses.

        Args:
            markets: List of all markets

        Returns:
            List of market data as dictionaries
        """
        candidates = self.filter_candidates(markets)
        return [m.to_dict() for m in candidates]
