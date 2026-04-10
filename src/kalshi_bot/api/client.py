"""Async HTTP client for Kalshi API."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urljoin

import httpx

from kalshi_bot.api.auth import KalshiAuth
from kalshi_bot.api.rate_limiter import RateLimiter
from kalshi_bot.config.settings import Settings
from kalshi_bot.core.exceptions import APIError, AuthenticationError, RateLimitError
from kalshi_bot.core.types import Candlestick, MarketData, Order, OrderBook, OrderBookLevel, Position, Side, OrderStatus, OrderType
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


def _dollars_to_cents(val: str | None) -> int | None:
    """Convert dollar string (e.g. '0.42') to cents (42)."""
    if val is None:
        return None
    try:
        return int(float(val) * 100)
    except (ValueError, TypeError):
        return None


class KalshiAPIClient:
    """Async HTTP client for Kalshi trading API."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize API client.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.base_url = settings.api_base_url
        self._auth = KalshiAuth(settings.api_key_id, settings.private_key_path)
        self._rate_limiter = RateLimiter(
            requests_per_second=settings.api.requests_per_second
        )
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> KalshiAPIClient:
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        """Initialize HTTP client."""
        if self._client is not None:
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.api.timeout),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        logger.info(f"API client connected to {self.base_url}")

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("API client closed")

    def _get_path(self, endpoint: str) -> str:
        """Get full path for endpoint (for signing)."""
        if endpoint.startswith("/"):
            return f"/trade-api/v2{endpoint}"
        return f"/trade-api/v2/{endpoint}"

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make authenticated API request.

        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json_data: JSON body data

        Returns:
            Response JSON data

        Raises:
            APIError: On API errors
            RateLimitError: On rate limit exceeded
            AuthenticationError: On auth errors
        """
        if self._client is None:
            await self.connect()
            if self._client is None:
                raise APIError("Failed to initialize HTTP client")

        await self._rate_limiter.acquire()

        path = self._get_path(endpoint)
        url = self.base_url.rstrip("/") + "/" + endpoint.lstrip("/")

        try:
            auth_headers = self._auth.get_auth_headers(method.upper(), path)

            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=auth_headers,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                retry_seconds = float(retry_after) if retry_after else None
                await self._rate_limiter.report_rate_limit(retry_seconds)
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=retry_seconds,
                    details={"endpoint": endpoint},
                )

            if response.status_code == 401:
                raise AuthenticationError(
                    "Authentication failed",
                    details={"endpoint": endpoint, "status": response.status_code},
                )

            if response.status_code == 403:
                raise AuthenticationError(
                    "Access forbidden",
                    details={"endpoint": endpoint, "status": response.status_code},
                )

            if response.status_code >= 400:
                error_body = response.json() if response.content else {}
                logger.error(f"API {response.status_code} error on {endpoint}: {error_body}")
                raise APIError(
                    f"API error: {response.status_code} - {error_body}",
                    status_code=response.status_code,
                    response_body=error_body,
                    details={"endpoint": endpoint},
                )

            await self._rate_limiter.report_success()

            if response.content:
                return response.json()
            return {}

        except httpx.TimeoutException as e:
            raise APIError(
                f"Request timeout: {endpoint}",
                details={"endpoint": endpoint, "error": str(e)},
            ) from e
        except httpx.RequestError as e:
            raise APIError(
                f"Request error: {endpoint}",
                details={"endpoint": endpoint, "error": str(e)},
            ) from e

    async def get(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make GET request."""
        return await self._request("GET", endpoint, params=params)

    async def post(
        self, endpoint: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make POST request."""
        return await self._request("POST", endpoint, json_data=data)

    async def delete(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make DELETE request."""
        return await self._request("DELETE", endpoint, params=params)

    # Account endpoints

    async def get_balance(self) -> dict[str, Any]:
        """Get account balance."""
        return await self.get("/portfolio/balance")

    async def get_positions(self) -> list[Position]:
        """Get all positions."""
        response = await self.get("/portfolio/positions")
        positions = []

        for pos_data in response.get("market_positions", []):
            position = Position(
                market_ticker=pos_data["ticker"],
                side=Side.YES if pos_data.get("position", 0) > 0 else Side.NO,
                quantity=abs(pos_data.get("position", 0)),
                average_price=pos_data.get("average_cost", 0),
                market_exposure=pos_data.get("market_exposure", 0) / 100,
                realized_pnl=pos_data.get("realized_pnl", 0) / 100,
            )
            positions.append(position)

        return positions

    async def get_fills(
        self,
        ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Get trade fills."""
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor

        return await self.get("/portfolio/fills", params)

    # Market endpoints

    async def get_markets(
        self,
        event_ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[MarketData], str | None]:
        """
        Get markets with pagination support.

        Returns:
            Tuple of (markets list, next cursor or None if no more pages)
        """
        params: dict[str, Any] = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        response = await self.get("/markets", params)
        markets = []

        for market in response.get("markets", []):
            market_data = MarketData(
                ticker=market["ticker"],
                event_ticker=market.get("event_ticker", ""),
                title=market.get("title", ""),
                subtitle=market.get("subtitle"),
                status=market.get("status", "open"),
                yes_bid=_dollars_to_cents(market.get("yes_bid_dollars")),
                yes_ask=_dollars_to_cents(market.get("yes_ask_dollars")),
                no_bid=_dollars_to_cents(market.get("no_bid_dollars")),
                no_ask=_dollars_to_cents(market.get("no_ask_dollars")),
                last_price=_dollars_to_cents(market.get("last_price_dollars")),
                volume=market.get("volume", 0),
                open_interest=market.get("open_interest", 0),
                close_time=datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
                if market.get("close_time")
                else None,
                expiration_time=datetime.fromisoformat(
                    market["expiration_time"].replace("Z", "+00:00")
                )
                if market.get("expiration_time")
                else None,
                result=market.get("result"),
            )
            markets.append(market_data)

        next_cursor = response.get("cursor")
        return markets, next_cursor

    async def get_all_markets(
        self,
        status: str | None = None,
        min_volume: int = 0,
        min_open_interest: int = 0,
        require_orderbook: bool = True,
        max_pages: int | None = None,
    ) -> list[MarketData]:
        """
        Fetch all markets with pagination, optionally filtering for liquidity.

        Args:
            status: Market status filter (e.g., "open")
            min_volume: Minimum volume to include market (0 = no filter)
            min_open_interest: Minimum open interest to include (0 = no filter)
            require_orderbook: If True, only include markets with bid/ask prices
            max_pages: Maximum pages to fetch (None for unlimited)

        Returns:
            List of all markets matching the criteria
        """
        all_markets: list[MarketData] = []
        cursor: str | None = None
        page_count = 0

        while True:
            markets, cursor = await self.get_markets(
                status=status, limit=100, cursor=cursor
            )
            page_count += 1

            for market in markets:
                # Apply liquidity filters
                # Require at least volume OR open_interest to meet threshold
                has_volume = market.volume >= min_volume if min_volume > 0 else True
                has_open_interest = market.open_interest >= min_open_interest if min_open_interest > 0 else True

                # If both thresholds specified, require at least one to pass (OR logic)
                if min_volume > 0 and min_open_interest > 0:
                    if not has_volume and not has_open_interest:
                        continue
                else:
                    # If only one threshold specified, require it to pass
                    if min_volume > 0 and not has_volume:
                        continue
                    if min_open_interest > 0 and not has_open_interest:
                        continue

                if require_orderbook and (market.yes_bid is None and market.yes_ask is None):
                    continue
                all_markets.append(market)

            logger.info(
                f"Fetched page {page_count}: {len(markets)} markets, "
                f"{len(all_markets)} total after filtering"
            )

            # Stop if page limit reached
            if max_pages and page_count >= max_pages:
                logger.info(f"Reached max_pages limit ({max_pages})")
                break

            # No more pages if we got fewer than requested or no cursor
            if len(markets) < 100 or cursor is None:
                break

        logger.info(
            f"Fetched {len(all_markets)} liquid markets from {page_count} pages"
        )
        return all_markets

    async def get_market(self, ticker: str) -> MarketData:
        """Get single market by ticker."""
        response = await self.get(f"/markets/{ticker}")
        market = response.get("market", response)

        return MarketData(
            ticker=market["ticker"],
            event_ticker=market.get("event_ticker", ""),
            title=market.get("title", ""),
            subtitle=market.get("subtitle"),
            status=market.get("status", "open"),
            yes_bid=_dollars_to_cents(market.get("yes_bid_dollars")),
            yes_ask=_dollars_to_cents(market.get("yes_ask_dollars")),
            no_bid=_dollars_to_cents(market.get("no_bid_dollars")),
            no_ask=_dollars_to_cents(market.get("no_ask_dollars")),
            last_price=_dollars_to_cents(market.get("last_price_dollars")),
            volume=market.get("volume", 0),
            open_interest=market.get("open_interest", 0),
            close_time=datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
            if market.get("close_time")
            else None,
            expiration_time=datetime.fromisoformat(
                market["expiration_time"].replace("Z", "+00:00")
            )
            if market.get("expiration_time")
            else None,
            result=market.get("result"),
        )

    async def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        """Get order book for market."""
        response = await self.get(f"/markets/{ticker}/orderbook", {"depth": depth})
        orderbook = response.get("orderbook", response)

        def parse_levels(levels: list[list[int]]) -> list[OrderBookLevel]:
            return [OrderBookLevel(price=level[0], quantity=level[1]) for level in levels]

        return OrderBook(
            market_ticker=ticker,
            yes_bids=parse_levels(orderbook.get("yes", [[]])[0] if orderbook.get("yes") else []),
            yes_asks=parse_levels(orderbook.get("yes", [[], []])[1] if len(orderbook.get("yes", [])) > 1 else []),
            no_bids=parse_levels(orderbook.get("no", [[]])[0] if orderbook.get("no") else []),
            no_asks=parse_levels(orderbook.get("no", [[], []])[1] if len(orderbook.get("no", [])) > 1 else []),
        )

    async def get_events(
        self,
        status: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Get events."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        return await self.get("/events", params)

    # Order endpoints

    async def create_order(
        self,
        ticker: str,
        side: Side,
        order_type: OrderType,
        quantity: int,
        price: int | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Create a new order.

        Args:
            ticker: Market ticker
            side: Order side (yes/no)
            order_type: Order type (limit/market)
            quantity: Number of contracts
            price: Price in cents (required for limit orders)
            client_order_id: Optional client order ID

        Returns:
            Created order
        """
        data: dict[str, Any] = {
            "ticker": ticker,
            "action": "buy",
            "side": side.value,
            "type": order_type.value,
            "count": quantity,
        }

        if order_type == OrderType.LIMIT and price is not None:
            data["yes_price" if side == Side.YES else "no_price"] = price

        if client_order_id:
            data["client_order_id"] = client_order_id

        logger.info(f"Creating order: {data}")
        response = await self.post("/portfolio/orders", data)
        order_data = response.get("order", response)

        return Order(
            order_id=order_data["order_id"],
            market_ticker=ticker,
            side=side,
            order_type=order_type,
            price=price or 0,
            quantity=quantity,
            status=OrderStatus(order_data.get("status", "pending")),
            filled_quantity=order_data.get("fill_count", 0),
            remaining_quantity=order_data.get("remaining_count", quantity),
        )

    async def get_order(self, order_id: str) -> Order:
        """Get order by ID."""
        response = await self.get(f"/portfolio/orders/{order_id}")
        order_data = response.get("order", response)

        return Order(
            order_id=order_data["order_id"],
            market_ticker=order_data["ticker"],
            side=Side(order_data["side"]),
            order_type=OrderType(order_data["type"]),
            price=order_data.get("yes_price") or order_data.get("no_price") or 0,
            quantity=order_data.get("initial_count", 0),
            status=OrderStatus(order_data.get("status", "pending")),
            filled_quantity=order_data.get("fill_count", 0),
            remaining_quantity=order_data.get("remaining_count", 0),
        )

    async def get_orders(
        self,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Order]:
        """Get orders."""
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status

        response = await self.get("/portfolio/orders", params)
        orders = []

        for order_data in response.get("orders", []):
            order = Order(
                order_id=order_data["order_id"],
                market_ticker=order_data["ticker"],
                side=Side(order_data["side"]),
                order_type=OrderType(order_data["type"]),
                price=order_data.get("yes_price") or order_data.get("no_price") or 0,
                quantity=order_data.get("initial_count", 0),
                status=OrderStatus(order_data.get("status", "pending")),
                filled_quantity=order_data.get("fill_count", 0),
                remaining_quantity=order_data.get("remaining_count", 0),
            )
            orders.append(order)

        return orders

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order."""
        return await self.delete(f"/portfolio/orders/{order_id}")

    async def cancel_all_orders(self, ticker: str | None = None) -> dict[str, Any]:
        """Cancel all orders, optionally for a specific ticker."""
        params = {"ticker": ticker} if ticker else None
        return await self.delete("/portfolio/orders", params)

    # Historical data endpoints

    async def get_market_candlesticks(
        self,
        ticker: str,
        series_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[Candlestick]:
        """
        Get candlestick OHLC data for a market.

        Args:
            ticker: Market ticker
            series_ticker: Series ticker (e.g., event category)
            start_ts: Start timestamp (Unix epoch seconds)
            end_ts: End timestamp (Unix epoch seconds)
            period_interval: Interval in minutes (1, 60, or 1440)

        Returns:
            List of candlestick data points
        """
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }

        response = await self.get(
            f"/series/{series_ticker}/markets/{ticker}/candlesticks",
            params,
        )

        candlesticks = []
        for candle in response.get("candlesticks", []):
            candlesticks.append(Candlestick(
                ticker=ticker,
                end_period_ts=candle.get("end_period_ts", 0),
                open_price=candle.get("open", 0),
                high_price=candle.get("high", 0),
                low_price=candle.get("low", 0),
                close_price=candle.get("close", 0),
                volume=candle.get("volume", 0),
                open_interest=candle.get("open_interest", 0),
                yes_price=candle.get("yes_price"),
                no_price=candle.get("no_price"),
            ))

        return candlesticks

    async def get_batch_candlesticks(
        self,
        series_ticker: str,
        tickers: list[str],
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict[str, list[Candlestick]]:
        """
        Get candlestick data for multiple markets at once (up to 100).

        Args:
            series_ticker: Series ticker
            tickers: List of market tickers (max 100)
            start_ts: Start timestamp (Unix epoch seconds)
            end_ts: End timestamp (Unix epoch seconds)
            period_interval: Interval in minutes (1, 60, or 1440)

        Returns:
            Dictionary mapping tickers to their candlestick lists
        """
        if len(tickers) > 100:
            logger.warning(f"Batch candlesticks limited to 100 tickers, got {len(tickers)}")
            tickers = tickers[:100]

        params = {
            "tickers": ",".join(tickers),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }

        response = await self.get(
            f"/series/{series_ticker}/markets/candlesticks",
            params,
        )

        result: dict[str, list[Candlestick]] = {}
        for market_data in response.get("markets", []):
            ticker = market_data.get("ticker", "")
            candlesticks = []
            for candle in market_data.get("candlesticks", []):
                candlesticks.append(Candlestick(
                    ticker=ticker,
                    end_period_ts=candle.get("end_period_ts", 0),
                    open_price=candle.get("open", 0),
                    high_price=candle.get("high", 0),
                    low_price=candle.get("low", 0),
                    close_price=candle.get("close", 0),
                    volume=candle.get("volume", 0),
                    open_interest=candle.get("open_interest", 0),
                    yes_price=candle.get("yes_price"),
                    no_price=candle.get("no_price"),
                ))
            result[ticker] = candlesticks

        return result

    async def get_settled_markets(
        self,
        limit: int = 100,
        cursor: str | None = None,
        event_ticker: str | None = None,
    ) -> tuple[list[MarketData], str | None]:
        """
        Get settled (resolved) markets for training data.

        Args:
            limit: Maximum markets per page (max 100)
            cursor: Pagination cursor
            event_ticker: Optional filter by event

        Returns:
            Tuple of (markets list, next cursor or None)
        """
        return await self.get_markets(
            status="settled",
            limit=limit,
            cursor=cursor,
            event_ticker=event_ticker,
        )

    async def get_all_settled_markets(
        self,
        max_pages: int | None = None,
        min_volume: int = 0,
    ) -> list[MarketData]:
        """
        Fetch all settled markets with pagination.

        Args:
            max_pages: Maximum pages to fetch (None for unlimited)
            min_volume: Minimum volume filter

        Returns:
            List of all settled markets
        """
        all_markets: list[MarketData] = []
        cursor: str | None = None
        page_count = 0

        while True:
            markets, cursor = await self.get_settled_markets(
                limit=100, cursor=cursor
            )
            page_count += 1

            for market in markets:
                if min_volume > 0 and market.volume < min_volume:
                    continue
                all_markets.append(market)

            logger.info(
                f"Fetched page {page_count}: {len(markets)} settled markets, "
                f"{len(all_markets)} total"
            )

            if max_pages and page_count >= max_pages:
                break

            if len(markets) < 100 or cursor is None:
                break

        return all_markets

    async def get_historical_cutoff(self) -> dict[str, Any]:
        """
        Get the cutoff date for historical vs live data.

        Returns:
            Dictionary with cutoff timestamp information
        """
        return await self.get("/historical/cutoff")

    async def get_historical_markets(
        self,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[MarketData], str | None]:
        """
        Get archived/historical markets (older than cutoff date).

        Args:
            limit: Maximum markets per page
            cursor: Pagination cursor

        Returns:
            Tuple of (markets list, next cursor or None)
        """
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor

        response = await self.get("/historical/markets", params)
        markets = []

        for market in response.get("markets", []):
            market_data = MarketData(
                ticker=market["ticker"],
                event_ticker=market.get("event_ticker", ""),
                title=market.get("title", ""),
                subtitle=market.get("subtitle"),
                status=market.get("status", "settled"),
                yes_bid=_dollars_to_cents(market.get("yes_bid_dollars")),
                yes_ask=_dollars_to_cents(market.get("yes_ask_dollars")),
                no_bid=_dollars_to_cents(market.get("no_bid_dollars")),
                no_ask=_dollars_to_cents(market.get("no_ask_dollars")),
                last_price=_dollars_to_cents(market.get("last_price_dollars")),
                volume=market.get("volume", 0),
                open_interest=market.get("open_interest", 0),
                close_time=datetime.fromisoformat(market["close_time"].replace("Z", "+00:00"))
                if market.get("close_time")
                else None,
                expiration_time=datetime.fromisoformat(
                    market["expiration_time"].replace("Z", "+00:00")
                )
                if market.get("expiration_time")
                else None,
                result=market.get("result"),
            )
            markets.append(market_data)

        next_cursor = response.get("cursor")
        return markets, next_cursor

    async def get_historical_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> list[Candlestick]:
        """
        Get historical/archived candlestick data (older than 3 months).

        Args:
            ticker: Market ticker
            start_ts: Start timestamp (Unix epoch seconds)
            end_ts: End timestamp (Unix epoch seconds)
            period_interval: Interval in minutes (1, 60, or 1440)

        Returns:
            List of historical candlestick data points
        """
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }

        response = await self.get(
            f"/historical/markets/{ticker}/candlesticks",
            params,
        )

        candlesticks = []
        for candle in response.get("candlesticks", []):
            candlesticks.append(Candlestick(
                ticker=ticker,
                end_period_ts=candle.get("end_period_ts", 0),
                open_price=candle.get("open", 0),
                high_price=candle.get("high", 0),
                low_price=candle.get("low", 0),
                close_price=candle.get("close", 0),
                volume=candle.get("volume", 0),
                open_interest=candle.get("open_interest", 0),
                yes_price=candle.get("yes_price"),
                no_price=candle.get("no_price"),
            ))

        return candlesticks

    # Utility methods

    async def health_check(self) -> bool:
        """Check API connectivity."""
        try:
            await self.get("/exchange/status")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_rate_limiter_status(self) -> dict[str, Any]:
        """Get rate limiter status."""
        return self._rate_limiter.get_status()
