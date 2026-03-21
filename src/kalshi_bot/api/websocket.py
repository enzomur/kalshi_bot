"""WebSocket manager with auto-reconnect for Kalshi streaming API."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

import websockets
from websockets.client import WebSocketClientProtocol

from kalshi_bot.api.auth import KalshiAuth
from kalshi_bot.config.settings import Settings
from kalshi_bot.core.exceptions import WebSocketError
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class SubscriptionType(str, Enum):
    """WebSocket subscription types."""

    ORDERBOOK = "orderbook_delta"
    TICKER = "ticker"
    TRADE = "trade"
    FILL = "fill"
    ORDER = "order_update"


@dataclass
class Subscription:
    """Represents a WebSocket subscription."""

    sub_type: SubscriptionType
    market_ticker: str | None = None
    callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None


@dataclass
class WebSocketState:
    """WebSocket connection state."""

    connected: bool = False
    reconnect_attempts: int = 0
    last_message_time: float = 0.0
    subscriptions: dict[str, Subscription] = field(default_factory=dict)


class WebSocketManager:
    """
    Manages WebSocket connections to Kalshi streaming API.

    Handles automatic reconnection, subscription management, and message routing.
    """

    def __init__(self, settings: Settings) -> None:
        """
        Initialize WebSocket manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.base_url = settings.ws_base_url
        self._auth = KalshiAuth(settings.api_key_id, settings.private_key_path)

        self._state = WebSocketState()
        self._ws: WebSocketClientProtocol | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._reconnect_lock = asyncio.Lock()
        self._message_handlers: dict[str, Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = {}
        self._shutdown = False

    async def connect(self) -> None:
        """Establish WebSocket connection."""
        if self._state.connected:
            return

        async with self._reconnect_lock:
            if self._state.connected:
                return

            try:
                timestamp = int(time.time() * 1000)
                path = "/trade-api/ws/v2"
                signature = self._auth.generate_signature(timestamp, "GET", path)

                headers = {
                    "KALSHI-ACCESS-KEY": self._auth.api_key_id,
                    "KALSHI-ACCESS-SIGNATURE": signature,
                    "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
                }

                self._ws = await websockets.connect(
                    self.base_url,
                    additional_headers=headers,
                    ping_interval=30,
                    ping_timeout=10,
                )

                self._state.connected = True
                self._state.reconnect_attempts = 0
                self._state.last_message_time = time.time()

                self._receive_task = asyncio.create_task(self._receive_loop())
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                logger.info("WebSocket connected")

                await self._resubscribe_all()

            except Exception as e:
                logger.error(f"WebSocket connection failed: {e}")
                raise WebSocketError(
                    f"Failed to connect: {e}",
                    reconnect_attempt=self._state.reconnect_attempts,
                )

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._shutdown = True

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._state.connected = False
        logger.info("WebSocket disconnected")

    async def _receive_loop(self) -> None:
        """Main receive loop for WebSocket messages."""
        while not self._shutdown and self._ws:
            try:
                message = await self._ws.recv()
                self._state.last_message_time = time.time()

                if isinstance(message, bytes):
                    message = message.decode("utf-8")

                data = json.loads(message)
                await self._handle_message(data)

            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                await self._handle_disconnect()
                break

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse WebSocket message: {e}")

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and check connection health."""
        while not self._shutdown and self._state.connected:
            try:
                await asyncio.sleep(15)

                if not self._state.connected:
                    break

                stale_threshold = 60
                if time.time() - self._state.last_message_time > stale_threshold:
                    logger.warning("WebSocket connection appears stale, reconnecting")
                    await self._handle_disconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

    async def _handle_disconnect(self) -> None:
        """Handle disconnection and attempt reconnect."""
        self._state.connected = False

        if self._shutdown:
            return

        max_reconnects = self.settings.api.ws_max_reconnects
        reconnect_delay = self.settings.api.ws_reconnect_delay

        while self._state.reconnect_attempts < max_reconnects and not self._shutdown:
            self._state.reconnect_attempts += 1
            delay = reconnect_delay * (2 ** (self._state.reconnect_attempts - 1))
            delay = min(delay, 60)

            logger.info(
                f"Attempting reconnect {self._state.reconnect_attempts}/{max_reconnects} "
                f"in {delay}s"
            )

            await asyncio.sleep(delay)

            try:
                await self.connect()
                return
            except WebSocketError:
                continue

        if not self._shutdown:
            logger.error("Max reconnection attempts reached")
            raise WebSocketError(
                "Max reconnection attempts reached",
                reconnect_attempt=self._state.reconnect_attempts,
            )

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """Route incoming message to appropriate handler."""
        msg_type = data.get("type")

        if msg_type == "subscribed":
            logger.debug(f"Subscription confirmed: {data}")
            return

        if msg_type == "unsubscribed":
            logger.debug(f"Unsubscription confirmed: {data}")
            return

        if msg_type == "error":
            logger.error(f"WebSocket error: {data}")
            return

        channel = data.get("channel")
        if channel and channel in self._message_handlers:
            try:
                await self._message_handlers[channel](data)
            except Exception as e:
                logger.error(f"Error handling message for {channel}: {e}")

        for sub_id, subscription in self._state.subscriptions.items():
            if subscription.callback:
                ticker = data.get("msg", {}).get("ticker")
                if subscription.market_ticker is None or ticker == subscription.market_ticker:
                    try:
                        await subscription.callback(data)
                    except Exception as e:
                        logger.error(f"Error in subscription callback {sub_id}: {e}")

    async def subscribe(
        self,
        sub_type: SubscriptionType,
        market_ticker: str | None = None,
        callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
    ) -> str:
        """
        Subscribe to a channel.

        Args:
            sub_type: Type of subscription
            market_ticker: Market ticker for market-specific subscriptions
            callback: Async callback for received messages

        Returns:
            Subscription ID
        """
        sub_id = f"{sub_type.value}:{market_ticker or 'all'}"

        subscription = Subscription(
            sub_type=sub_type,
            market_ticker=market_ticker,
            callback=callback,
        )
        self._state.subscriptions[sub_id] = subscription

        if self._state.connected and self._ws:
            await self._send_subscribe(subscription)

        return sub_id

    async def unsubscribe(self, sub_id: str) -> None:
        """
        Unsubscribe from a channel.

        Args:
            sub_id: Subscription ID returned from subscribe()
        """
        if sub_id not in self._state.subscriptions:
            return

        subscription = self._state.subscriptions.pop(sub_id)

        if self._state.connected and self._ws:
            await self._send_unsubscribe(subscription)

    async def _send_subscribe(self, subscription: Subscription) -> None:
        """Send subscription message."""
        if not self._ws:
            return

        message: dict[str, Any] = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": [subscription.sub_type.value],
            },
        }

        if subscription.market_ticker:
            message["params"]["market_ticker"] = subscription.market_ticker

        await self._ws.send(json.dumps(message))
        logger.debug(f"Sent subscription: {subscription.sub_type.value}")

    async def _send_unsubscribe(self, subscription: Subscription) -> None:
        """Send unsubscription message."""
        if not self._ws:
            return

        message: dict[str, Any] = {
            "id": 1,
            "cmd": "unsubscribe",
            "params": {
                "channels": [subscription.sub_type.value],
            },
        }

        if subscription.market_ticker:
            message["params"]["market_ticker"] = subscription.market_ticker

        await self._ws.send(json.dumps(message))
        logger.debug(f"Sent unsubscription: {subscription.sub_type.value}")

    async def _resubscribe_all(self) -> None:
        """Resubscribe to all channels after reconnect."""
        for subscription in self._state.subscriptions.values():
            try:
                await self._send_subscribe(subscription)
            except Exception as e:
                logger.error(f"Failed to resubscribe: {e}")

    def register_handler(
        self,
        channel: str,
        handler: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register a message handler for a channel.

        Args:
            channel: Channel name
            handler: Async handler function
        """
        self._message_handlers[channel] = handler

    def unregister_handler(self, channel: str) -> None:
        """Unregister a message handler."""
        self._message_handlers.pop(channel, None)

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._state.connected

    def get_status(self) -> dict[str, Any]:
        """Get WebSocket status."""
        return {
            "connected": self._state.connected,
            "reconnect_attempts": self._state.reconnect_attempts,
            "last_message_time": self._state.last_message_time,
            "subscriptions": list(self._state.subscriptions.keys()),
            "time_since_last_message": time.time() - self._state.last_message_time
            if self._state.last_message_time > 0
            else None,
        }
