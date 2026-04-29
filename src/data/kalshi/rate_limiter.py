"""Token bucket rate limiter with exponential backoff."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from src.core.exceptions import RateLimitError
from src.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimiterState:
    """State for the token bucket algorithm."""

    tokens: float
    last_update: float
    consecutive_rate_limits: int = 0
    backoff_until: float = 0.0


class RateLimiter:
    """
    Token bucket rate limiter with exponential backoff.

    Implements a token bucket algorithm where tokens are added at a constant rate
    and consumed by API requests. When rate limits are hit, exponential backoff
    is applied.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int | None = None,
        max_backoff: float = 60.0,
        base_backoff: float = 1.0,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum sustained request rate
            burst_size: Maximum burst size (defaults to 2x requests_per_second)
            max_backoff: Maximum backoff time in seconds
            base_backoff: Base backoff time for exponential calculation
        """
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size or int(requests_per_second * 2)
        self.max_backoff = max_backoff
        self.base_backoff = base_backoff

        self._state = RateLimiterState(
            tokens=float(self.burst_size),
            last_update=time.monotonic(),
        )
        self._lock = asyncio.Lock()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._state.last_update
        self._state.last_update = now

        new_tokens = elapsed * self.requests_per_second
        self._state.tokens = min(
            self._state.tokens + new_tokens,
            float(self.burst_size),
        )

    def _calculate_backoff(self) -> float:
        """Calculate exponential backoff time."""
        if self._state.consecutive_rate_limits == 0:
            return 0.0

        backoff = self.base_backoff * (2 ** (self._state.consecutive_rate_limits - 1))
        return min(backoff, self.max_backoff)

    async def acquire(self, tokens: int = 1, timeout: float | None = None) -> None:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None for unlimited)

        Raises:
            RateLimitError: If tokens cannot be acquired within timeout
        """
        start_time = time.monotonic()

        async with self._lock:
            while True:
                now = time.monotonic()

                if self._state.backoff_until > now:
                    wait_time = self._state.backoff_until - now
                    if timeout is not None and (now - start_time + wait_time) > timeout:
                        raise RateLimitError(
                            "Rate limit backoff timeout",
                            retry_after=wait_time,
                        )
                    logger.debug(f"Rate limit backoff: waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    continue

                self._refill_tokens()

                if self._state.tokens >= tokens:
                    self._state.tokens -= tokens
                    return

                tokens_needed = tokens - self._state.tokens
                wait_time = tokens_needed / self.requests_per_second

                if timeout is not None:
                    elapsed = now - start_time
                    if elapsed + wait_time > timeout:
                        raise RateLimitError(
                            f"Rate limit timeout: need {wait_time:.2f}s, "
                            f"only {timeout - elapsed:.2f}s remaining",
                            retry_after=wait_time,
                        )

                logger.debug(f"Rate limiter: waiting {wait_time:.2f}s for tokens")
                await asyncio.sleep(wait_time)

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self._lock:
            now = time.monotonic()

            if self._state.backoff_until > now:
                return False

            self._refill_tokens()

            if self._state.tokens >= tokens:
                self._state.tokens -= tokens
                return True

            return False

    async def report_rate_limit(self, retry_after: float | None = None) -> None:
        """
        Report that a rate limit response was received.

        This triggers exponential backoff for subsequent requests.

        Args:
            retry_after: Server-suggested retry time (if provided)
        """
        async with self._lock:
            self._state.consecutive_rate_limits += 1
            backoff = self._calculate_backoff()

            if retry_after is not None:
                backoff = max(backoff, retry_after)

            self._state.backoff_until = time.monotonic() + backoff

            logger.warning(
                f"Rate limit hit (#{self._state.consecutive_rate_limits}), "
                f"backing off for {backoff:.2f}s"
            )

    async def report_success(self) -> None:
        """Report a successful request, resetting the backoff counter."""
        async with self._lock:
            if self._state.consecutive_rate_limits > 0:
                logger.debug("Rate limit backoff reset after successful request")
                self._state.consecutive_rate_limits = 0
                self._state.backoff_until = 0.0

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate, without locking)."""
        self._refill_tokens()
        return self._state.tokens

    @property
    def is_backing_off(self) -> bool:
        """Check if currently in backoff mode."""
        return time.monotonic() < self._state.backoff_until

    def get_status(self) -> dict[str, float | int | bool]:
        """Get current rate limiter status."""
        now = time.monotonic()
        return {
            "tokens_available": self._state.tokens,
            "requests_per_second": self.requests_per_second,
            "burst_size": self.burst_size,
            "is_backing_off": self._state.backoff_until > now,
            "backoff_remaining": max(0.0, self._state.backoff_until - now),
            "consecutive_rate_limits": self._state.consecutive_rate_limits,
        }
