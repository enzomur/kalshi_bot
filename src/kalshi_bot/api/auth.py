"""RSA-PSS signature generation for Kalshi API authentication."""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kalshi_bot.core.exceptions import AuthenticationError
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class KalshiAuth:
    """Handles RSA-PSS authentication for Kalshi API."""

    def __init__(self, api_key_id: str, private_key_path: str) -> None:
        """
        Initialize authentication handler.

        Args:
            api_key_id: Kalshi API key ID
            private_key_path: Path to RSA private key PEM file
        """
        self.api_key_id = api_key_id
        self._private_key: rsa.RSAPrivateKey | None = None
        self._private_key_path = private_key_path

        if private_key_path:
            self._load_private_key(private_key_path)

    def _load_private_key(self, key_path: str) -> None:
        """Load RSA private key from PEM file."""
        path = Path(key_path)

        if not path.exists():
            raise AuthenticationError(
                f"Private key file not found: {key_path}",
                details={"path": key_path},
            )

        try:
            with open(path, "rb") as f:
                key_data = f.read()

            self._private_key = serialization.load_pem_private_key(
                key_data,
                password=None,
            )

            if not isinstance(self._private_key, rsa.RSAPrivateKey):
                raise AuthenticationError(
                    "Invalid key type: expected RSA private key",
                    details={"key_type": type(self._private_key).__name__},
                )

            key_size = self._private_key.key_size
            if key_size < 2048:
                logger.warning(
                    f"RSA key size {key_size} is below recommended 2048 bits"
                )

            logger.info(f"Loaded RSA private key ({key_size} bits) from {key_path}")

        except ValueError as e:
            raise AuthenticationError(
                f"Failed to parse private key: {e}",
                details={"path": key_path, "error": str(e)},
            ) from e

    def generate_signature(
        self, timestamp: int, method: str, path: str
    ) -> str:
        """
        Generate RSA-PSS signature for API request.

        Kalshi requires signatures in the format:
        sign(timestamp + method + path)

        Args:
            timestamp: Unix timestamp in milliseconds
            method: HTTP method (GET, POST, DELETE, etc.)
            path: Request path (e.g., /trade-api/v2/markets)

        Returns:
            Base64-encoded signature
        """
        if self._private_key is None:
            raise AuthenticationError(
                "Private key not loaded",
                details={"api_key_id": self.api_key_id},
            )

        message = f"{timestamp}{method}{path}"
        message_bytes = message.encode("utf-8")

        try:
            signature = self._private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            return base64.b64encode(signature).decode("utf-8")

        except Exception as e:
            raise AuthenticationError(
                f"Failed to generate signature: {e}",
                details={"method": method, "path": path, "error": str(e)},
            ) from e

    def get_auth_headers(self, method: str, path: str) -> dict[str, str]:
        """
        Get authentication headers for an API request.

        Args:
            method: HTTP method
            path: Request path (full path including /trade-api/v2/)

        Returns:
            Dictionary of authentication headers
        """
        timestamp = int(time.time() * 1000)
        signature = self.generate_signature(timestamp, method, path)

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
        }

    @property
    def is_configured(self) -> bool:
        """Check if authentication is properly configured."""
        return bool(self.api_key_id and self._private_key is not None)
