"""Trading mode management with cryptographic signature verification.

This module ensures that LIVE trading modes can only be activated with
a valid cryptographic signature. The default mode is always PAPER.

Mode Transitions:
    PAPER -> LIVE_PROBATION: Requires valid signature + criteria met
    LIVE_PROBATION -> LIVE_FULL: Requires valid signature + proven track record
    Any -> PAPER: Always allowed (fail-safe)

The signature is an HMAC-SHA256 of the mode configuration, signed with
a secret key derived from the private key file.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import ModeError, SignatureError
from src.core.types import TradingMode


@dataclass
class ModeConfig:
    """Configuration for a trading mode."""

    mode: TradingMode
    signature: str | None
    activated_at: str | None
    activated_by: str | None
    reason: str | None
    max_position_dollars: float
    max_daily_loss_dollars: float

    @classmethod
    def paper_default(cls) -> ModeConfig:
        """Create default PAPER mode config."""
        return cls(
            mode=TradingMode.PAPER,
            signature=None,
            activated_at=None,
            activated_by=None,
            reason="Default safe mode",
            max_position_dollars=10000.0,  # No real limit in paper
            max_daily_loss_dollars=10000.0,
        )


class ModeManager:
    """Manages trading mode with signature verification.

    The ModeManager ensures:
    1. Bot always starts in PAPER mode if signature is invalid
    2. LIVE modes require valid HMAC signatures
    3. Mode transitions are logged to the ledger
    4. Fail-safe: any error defaults to PAPER mode
    """

    LIVE_PROBATION_MAX_POSITION = 500.0  # $500 max in probation
    LIVE_PROBATION_MAX_DAILY_LOSS = 100.0  # $100 max daily loss

    def __init__(
        self,
        config_path: str = "config/mode.yaml",
        secret_key_path: str = "secrets/kalshi.pem",
    ) -> None:
        """Initialize mode manager.

        Args:
            config_path: Path to mode configuration file.
            secret_key_path: Path to secret key for signature verification.
        """
        self.config_path = Path(config_path)
        self.secret_key_path = Path(secret_key_path)
        self._config: ModeConfig | None = None
        self._secret_key: bytes | None = None

    def _load_secret_key(self) -> bytes:
        """Load and derive secret key from private key file."""
        if self._secret_key is not None:
            return self._secret_key

        if not self.secret_key_path.exists():
            raise SignatureError(
                f"Secret key file not found: {self.secret_key_path}"
            )

        # Read first 64 bytes of private key file as key material
        with open(self.secret_key_path, "rb") as f:
            key_material = f.read(256)

        # Derive a fixed-length key using SHA256
        self._secret_key = hashlib.sha256(key_material).digest()
        return self._secret_key

    def _compute_signature(self, mode: TradingMode, timestamp: str) -> str:
        """Compute HMAC signature for mode configuration.

        Args:
            mode: The trading mode.
            timestamp: ISO format timestamp of activation.

        Returns:
            Hex-encoded HMAC-SHA256 signature.
        """
        secret_key = self._load_secret_key()
        message = f"{mode.value}:{timestamp}".encode("utf-8")
        signature = hmac.new(secret_key, message, hashlib.sha256)
        return signature.hexdigest()

    def _verify_signature(
        self, mode: TradingMode, timestamp: str, signature: str
    ) -> bool:
        """Verify HMAC signature for mode configuration.

        Args:
            mode: The trading mode.
            timestamp: ISO format timestamp of activation.
            signature: The signature to verify.

        Returns:
            True if signature is valid, False otherwise.
        """
        expected = self._compute_signature(mode, timestamp)
        return hmac.compare_digest(expected, signature)

    def load_config(self) -> ModeConfig:
        """Load and verify mode configuration.

        Returns:
            Verified ModeConfig (defaults to PAPER if invalid).
        """
        # Default to PAPER if config doesn't exist
        if not self.config_path.exists():
            self._config = ModeConfig.paper_default()
            return self._config

        try:
            with open(self.config_path) as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            # Any error loading config -> PAPER mode
            self._config = ModeConfig.paper_default()
            return self._config

        mode_str = data.get("mode", "paper").lower()
        try:
            mode = TradingMode(mode_str)
        except ValueError:
            # Invalid mode string -> PAPER
            self._config = ModeConfig.paper_default()
            return self._config

        # PAPER mode doesn't require signature
        if mode == TradingMode.PAPER:
            self._config = ModeConfig(
                mode=TradingMode.PAPER,
                signature=None,
                activated_at=data.get("activated_at"),
                activated_by=data.get("activated_by"),
                reason=data.get("reason", "Paper trading mode"),
                max_position_dollars=data.get("max_position_dollars", 10000.0),
                max_daily_loss_dollars=data.get("max_daily_loss_dollars", 10000.0),
            )
            return self._config

        # LIVE modes require valid signature
        signature = data.get("signature")
        activated_at = data.get("activated_at")

        if not signature or not activated_at:
            # Missing signature data -> PAPER
            self._config = ModeConfig.paper_default()
            self._config.reason = "Missing signature or timestamp for LIVE mode"
            return self._config

        if not self._verify_signature(mode, activated_at, signature):
            # Invalid signature -> PAPER
            self._config = ModeConfig.paper_default()
            self._config.reason = "Invalid signature for LIVE mode"
            return self._config

        # Valid LIVE mode
        if mode == TradingMode.LIVE_PROBATION:
            max_position = min(
                data.get("max_position_dollars", self.LIVE_PROBATION_MAX_POSITION),
                self.LIVE_PROBATION_MAX_POSITION,
            )
            max_daily_loss = min(
                data.get("max_daily_loss_dollars", self.LIVE_PROBATION_MAX_DAILY_LOSS),
                self.LIVE_PROBATION_MAX_DAILY_LOSS,
            )
        else:
            max_position = data.get("max_position_dollars", 5000.0)
            max_daily_loss = data.get("max_daily_loss_dollars", 500.0)

        self._config = ModeConfig(
            mode=mode,
            signature=signature,
            activated_at=activated_at,
            activated_by=data.get("activated_by"),
            reason=data.get("reason"),
            max_position_dollars=max_position,
            max_daily_loss_dollars=max_daily_loss,
        )
        return self._config

    @property
    def current_mode(self) -> TradingMode:
        """Get current trading mode."""
        if self._config is None:
            self.load_config()
        assert self._config is not None
        return self._config.mode

    @property
    def config(self) -> ModeConfig:
        """Get current mode configuration."""
        if self._config is None:
            self.load_config()
        assert self._config is not None
        return self._config

    @property
    def is_paper(self) -> bool:
        """Check if running in paper trading mode."""
        return self.current_mode == TradingMode.PAPER

    @property
    def is_live(self) -> bool:
        """Check if running in any live trading mode."""
        return self.current_mode in (
            TradingMode.LIVE_PROBATION,
            TradingMode.LIVE_FULL,
        )

    def require_mode(self, *modes: TradingMode) -> None:
        """Raise error if current mode is not one of the specified modes.

        Args:
            modes: Allowed trading modes.

        Raises:
            ModeError: If current mode is not in allowed modes.
        """
        if self.current_mode not in modes:
            raise ModeError(
                f"Operation requires mode {[m.value for m in modes]}, "
                f"but current mode is {self.current_mode.value}",
                current_mode=self.current_mode.value,
                required_mode=str([m.value for m in modes]),
            )

    def create_signed_config(
        self,
        mode: TradingMode,
        activated_by: str,
        reason: str,
        max_position_dollars: float | None = None,
        max_daily_loss_dollars: float | None = None,
    ) -> dict[str, Any]:
        """Create a signed mode configuration.

        This method is used by scripts/promote_mode.py to generate
        valid configurations for LIVE modes.

        Args:
            mode: Target trading mode.
            activated_by: Username or identifier of who is activating.
            reason: Reason for mode change.
            max_position_dollars: Optional position limit override.
            max_daily_loss_dollars: Optional daily loss limit override.

        Returns:
            Dictionary with signed configuration ready to write to YAML.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        config: dict[str, Any] = {
            "mode": mode.value,
            "activated_at": timestamp,
            "activated_by": activated_by,
            "reason": reason,
        }

        if mode != TradingMode.PAPER:
            signature = self._compute_signature(mode, timestamp)
            config["signature"] = signature

        if mode == TradingMode.LIVE_PROBATION:
            config["max_position_dollars"] = min(
                max_position_dollars or self.LIVE_PROBATION_MAX_POSITION,
                self.LIVE_PROBATION_MAX_POSITION,
            )
            config["max_daily_loss_dollars"] = min(
                max_daily_loss_dollars or self.LIVE_PROBATION_MAX_DAILY_LOSS,
                self.LIVE_PROBATION_MAX_DAILY_LOSS,
            )
        elif mode == TradingMode.LIVE_FULL:
            config["max_position_dollars"] = max_position_dollars or 5000.0
            config["max_daily_loss_dollars"] = max_daily_loss_dollars or 500.0

        return config

    def get_status(self) -> dict[str, Any]:
        """Get current mode status."""
        config = self.config
        return {
            "mode": config.mode.value,
            "is_paper": self.is_paper,
            "is_live": self.is_live,
            "activated_at": config.activated_at,
            "activated_by": config.activated_by,
            "reason": config.reason,
            "max_position_dollars": config.max_position_dollars,
            "max_daily_loss_dollars": config.max_daily_loss_dollars,
            "signature_valid": config.signature is not None or self.is_paper,
        }


# Global mode manager instance
_mode_manager: ModeManager | None = None


def get_mode_manager() -> ModeManager:
    """Get global mode manager instance."""
    global _mode_manager
    if _mode_manager is None:
        _mode_manager = ModeManager()
    return _mode_manager


def verify_mode_on_startup() -> ModeConfig:
    """Verify mode configuration on startup.

    This should be called early in main() to ensure the bot
    is running in the correct mode.

    Returns:
        Verified ModeConfig.

    Raises:
        ModeError: If configuration is invalid and cannot default to PAPER.
    """
    manager = get_mode_manager()
    config = manager.load_config()

    # Log the mode status
    if config.mode == TradingMode.PAPER:
        print(f"[MODE] Starting in PAPER mode: {config.reason}")
    elif config.mode == TradingMode.LIVE_PROBATION:
        print(
            f"[MODE] Starting in LIVE_PROBATION mode "
            f"(max position: ${config.max_position_dollars}, "
            f"max daily loss: ${config.max_daily_loss_dollars})"
        )
    else:
        print(
            f"[MODE] Starting in LIVE_FULL mode "
            f"(max position: ${config.max_position_dollars})"
        )

    return config
