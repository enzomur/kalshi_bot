"""Pydantic settings with validation for Kalshi bot configuration."""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Kalshi API environment."""

    DEMO = "demo"
    PRODUCTION = "production"


class TradingSettings(BaseModel):
    """Trading parameters."""

    min_profit_cents: int = Field(default=2, ge=1, le=100)
    max_position_size: int = Field(default=100, ge=1, le=10000)
    max_position_pct: float = Field(default=0.10, ge=0.01, le=0.50)
    kelly_fraction: float = Field(default=0.25, ge=0.05, le=1.0)
    execution_confidence: float = Field(default=0.9, ge=0.5, le=1.0)
    min_edge: float = Field(default=0.005, ge=0.001, le=0.10)

    # Execution safeguards
    max_slippage_cents: int = Field(default=2, ge=1, le=10)
    min_net_profit_pct: float = Field(default=0.02, ge=0.001, le=0.10)

    # Paper trading mode - enabled by default for safety
    paper_trading_mode: bool = Field(default=True)


class RiskSettings(BaseModel):
    """Risk management parameters."""

    max_drawdown: float = Field(default=0.20, ge=0.05, le=0.50)
    max_daily_loss: float = Field(default=100.0, ge=1.0, le=10000.0)
    max_consecutive_failures: int = Field(default=5, ge=1, le=20)
    min_success_rate: float = Field(default=0.40, ge=0.20, le=0.90)
    success_rate_window: int = Field(default=20, ge=5, le=100)

    # Enhanced circuit breaker settings
    circuit_breaker_cooldown: int = Field(default=1800, ge=60, le=7200)  # 30 min default
    circuit_breaker_reset_count: int = Field(default=3, ge=1, le=10)  # 3 successes to reset
    portfolio_stop_threshold: float = Field(default=0.50, ge=0.10, le=0.90)  # 50% loss = hard stop


class PortfolioSettings(BaseModel):
    """Portfolio management parameters."""

    initial_principal: float = Field(default=1000.0, ge=10.0)
    profit_lock_multiplier: float = Field(default=2.0, ge=1.5, le=10.0)
    sync_interval: int = Field(default=60, ge=10, le=600)
    take_profit_pct: float = Field(default=1.0, ge=0.1, le=10.0)


class ArbitrageSettings(BaseModel):
    """Arbitrage detection parameters."""

    # TRUE ARBITRAGE ONLY - speculative strategies removed
    enable_single_market: bool = True  # YES + NO < 100 - TRUE ARBITRAGE
    enable_multi_outcome: bool = True  # Sum of outcomes < 100 - TRUE ARBITRAGE
    enable_cross_market: bool = False  # Correlated markets - DISABLED by default (risky)
    allow_dynamic_detection: bool = False  # Dynamic cross-market detection - DISABLED

    min_liquidity: int = Field(default=5, ge=1, le=1000)
    max_spread: int = Field(default=8, ge=1, le=20)


class APISettings(BaseModel):
    """API connection parameters."""

    timeout: int = Field(default=30, ge=5, le=120)
    requests_per_second: int = Field(default=10, ge=1, le=50)
    ws_reconnect_delay: int = Field(default=5, ge=1, le=60)
    ws_max_reconnects: int = Field(default=10, ge=1, le=100)


class DashboardSettings(BaseModel):
    """Dashboard settings."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1024, le=65535)


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    file: str | None = "logs/kalshi_bot.log"
    json_format: bool = False
    max_file_size_mb: int = Field(default=10, ge=1, le=100)
    backup_count: int = Field(default=5, ge=1, le=20)

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper


class MLSettings(BaseModel):
    """Machine learning settings for self-learning bot."""

    # Master enable/disable switch - disabled until enough data collected
    enabled: bool = Field(default=False)

    # Data collection
    snapshot_interval_seconds: int = Field(default=300, ge=60, le=3600)  # 5 min default
    min_volume_for_snapshot: int = Field(default=0, ge=0)
    min_open_interest_for_snapshot: int = Field(default=0, ge=0)

    # Trading parameters
    min_edge_threshold: float = Field(default=0.05, ge=0.01, le=0.30)  # 5% edge required
    min_confidence: float = Field(default=0.60, ge=0.50, le=0.95)
    max_kelly_fraction: float = Field(default=0.25, ge=0.05, le=0.50)

    # Training parameters
    min_settlements_for_training: int = Field(default=100, ge=50, le=1000)
    min_snapshots_per_market: int = Field(default=10, ge=5, le=100)
    retraining_settlement_threshold: int = Field(default=100, ge=25, le=500)
    retraining_days_threshold: int = Field(default=7, ge=1, le=30)

    # Model selection
    default_model_type: str = Field(default="logistic")  # logistic or gradient_boost
    gradient_boost_min_settlements: int = Field(default=1000, ge=500)

    # Self-correction thresholds
    accuracy_threshold_full: float = Field(default=0.60, ge=0.50, le=0.80)
    accuracy_threshold_half: float = Field(default=0.50, ge=0.40, le=0.70)
    accuracy_threshold_disable: float = Field(default=0.45, ge=0.35, le=0.55)
    consecutive_loss_disable: int = Field(default=5, ge=3, le=10)
    daily_loss_disable: float = Field(default=50.0, ge=10.0, le=500.0)
    drawdown_disable: float = Field(default=0.20, ge=0.10, le=0.40)

    @field_validator("default_model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        valid_types = {"logistic", "gradient_boost"}
        if v not in valid_types:
            raise ValueError(f"Invalid model type: {v}. Must be one of {valid_types}")
        return v


class WeatherResearchSettings(BaseModel):
    """Settings for the Weather Research Agent."""

    enabled: bool = Field(default=True)
    update_interval_minutes: int = Field(default=15, ge=5, le=60)
    enabled_locations: list[str] = Field(
        default_factory=lambda: ["NYC", "CHI", "LAX", "MIA"]
    )

    @field_validator("enabled_locations")
    @classmethod
    def validate_locations(cls, v: list[str]) -> list[str]:
        valid_locations = {
            "NYC", "CHI", "LAX", "MIA", "DFW", "PHX", "HOU", "ATL",
            "BOS", "SEA", "DEN", "PHL", "SFO", "DCA", "MSP"
        }
        for loc in v:
            if loc not in valid_locations:
                raise ValueError(f"Invalid location: {loc}. Must be one of {valid_locations}")
        return v


class SignalTesterSettings(BaseModel):
    """Settings for the Signal Tester Agent."""

    enabled: bool = Field(default=True)
    backtest_days: int = Field(default=90, ge=30, le=365)
    required_win_rate: float = Field(default=0.55, ge=0.50, le=0.80)
    update_interval_hours: int = Field(default=24, ge=1, le=168)


class WeatherRiskSettings(BaseModel):
    """Settings for the Weather Risk Agent."""

    enabled: bool = Field(default=True)
    max_weather_exposure_pct: float = Field(default=0.30, ge=0.10, le=0.50)
    max_single_location_pct: float = Field(default=0.15, ge=0.05, le=0.30)
    min_forecast_confidence: float = Field(default=0.60, ge=0.40, le=0.90)


class AgentsSettings(BaseModel):
    """Settings for all trading agents."""

    weather_research: WeatherResearchSettings = Field(
        default_factory=WeatherResearchSettings
    )
    signal_tester: SignalTesterSettings = Field(
        default_factory=SignalTesterSettings
    )
    weather_risk: WeatherRiskSettings = Field(
        default_factory=WeatherRiskSettings
    )


class Settings(BaseSettings):
    """Main settings class combining env vars and YAML config."""

    model_config = SettingsConfigDict(
        env_prefix="KALSHI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment variables (secrets)
    api_key_id: str = Field(default="", description="Kalshi API Key ID")
    private_key_path: str = Field(default="", description="Path to RSA private key")
    environment: Environment = Field(default=Environment.DEMO)

    # Optional env overrides
    database_path: str = Field(default="data/kalshi_bot.db")
    settings_path: str = Field(default="settings.yaml")
    log_level: str | None = Field(default=None)

    # YAML config sections
    trading: TradingSettings = Field(default_factory=TradingSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    portfolio: PortfolioSettings = Field(default_factory=PortfolioSettings)
    arbitrage: ArbitrageSettings = Field(default_factory=ArbitrageSettings)
    api: APISettings = Field(default_factory=APISettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    agents: AgentsSettings = Field(default_factory=AgentsSettings)

    @property
    def api_base_url(self) -> str:
        """Get the appropriate API base URL for the environment."""
        if self.environment == Environment.DEMO:
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://api.elections.kalshi.com/trade-api/v2"

    @property
    def ws_base_url(self) -> str:
        """Get the appropriate WebSocket URL for the environment."""
        if self.environment == Environment.DEMO:
            return "wss://demo-api.kalshi.co/trade-api/ws/v2"
        return "wss://api.elections.kalshi.com/trade-api/ws/v2"

    @model_validator(mode="before")
    @classmethod
    def load_yaml_config(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Load settings from YAML file and merge with env vars."""
        settings_path = data.get("settings_path") or os.getenv(
            "KALSHI_SETTINGS_PATH", "settings.yaml"
        )

        yaml_config: dict[str, Any] = {}
        if Path(settings_path).exists():
            with open(settings_path) as f:
                yaml_config = yaml.safe_load(f) or {}

        # Merge YAML config with env vars (env vars take precedence)
        for key, value in yaml_config.items():
            if key not in data or data[key] is None:
                data[key] = value

        return data

    @field_validator("private_key_path")
    @classmethod
    def validate_private_key_path(cls, v: str) -> str:
        """Validate that private key file exists (if provided)."""
        if v and not Path(v).exists():
            raise ValueError(f"Private key file not found: {v}")
        return v

    def validate_for_trading(self) -> None:
        """Validate that all required settings are present for trading."""
        errors = []

        if not self.api_key_id:
            errors.append("KALSHI_API_KEY_ID is required")

        if not self.private_key_path:
            errors.append("KALSHI_PRIVATE_KEY_PATH is required")
        elif not Path(self.private_key_path).exists():
            errors.append(f"Private key file not found: {self.private_key_path}")

        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    def get_effective_log_level(self) -> str:
        """Get log level with env var override."""
        return self.log_level or self.logging.level


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
