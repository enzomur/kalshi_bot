"""FastAPI dashboard for monitoring and control."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

if TYPE_CHECKING:
    from kalshi_bot.bot import KalshiArbitrageBot


class StatusResponse(BaseModel):
    """Bot status response."""

    running: bool
    timestamp: str
    portfolio: dict[str, Any] | None
    risk: dict[str, Any] | None


class PortfolioResponse(BaseModel):
    """Portfolio status response."""

    balance: float
    positions_count: int
    positions_value: float
    total_value: float
    tradeable_balance: float
    profit_locked: bool
    locked_principal: float
    drawdown: float


class TradeResponse(BaseModel):
    """Trade record response."""

    trade_id: str
    market_ticker: str
    side: str
    price: int
    quantity: int
    fee: float
    executed_at: str


class ControlAction(BaseModel):
    """Control action request."""

    action: str
    params: dict[str, Any] | None = None


def create_app(bot: KalshiArbitrageBot | None = None) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        bot: Bot instance for accessing state

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Kalshi Arbitrage Bot",
        description="Dashboard for monitoring and controlling the Kalshi arbitrage trading bot",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.bot = bot

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint."""
        return {
            "name": "Kalshi Arbitrage Bot",
            "version": "0.1.0",
            "status": "running" if app.state.bot and app.state.bot.is_running else "stopped",
        }

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status() -> StatusResponse:
        """Get comprehensive bot status."""
        if not app.state.bot:
            raise HTTPException(status_code=503, detail="Bot not initialized")

        status = app.state.bot.get_status()

        return StatusResponse(
            running=status.get("running", False),
            timestamp=datetime.utcnow().isoformat(),
            portfolio=status.get("portfolio"),
            risk=status.get("risk"),
        )

    @app.get("/api/portfolio")
    async def get_portfolio() -> dict[str, Any]:
        """Get portfolio details."""
        if not app.state.bot or not app.state.bot.portfolio_manager:
            raise HTTPException(status_code=503, detail="Portfolio manager not available")

        return app.state.bot.portfolio_manager.get_status()

    @app.get("/api/positions")
    async def get_positions() -> list[dict[str, Any]]:
        """Get current positions."""
        if not app.state.bot or not app.state.bot.portfolio_manager:
            raise HTTPException(status_code=503, detail="Portfolio manager not available")

        positions = app.state.bot.portfolio_manager.positions
        return [p.to_dict() for p in positions]

    @app.get("/api/trades")
    async def get_trades(limit: int = 50) -> list[dict[str, Any]]:
        """Get recent executed trades (from opportunities with execution results)."""
        if not app.state.bot or not app.state.bot._opportunity_repo:
            raise HTTPException(status_code=503, detail="Opportunity repository not available")

        # Return opportunities that were executed (have executed_at timestamp)
        opportunities = await app.state.bot._opportunity_repo.get_recent(limit * 2, status=None)
        executed = [o for o in opportunities if o.get("executed_at") is not None]
        return executed[:limit]

    @app.get("/api/opportunities")
    async def get_opportunities(limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        """Get recent arbitrage opportunities."""
        if not app.state.bot or not app.state.bot._opportunity_repo:
            raise HTTPException(status_code=503, detail="Opportunity repository not available")

        opportunities = await app.state.bot._opportunity_repo.get_recent(limit, status)
        return opportunities

    @app.get("/api/risk")
    async def get_risk_status() -> dict[str, Any]:
        """Get risk management status."""
        if not app.state.bot or not app.state.bot.risk_manager:
            raise HTTPException(status_code=503, detail="Risk manager not available")

        return app.state.bot.risk_manager.get_status()

    @app.get("/api/circuit-breakers")
    async def get_circuit_breakers() -> dict[str, Any]:
        """Get circuit breaker status."""
        if not app.state.bot or not app.state.bot._circuit_breaker:
            raise HTTPException(status_code=503, detail="Circuit breaker not available")

        return app.state.bot._circuit_breaker.get_all_status()

    @app.post("/api/controls")
    async def execute_control(action: ControlAction) -> dict[str, Any]:
        """Execute a control action."""
        if not app.state.bot:
            raise HTTPException(status_code=503, detail="Bot not initialized")

        if action.action == "stop":
            await app.state.bot.stop()
            return {"status": "success", "message": "Bot stopped"}

        elif action.action == "emergency_stop":
            if app.state.bot.risk_manager:
                reason = (action.params or {}).get("reason", "Manual emergency stop")
                await app.state.bot.risk_manager.emergency_stop(reason)
            return {"status": "success", "message": "Emergency stop activated"}

        elif action.action == "clear_emergency_stop":
            if app.state.bot.risk_manager:
                await app.state.bot.risk_manager.clear_emergency_stop()
            return {"status": "success", "message": "Emergency stop cleared"}

        elif action.action == "reset_circuit_breaker":
            if app.state.bot._circuit_breaker:
                breaker_type = (action.params or {}).get("breaker_type")
                if breaker_type:
                    from kalshi_bot.risk.circuit_breaker import BreakerType
                    await app.state.bot._circuit_breaker.manual_reset(BreakerType(breaker_type))
                else:
                    await app.state.bot._circuit_breaker.manual_reset()
            return {"status": "success", "message": "Circuit breaker reset"}

        elif action.action == "cancel_all_orders":
            if app.state.bot._executor:
                cancelled = await app.state.bot._executor.cancel_all_pending()
                return {"status": "success", "message": f"Cancelled {cancelled} orders"}
            return {"status": "error", "message": "Executor not available"}

        elif action.action == "sync_portfolio":
            if app.state.bot.portfolio_manager:
                await app.state.bot.portfolio_manager.sync_with_kalshi()
            return {"status": "success", "message": "Portfolio synced"}

        elif action.action == "lock_profits":
            if app.state.bot._profit_lock:
                await app.state.bot._profit_lock.manual_lock()
            return {"status": "success", "message": "Profits locked"}

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown action: {action.action}",
            )

    @app.get("/api/audit-log")
    async def get_audit_log(
        limit: int = 100,
        event_type: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get audit log entries."""
        if not app.state.bot or not app.state.bot._audit_repo:
            raise HTTPException(status_code=503, detail="Audit repository not available")

        return await app.state.bot._audit_repo.get_recent(limit, event_type, severity)

    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        if not app.state.bot:
            return {"status": "unhealthy", "reason": "Bot not initialized"}

        if not app.state.bot.is_running:
            return {"status": "unhealthy", "reason": "Bot not running"}

        return {"status": "healthy"}

    @app.get("/api/opportunities/for-analysis")
    async def get_opportunities_for_analysis(
        min_volume: int = 100,
        max_spread: int = 10,
        max_days: int = 30,
        min_price: int = 20,
        max_price: int = 80,
        limit: int = 20,
    ) -> dict[str, Any]:
        """
        Get markets formatted for Claude Code analysis.

        Returns filtered market candidates for manual analysis.

        Query params:
            min_volume: Minimum contract volume
            max_spread: Maximum bid/ask spread in cents
            max_days: Maximum days until expiration
            min_price: Minimum YES price in cents
            max_price: Maximum YES price in cents
            limit: Maximum number of markets to return
        """
        if not app.state.bot or not app.state.bot._arbitrage_detector:
            raise HTTPException(status_code=503, detail="Arbitrage detector not available")

        from kalshi_bot.analysis.opportunity_exporter import OpportunityExporter

        # Get cached markets from detector
        markets = app.state.bot._arbitrage_detector._markets_cache

        if not markets:
            # Try to refresh if cache is empty
            await app.state.bot._arbitrage_detector.refresh_market_data(force=True)
            markets = app.state.bot._arbitrage_detector._markets_cache

        exporter = OpportunityExporter(
            min_volume=min_volume,
            max_spread_cents=max_spread,
            max_days_to_expiry=max_days,
            price_range=(min_price, max_price),
        )

        candidates = exporter.filter_candidates(markets)[:limit]

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_markets": len(markets),
            "candidates_found": len(exporter.filter_candidates(markets)),
            "candidates_returned": len(candidates),
            "filters": {
                "min_volume": min_volume,
                "max_spread": max_spread,
                "max_days_to_expiry": max_days,
                "price_range": [min_price, max_price],
            },
            "markets": [m.to_dict() for m in candidates],
        }

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        """Get non-sensitive configuration."""
        if not app.state.bot:
            raise HTTPException(status_code=503, detail="Bot not initialized")

        settings = app.state.bot.settings
        return {
            "environment": settings.environment.value,
            "trading": {
                "min_profit_cents": settings.trading.min_profit_cents,
                "max_position_size": settings.trading.max_position_size,
                "max_position_pct": settings.trading.max_position_pct,
                "kelly_fraction": settings.trading.kelly_fraction,
            },
            "risk": {
                "max_drawdown": settings.risk.max_drawdown,
                "max_daily_loss": settings.risk.max_daily_loss,
                "max_consecutive_failures": settings.risk.max_consecutive_failures,
            },
            "portfolio": {
                "initial_principal": settings.portfolio.initial_principal,
                "profit_lock_multiplier": settings.portfolio.profit_lock_multiplier,
            },
            "arbitrage": {
                "enable_single_market": settings.arbitrage.enable_single_market,
                "enable_multi_outcome": settings.arbitrage.enable_multi_outcome,
                "enable_cross_market": settings.arbitrage.enable_cross_market,
            },
            "ml": {
                "enabled": settings.ml.enabled,
                "min_edge_threshold": settings.ml.min_edge_threshold,
                "min_confidence": settings.ml.min_confidence,
                "min_settlements_for_training": settings.ml.min_settlements_for_training,
            },
        }

    # ===================
    # ML Endpoints
    # ===================

    @app.get("/api/ml/status")
    async def get_ml_status() -> dict[str, Any]:
        """Get ML system status including model info, accuracy, and data collection stats."""
        if not app.state.bot:
            raise HTTPException(status_code=503, detail="Bot not initialized")

        bot = app.state.bot
        status = bot.get_status()

        # Get training status
        training_status = None
        if bot.model_trainer:
            training_status = await bot.model_trainer.get_training_status()

        # Get prediction accuracy
        accuracy_stats = None
        if bot.edge_predictor:
            accuracy_stats = await bot.edge_predictor.get_accuracy_stats()

        # Get snapshot collection stats
        snapshot_status = None
        if bot._snapshot_collector:
            snapshot_status = bot._snapshot_collector.get_status()

        # Get outcome tracker stats
        outcome_status = None
        if bot._outcome_tracker:
            outcome_status = bot._outcome_tracker.get_status()

        return {
            "ml_enabled": bot.settings.ml.enabled,
            "ml_status": status.get("ml"),
            "training": training_status,
            "accuracy": accuracy_stats,
            "data_collection": {
                "snapshots": snapshot_status,
                "outcomes": outcome_status,
            },
        }

    @app.get("/api/ml/predictions")
    async def get_ml_predictions(
        limit: int = 50,
        settled_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Get recent ML predictions with outcomes."""
        if not app.state.bot or not app.state.bot.edge_predictor:
            raise HTTPException(status_code=503, detail="Edge predictor not available")

        return await app.state.bot.edge_predictor.get_recent_predictions(
            limit=limit,
            settled_only=settled_only,
        )

    @app.post("/api/ml/train")
    async def trigger_ml_training(
        model_type: str = "logistic",
    ) -> dict[str, Any]:
        """Trigger manual model retraining."""
        if not app.state.bot or not app.state.bot.training_scheduler:
            raise HTTPException(status_code=503, detail="Training scheduler not available")

        result = await app.state.bot.training_scheduler.force_training(model_type)
        return result

    @app.post("/api/ml/enable")
    async def enable_ml_trading() -> dict[str, Any]:
        """Enable ML trading strategy."""
        if not app.state.bot:
            raise HTTPException(status_code=503, detail="Bot not initialized")

        if not app.state.bot.strategy_disabler:
            raise HTTPException(status_code=503, detail="Strategy disabler not available")

        await app.state.bot.strategy_disabler.enable_strategy()
        return {"status": "success", "message": "ML trading enabled"}

    @app.post("/api/ml/disable")
    async def disable_ml_trading(reason: str = "Manual disable") -> dict[str, Any]:
        """Disable ML trading strategy."""
        if not app.state.bot:
            raise HTTPException(status_code=503, detail="Bot not initialized")

        if not app.state.bot.strategy_disabler:
            raise HTTPException(status_code=503, detail="Strategy disabler not available")

        await app.state.bot.strategy_disabler.disable_strategy_manual(reason)
        return {"status": "success", "message": f"ML trading disabled: {reason}"}

    @app.get("/api/ml/settlements")
    async def get_settlements(
        limit: int = 50,
        outcome: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent market settlements."""
        if not app.state.bot or not app.state.bot._outcome_tracker:
            raise HTTPException(status_code=503, detail="Outcome tracker not available")

        return await app.state.bot._outcome_tracker.get_recent_settlements(
            limit=limit,
            outcome=outcome,
        )

    @app.get("/api/ml/models")
    async def get_ml_models() -> dict[str, Any]:
        """Get information about trained models."""
        if not app.state.bot or not app.state.bot._db:
            raise HTTPException(status_code=503, detail="Database not available")

        db = app.state.bot._db

        # Get all models
        models = await db.fetch_all(
            """
            SELECT model_id, model_type, status, is_active, metrics, trained_at
            FROM ml_models
            ORDER BY trained_at DESC
            LIMIT 20
            """
        )

        # Parse metrics JSON
        import json
        for model in models:
            if model.get("metrics"):
                try:
                    model["metrics"] = json.loads(model["metrics"])
                except json.JSONDecodeError:
                    pass

        return {
            "models": models,
            "active_count": sum(1 for m in models if m.get("is_active")),
            "total_count": len(models),
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    return app
