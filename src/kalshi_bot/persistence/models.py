"""Repository classes for database operations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from kalshi_bot.core.types import (
    ArbitrageOpportunity,
    ArbitrageType,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    Trade,
)
from kalshi_bot.persistence.database import Database
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class AuditRepository:
    """Repository for audit log entries."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def log(
        self,
        event_type: str,
        event_data: dict[str, Any],
        severity: str = "info",
        component: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Log an audit event."""
        await self.db.execute(
            """
            INSERT INTO audit_log (event_type, event_data, severity, component, correlation_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_type, json.dumps(event_data), severity, component, correlation_id),
        )

    async def get_recent(
        self,
        limit: int = 100,
        event_type: str | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent audit log entries."""
        query = "SELECT * FROM audit_log WHERE 1=1"
        params: list[Any] = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if severity:
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = await self.db.fetch_all(query, tuple(params))

        for row in rows:
            row["event_data"] = json.loads(row["event_data"])

        return rows


class TradeRepository:
    """Repository for trade records."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(self, trade: Trade, opportunity_id: str | None = None) -> None:
        """Save a trade."""
        await self.db.execute(
            """
            INSERT INTO trades (trade_id, order_id, opportunity_id, market_ticker, side, price, quantity, fee, total_cost, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.trade_id,
                trade.order_id,
                opportunity_id,
                trade.market_ticker,
                trade.side.value,
                trade.price,
                trade.quantity,
                trade.fee,
                trade.total_cost,
                trade.executed_at.isoformat(),
            ),
        )

    async def get_by_id(self, trade_id: str) -> Trade | None:
        """Get trade by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM trades WHERE trade_id = ?",
            (trade_id,),
        )

        if row is None:
            return None

        return Trade(
            trade_id=row["trade_id"],
            order_id=row["order_id"],
            market_ticker=row["market_ticker"],
            side=Side(row["side"]),
            price=row["price"],
            quantity=row["quantity"],
            fee=row["fee"],
            executed_at=datetime.fromisoformat(row["executed_at"]),
        )

    async def get_recent(
        self,
        limit: int = 100,
        market_ticker: str | None = None,
    ) -> list[Trade]:
        """Get recent trades."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []

        if market_ticker:
            query += " AND market_ticker = ?"
            params.append(market_ticker)

        query += " ORDER BY executed_at DESC LIMIT ?"
        params.append(limit)

        rows = await self.db.fetch_all(query, tuple(params))

        return [
            Trade(
                trade_id=row["trade_id"],
                order_id=row["order_id"],
                market_ticker=row["market_ticker"],
                side=Side(row["side"]),
                price=row["price"],
                quantity=row["quantity"],
                fee=row["fee"],
                executed_at=datetime.fromisoformat(row["executed_at"]),
            )
            for row in rows
        ]

    async def get_by_opportunity(self, opportunity_id: str) -> list[Trade]:
        """Get trades for an opportunity."""
        rows = await self.db.fetch_all(
            "SELECT * FROM trades WHERE opportunity_id = ? ORDER BY executed_at",
            (opportunity_id,),
        )

        return [
            Trade(
                trade_id=row["trade_id"],
                order_id=row["order_id"],
                market_ticker=row["market_ticker"],
                side=Side(row["side"]),
                price=row["price"],
                quantity=row["quantity"],
                fee=row["fee"],
                executed_at=datetime.fromisoformat(row["executed_at"]),
            )
            for row in rows
        ]

    async def get_daily_stats(self, date: str) -> dict[str, Any]:
        """Get trade statistics for a specific date."""
        rows = await self.db.fetch_all(
            """
            SELECT
                COUNT(*) as total_trades,
                COALESCE(SUM(fee), 0) as total_fees,
                COALESCE(SUM(total_cost), 0) as total_cost
            FROM trades
            WHERE DATE(executed_at) = ?
            """,
            (date,),
        )

        return rows[0] if rows else {"total_trades": 0, "total_fees": 0, "total_cost": 0}


class OrderRepository:
    """Repository for order records."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(self, order: Order, opportunity_id: str | None = None) -> None:
        """Save or update an order."""
        await self.db.execute(
            """
            INSERT OR REPLACE INTO orders (order_id, market_ticker, side, order_type, price, quantity, status, filled_quantity, remaining_quantity, opportunity_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                order.order_id,
                order.market_ticker,
                order.side.value,
                order.order_type.value,
                order.price,
                order.quantity,
                order.status.value,
                order.filled_quantity,
                order.remaining_quantity,
                opportunity_id,
            ),
        )

    async def get_by_id(self, order_id: str) -> Order | None:
        """Get order by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,),
        )

        if row is None:
            return None

        return Order(
            order_id=row["order_id"],
            market_ticker=row["market_ticker"],
            side=Side(row["side"]),
            order_type=OrderType(row["order_type"]),
            price=row["price"] or 0,
            quantity=row["quantity"],
            status=OrderStatus(row["status"]),
            filled_quantity=row["filled_quantity"],
            remaining_quantity=row["remaining_quantity"],
        )

    async def update_status(
        self,
        order_id: str,
        status: OrderStatus,
        filled_quantity: int | None = None,
        remaining_quantity: int | None = None,
    ) -> None:
        """Update order status."""
        updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list[Any] = [status.value]

        if filled_quantity is not None:
            updates.append("filled_quantity = ?")
            params.append(filled_quantity)

        if remaining_quantity is not None:
            updates.append("remaining_quantity = ?")
            params.append(remaining_quantity)

        params.append(order_id)

        await self.db.execute(
            f"UPDATE orders SET {', '.join(updates)} WHERE order_id = ?",
            tuple(params),
        )


class OpportunityRepository:
    """Repository for arbitrage opportunities."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(self, opportunity: ArbitrageOpportunity) -> None:
        """Save an opportunity."""
        await self.db.execute(
            """
            INSERT INTO opportunities (opportunity_id, arbitrage_type, markets, expected_profit, expected_profit_pct, confidence, legs, max_quantity, total_cost, fees, net_profit, detected_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                opportunity.opportunity_id,
                opportunity.arbitrage_type.value,
                json.dumps(opportunity.markets),
                opportunity.expected_profit,
                opportunity.expected_profit_pct,
                opportunity.confidence,
                json.dumps(opportunity.legs),
                opportunity.max_quantity,
                opportunity.total_cost,
                opportunity.fees,
                opportunity.net_profit,
                opportunity.detected_at.isoformat(),
                opportunity.expires_at.isoformat() if opportunity.expires_at else None,
            ),
        )

    async def update_status(
        self,
        opportunity_id: str,
        status: str,
        execution_result: dict[str, Any] | None = None,
    ) -> None:
        """Update opportunity status."""
        executed_at = datetime.utcnow().isoformat() if status in ("executed", "failed") else None

        await self.db.execute(
            """
            UPDATE opportunities
            SET status = ?, executed_at = ?, execution_result = ?
            WHERE opportunity_id = ?
            """,
            (
                status,
                executed_at,
                json.dumps(execution_result) if execution_result else None,
                opportunity_id,
            ),
        )

    async def get_by_id(self, opportunity_id: str) -> ArbitrageOpportunity | None:
        """Get opportunity by ID."""
        row = await self.db.fetch_one(
            "SELECT * FROM opportunities WHERE opportunity_id = ?",
            (opportunity_id,),
        )

        if row is None:
            return None

        return ArbitrageOpportunity(
            opportunity_id=row["opportunity_id"],
            arbitrage_type=ArbitrageType(row["arbitrage_type"]),
            markets=json.loads(row["markets"]),
            expected_profit=row["expected_profit"],
            expected_profit_pct=row["expected_profit_pct"],
            confidence=row["confidence"],
            legs=json.loads(row["legs"]),
            max_quantity=row["max_quantity"],
            total_cost=row["total_cost"],
            fees=row["fees"],
            net_profit=row["net_profit"],
            detected_at=datetime.fromisoformat(row["detected_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
        )

    async def get_recent(
        self,
        limit: int = 100,
        status: str | None = None,
        arbitrage_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent opportunities."""
        query = "SELECT * FROM opportunities WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if arbitrage_type:
            query += " AND arbitrage_type = ?"
            params.append(arbitrage_type)

        query += " ORDER BY detected_at DESC LIMIT ?"
        params.append(limit)

        rows = await self.db.fetch_all(query, tuple(params))

        for row in rows:
            row["markets"] = json.loads(row["markets"])
            row["legs"] = json.loads(row["legs"])
            if row["execution_result"]:
                row["execution_result"] = json.loads(row["execution_result"])

        return rows


class PortfolioSnapshotRepository:
    """Repository for portfolio snapshots."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(self, snapshot: dict[str, Any]) -> None:
        """Save a portfolio snapshot."""
        await self.db.execute(
            """
            INSERT INTO portfolio_snapshots (balance, portfolio_value, positions_value, total_value, unrealized_pnl, realized_pnl, locked_principal, tradeable_balance, peak_value, drawdown, profit_locked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["balance"],
                snapshot["portfolio_value"],
                snapshot["positions_value"],
                snapshot["total_value"],
                snapshot["unrealized_pnl"],
                snapshot["realized_pnl"],
                snapshot["locked_principal"],
                snapshot["tradeable_balance"],
                snapshot["peak_value"],
                snapshot["drawdown"],
                snapshot["profit_locked"],
            ),
        )

    async def get_latest(self) -> dict[str, Any] | None:
        """Get the latest portfolio snapshot."""
        return await self.db.fetch_one(
            "SELECT * FROM portfolio_snapshots ORDER BY snapshot_at DESC LIMIT 1"
        )

    async def get_history(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get portfolio snapshot history."""
        query = "SELECT * FROM portfolio_snapshots WHERE 1=1"
        params: list[Any] = []

        if since:
            query += " AND snapshot_at >= ?"
            params.append(since.isoformat())

        query += " ORDER BY snapshot_at DESC LIMIT ?"
        params.append(limit)

        return await self.db.fetch_all(query, tuple(params))

    async def get_peak_value(self) -> float:
        """Get historical peak portfolio value."""
        row = await self.db.fetch_one(
            "SELECT MAX(total_value) as peak FROM portfolio_snapshots"
        )
        return row["peak"] if row and row["peak"] else 0.0


class CircuitBreakerRepository:
    """Repository for circuit breaker events."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def log_trigger(
        self,
        breaker_type: str,
        trigger_reason: str,
        trigger_value: float | None = None,
        threshold_value: float | None = None,
        cooldown_until: datetime | None = None,
    ) -> None:
        """Log a circuit breaker trigger event."""
        await self.db.execute(
            """
            INSERT INTO circuit_breaker_events (breaker_type, trigger_reason, trigger_value, threshold_value, cooldown_until)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                breaker_type,
                trigger_reason,
                trigger_value,
                threshold_value,
                cooldown_until.isoformat() if cooldown_until else None,
            ),
        )

    async def log_reset(self, breaker_type: str) -> None:
        """Log a circuit breaker reset."""
        await self.db.execute(
            """
            UPDATE circuit_breaker_events
            SET reset_at = CURRENT_TIMESTAMP
            WHERE breaker_type = ? AND reset_at IS NULL
            """,
            (breaker_type,),
        )

    async def get_recent(
        self,
        limit: int = 50,
        breaker_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get recent circuit breaker events."""
        query = "SELECT * FROM circuit_breaker_events WHERE 1=1"
        params: list[Any] = []

        if breaker_type:
            query += " AND breaker_type = ?"
            params.append(breaker_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        return await self.db.fetch_all(query, tuple(params))

    async def get_active_triggers(self) -> list[dict[str, Any]]:
        """Get currently active (not reset) circuit breaker triggers."""
        return await self.db.fetch_all(
            """
            SELECT * FROM circuit_breaker_events
            WHERE reset_at IS NULL AND (cooldown_until IS NULL OR cooldown_until > CURRENT_TIMESTAMP)
            ORDER BY created_at DESC
            """
        )


class PositionRepository:
    """Repository for position records."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def save(self, position: Position) -> None:
        """Save or update a position."""
        await self.db.execute(
            """
            INSERT OR REPLACE INTO positions (market_ticker, side, quantity, average_price, market_exposure, realized_pnl, unrealized_pnl, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                position.market_ticker,
                position.side.value,
                position.quantity,
                position.average_price,
                position.market_exposure,
                position.realized_pnl,
                position.unrealized_pnl,
            ),
        )

    async def get_all(self) -> list[Position]:
        """Get all positions."""
        rows = await self.db.fetch_all("SELECT * FROM positions WHERE quantity > 0")

        return [
            Position(
                market_ticker=row["market_ticker"],
                side=Side(row["side"]),
                quantity=row["quantity"],
                average_price=row["average_price"],
                market_exposure=row["market_exposure"],
                realized_pnl=row["realized_pnl"],
                unrealized_pnl=row["unrealized_pnl"],
            )
            for row in rows
        ]

    async def delete(self, market_ticker: str) -> None:
        """Delete a position."""
        await self.db.execute(
            "DELETE FROM positions WHERE market_ticker = ?",
            (market_ticker,),
        )

    async def clear_all(self) -> None:
        """Clear all positions (for sync)."""
        await self.db.execute("DELETE FROM positions")


class DailyStatsRepository:
    """Repository for daily statistics."""

    def __init__(self, db: Database) -> None:
        self.db = db

    async def get_or_create(self, date: str, starting_balance: float) -> dict[str, Any]:
        """Get or create daily stats for a date."""
        row = await self.db.fetch_one(
            "SELECT * FROM daily_stats WHERE date = ?",
            (date,),
        )

        if row:
            return row

        await self.db.execute(
            """
            INSERT INTO daily_stats (date, starting_balance, ending_balance, tradeable_balance)
            VALUES (?, ?, ?, ?)
            """,
            (date, starting_balance, starting_balance, starting_balance),
        )

        return await self.db.fetch_one(
            "SELECT * FROM daily_stats WHERE date = ?",
            (date,),
        )

    async def update(self, date: str, updates: dict[str, Any]) -> None:
        """Update daily stats."""
        set_clauses = []
        params = []

        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(date)

        await self.db.execute(
            f"UPDATE daily_stats SET {', '.join(set_clauses)} WHERE date = ?",
            tuple(params),
        )

    async def increment(self, date: str, field: str, amount: int = 1) -> None:
        """Increment a counter field."""
        await self.db.execute(
            f"UPDATE daily_stats SET {field} = {field} + ?, updated_at = CURRENT_TIMESTAMP WHERE date = ?",
            (amount, date),
        )
