-- Initial schema for Kalshi Bot database
-- SQLite with WAL mode for concurrent reads

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Trades table: records all executed trades
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    order_id TEXT NOT NULL,
    opportunity_id TEXT,
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    price INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    fee REAL NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL,
    executed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_trades_market_ticker ON trades(market_ticker);
CREATE INDEX IF NOT EXISTS idx_trades_executed_at ON trades(executed_at);
CREATE INDEX IF NOT EXISTS idx_trades_opportunity_id ON trades(opportunity_id);

-- Positions table: current positions (synced from Kalshi)
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_ticker TEXT UNIQUE NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    quantity INTEGER NOT NULL,
    average_price REAL NOT NULL,
    market_exposure REAL NOT NULL,
    realized_pnl REAL NOT NULL DEFAULT 0,
    unrealized_pnl REAL NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_positions_market_ticker ON positions(market_ticker);

-- Opportunities table: detected arbitrage opportunities
CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id TEXT UNIQUE NOT NULL,
    arbitrage_type TEXT NOT NULL CHECK (arbitrage_type IN ('single_market', 'multi_outcome', 'cross_market', 'dependency_based')),
    markets TEXT NOT NULL,  -- JSON array of market tickers
    expected_profit REAL NOT NULL,
    expected_profit_pct REAL NOT NULL,
    confidence REAL NOT NULL,
    legs TEXT NOT NULL,  -- JSON array of trade legs
    max_quantity INTEGER NOT NULL,
    total_cost REAL NOT NULL,
    fees REAL NOT NULL,
    net_profit REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'detected' CHECK (status IN ('detected', 'executing', 'executed', 'failed', 'expired')),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    executed_at TIMESTAMP,
    execution_result TEXT,  -- JSON with execution details
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_opportunities_status ON opportunities(status);
CREATE INDEX IF NOT EXISTS idx_opportunities_detected_at ON opportunities(detected_at);
CREATE INDEX IF NOT EXISTS idx_opportunities_arbitrage_type ON opportunities(arbitrage_type);

-- Portfolio snapshots: periodic snapshots for tracking
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance REAL NOT NULL,
    portfolio_value REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_value REAL NOT NULL,
    unrealized_pnl REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    locked_principal REAL NOT NULL DEFAULT 0,
    tradeable_balance REAL NOT NULL,
    peak_value REAL NOT NULL,
    drawdown REAL NOT NULL,
    profit_locked BOOLEAN NOT NULL DEFAULT FALSE,
    snapshot_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_snapshot_at ON portfolio_snapshots(snapshot_at);

-- Audit log: comprehensive logging of all bot actions
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    event_data TEXT NOT NULL,  -- JSON
    severity TEXT NOT NULL DEFAULT 'info' CHECK (severity IN ('debug', 'info', 'warning', 'error', 'critical')),
    component TEXT,
    correlation_id TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_severity ON audit_log(severity);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_correlation_id ON audit_log(correlation_id);

-- Circuit breaker events: track circuit breaker triggers
CREATE TABLE IF NOT EXISTS circuit_breaker_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    breaker_type TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    trigger_value REAL,
    threshold_value REAL,
    cooldown_until TIMESTAMP,
    reset_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_circuit_breaker_events_breaker_type ON circuit_breaker_events(breaker_type);
CREATE INDEX IF NOT EXISTS idx_circuit_breaker_events_created_at ON circuit_breaker_events(created_at);

-- Orders table: track order states
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE NOT NULL,
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    order_type TEXT NOT NULL CHECK (order_type IN ('limit', 'market')),
    price INTEGER,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    remaining_quantity INTEGER NOT NULL,
    opportunity_id TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_market_ticker ON orders(market_ticker);
CREATE INDEX IF NOT EXISTS idx_orders_opportunity_id ON orders(opportunity_id);

-- Daily statistics table
CREATE TABLE IF NOT EXISTS daily_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE NOT NULL,
    starting_balance REAL NOT NULL,
    ending_balance REAL NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    successful_trades INTEGER NOT NULL DEFAULT 0,
    failed_trades INTEGER NOT NULL DEFAULT 0,
    total_pnl REAL NOT NULL DEFAULT 0,
    total_fees REAL NOT NULL DEFAULT 0,
    opportunities_detected INTEGER NOT NULL DEFAULT 0,
    opportunities_executed INTEGER NOT NULL DEFAULT 0,
    circuit_breaker_triggers INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_daily_stats_date ON daily_stats(date);

-- Settings/state persistence
CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
