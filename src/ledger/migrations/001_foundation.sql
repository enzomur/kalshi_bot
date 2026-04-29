-- Foundation schema for Kalshi Trading Bot v2
-- This migration creates the core tables for signal tracking, risk decisions,
-- order management, and mode transitions.

-- Bot state key-value store (for migrations and config)
CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Signals emitted by strategies
CREATE TABLE IF NOT EXISTS signals (
    signal_id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('yes', 'no')),
    target_probability REAL NOT NULL CHECK (target_probability >= 0 AND target_probability <= 1),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    edge REAL NOT NULL,
    max_position INTEGER NOT NULL,
    metadata TEXT,  -- JSON
    created_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    -- Outcome tracking
    actual_outcome TEXT CHECK (actual_outcome IN ('yes', 'no', NULL)),
    settled_at TIMESTAMP,
    was_correct INTEGER CHECK (was_correct IN (0, 1, NULL)),

    -- Indexes for common queries
    CONSTRAINT valid_max_position CHECK (max_position >= 0)
);

CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy_name);
CREATE INDEX IF NOT EXISTS idx_signals_market ON signals(market_ticker);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_unsettled ON signals(settled_at) WHERE settled_at IS NULL;

-- Risk engine decisions
CREATE TABLE IF NOT EXISTS risk_decisions (
    decision_id TEXT PRIMARY KEY,
    signal_id TEXT NOT NULL,
    approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
    approved_size INTEGER NOT NULL,
    rejection_reason TEXT,
    -- Risk checks performed
    circuit_breaker_status TEXT,  -- JSON with breaker states
    position_check TEXT,  -- JSON with position limits
    kelly_calculation TEXT,  -- JSON with Kelly details
    mode_caps TEXT,  -- JSON with mode-specific limits
    -- Metadata
    trading_mode TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
);

CREATE INDEX IF NOT EXISTS idx_risk_decisions_signal ON risk_decisions(signal_id);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_approved ON risk_decisions(approved);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_created ON risk_decisions(created_at DESC);

-- Orders (both paper and live)
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    decision_id TEXT,  -- NULL for external orders
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    order_type TEXT NOT NULL CHECK (order_type IN ('limit', 'market')),
    price INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    status TEXT NOT NULL,
    filled_quantity INTEGER DEFAULT 0,
    remaining_quantity INTEGER,
    is_paper INTEGER NOT NULL DEFAULT 1 CHECK (is_paper IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (decision_id) REFERENCES risk_decisions(decision_id),
    CONSTRAINT valid_quantity CHECK (quantity > 0),
    CONSTRAINT valid_price CHECK (price >= 0 AND price <= 100)
);

CREATE INDEX IF NOT EXISTS idx_orders_market ON orders(market_ticker);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at DESC);

-- Fills (execution records)
CREATE TABLE IF NOT EXISTS fills (
    fill_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    price INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    fee REAL DEFAULT 0,
    is_paper INTEGER NOT NULL DEFAULT 1 CHECK (is_paper IN (0, 1)),
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    CONSTRAINT valid_fill_quantity CHECK (quantity > 0),
    CONSTRAINT valid_fill_price CHECK (price >= 0 AND price <= 100)
);

CREATE INDEX IF NOT EXISTS idx_fills_order ON fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_market ON fills(market_ticker);
CREATE INDEX IF NOT EXISTS idx_fills_executed ON fills(executed_at DESC);

-- Current positions
CREATE TABLE IF NOT EXISTS positions (
    position_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_ticker TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('yes', 'no')),
    quantity INTEGER NOT NULL,
    average_price REAL NOT NULL,
    market_exposure REAL NOT NULL,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    is_paper INTEGER NOT NULL DEFAULT 1 CHECK (is_paper IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(market_ticker, side, is_paper),
    CONSTRAINT valid_position_quantity CHECK (quantity >= 0)
);

CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market_ticker);
CREATE INDEX IF NOT EXISTS idx_positions_paper ON positions(is_paper);

-- Mode transitions (audit trail)
CREATE TABLE IF NOT EXISTS mode_transitions (
    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_mode TEXT NOT NULL,
    to_mode TEXT NOT NULL,
    activated_by TEXT NOT NULL,
    reason TEXT,
    signature_valid INTEGER CHECK (signature_valid IN (0, 1)),
    transitioned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_mode_transitions_time ON mode_transitions(transitioned_at DESC);

-- Daily P&L tracking for circuit breakers
CREATE TABLE IF NOT EXISTS daily_pnl (
    date TEXT PRIMARY KEY,  -- YYYY-MM-DD
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    is_paper INTEGER NOT NULL DEFAULT 1 CHECK (is_paper IN (0, 1)),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Strategy performance tracking
CREATE TABLE IF NOT EXISTS strategy_performance (
    strategy_name TEXT PRIMARY KEY,
    total_signals INTEGER DEFAULT 0,
    approved_signals INTEGER DEFAULT 0,
    executed_signals INTEGER DEFAULT 0,
    settled_signals INTEGER DEFAULT 0,
    correct_signals INTEGER DEFAULT 0,
    accuracy REAL,
    total_pnl REAL DEFAULT 0,
    is_enabled INTEGER DEFAULT 1 CHECK (is_enabled IN (0, 1)),
    consecutive_losses INTEGER DEFAULT 0,
    last_signal_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Market snapshots for backtesting
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_ticker TEXT NOT NULL,
    yes_bid INTEGER,
    yes_ask INTEGER,
    no_bid INTEGER,
    no_ask INTEGER,
    last_price INTEGER,
    volume INTEGER,
    open_interest INTEGER,
    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market ON market_snapshots(market_ticker);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(captured_at DESC);
