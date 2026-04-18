-- Migration 004: Event-driven trading agents
-- Adds tables for settlement momentum, new market detection, consistency arbitrage, and economic data

-- Track known markets for new market detection
CREATE TABLE IF NOT EXISTS known_markets (
    ticker TEXT PRIMARY KEY,
    event_ticker TEXT NOT NULL,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category TEXT,
    initial_price INTEGER,
    fair_value_estimate REAL,
    historical_similar_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_known_markets_first_seen ON known_markets(first_seen_at);
CREATE INDEX IF NOT EXISTS idx_known_markets_category ON known_markets(category);

-- Economic release calendar
CREATE TABLE IF NOT EXISTS economic_releases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_name TEXT NOT NULL,
    series_id TEXT,
    release_time TIMESTAMP NOT NULL,
    actual_value REAL,
    expected_value REAL,
    prediction_direction TEXT,
    prediction_confidence REAL,
    was_correct INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(series_id, release_time)
);

CREATE INDEX IF NOT EXISTS idx_economic_releases_time ON economic_releases(release_time);
CREATE INDEX IF NOT EXISTS idx_economic_releases_series ON economic_releases(series_id);

-- Momentum signals for settlement momentum trading
CREATE TABLE IF NOT EXISTS momentum_signals (
    ticker TEXT PRIMARY KEY,
    event_ticker TEXT,
    hours_to_settlement REAL,
    current_price INTEGER,
    price_24h_ago INTEGER,
    price_6h_ago INTEGER,
    momentum REAL,
    direction TEXT,
    confidence REAL,
    volume_24h INTEGER,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_momentum_signals_expires ON momentum_signals(expires_at);
CREATE INDEX IF NOT EXISTS idx_momentum_signals_direction ON momentum_signals(direction);

-- Market relationships for consistency arbitrage
CREATE TABLE IF NOT EXISTS market_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_a TEXT NOT NULL,
    market_b TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    constraint_value REAL,
    last_violation_at TIMESTAMP,
    violation_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(market_a, market_b, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_market_relationships_type ON market_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_market_relationships_market_a ON market_relationships(market_a);

-- Track agent-generated opportunities
CREATE TABLE IF NOT EXISTS agent_opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id TEXT UNIQUE NOT NULL,
    agent_name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    side TEXT NOT NULL,
    edge REAL NOT NULL,
    confidence REAL NOT NULL,
    quantity INTEGER,
    expected_profit REAL,
    status TEXT DEFAULT 'pending',
    execution_result TEXT,
    actual_profit REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP,
    settled_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_agent_opportunities_agent ON agent_opportunities(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_opportunities_status ON agent_opportunities(status);
CREATE INDEX IF NOT EXISTS idx_agent_opportunities_ticker ON agent_opportunities(ticker);

-- Historical fair values for new market pricing
CREATE TABLE IF NOT EXISTS historical_fair_values (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    pattern TEXT NOT NULL,
    avg_initial_price REAL,
    avg_settlement_price REAL,
    sample_count INTEGER,
    yes_rate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(category, pattern)
);

CREATE INDEX IF NOT EXISTS idx_historical_fair_values_category ON historical_fair_values(category);

-- Economic indicators tracking
CREATE TABLE IF NOT EXISTS economic_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    value REAL,
    observation_date DATE,
    release_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(series_id, observation_date)
);

CREATE INDEX IF NOT EXISTS idx_economic_indicators_series ON economic_indicators(series_id);
CREATE INDEX IF NOT EXISTS idx_economic_indicators_date ON economic_indicators(observation_date);
