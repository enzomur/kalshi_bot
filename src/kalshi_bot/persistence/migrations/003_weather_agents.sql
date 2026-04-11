-- Migration 003: Weather Agents Schema
-- Adds tables for weather forecasts, market mappings, and signal testing

-- Weather forecast storage from NWS API
CREATE TABLE IF NOT EXISTS weather_forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_code TEXT NOT NULL,           -- e.g., "NYC", "CHI"
    forecast_time TIMESTAMP NOT NULL,       -- When the forecast period starts
    temperature_f REAL,                     -- Temperature in Fahrenheit
    precipitation_prob REAL,                -- 0-100 percentage
    confidence REAL DEFAULT 1.0,            -- Forecast confidence 0-1
    fetched_at TIMESTAMP NOT NULL,          -- When we fetched this forecast
    UNIQUE(location_code, forecast_time)
);

CREATE INDEX IF NOT EXISTS idx_weather_forecasts_location
    ON weather_forecasts(location_code, forecast_time DESC);

CREATE INDEX IF NOT EXISTS idx_weather_forecasts_fetched
    ON weather_forecasts(fetched_at DESC);


-- Mapping of Kalshi weather markets to NWS locations
CREATE TABLE IF NOT EXISTS weather_market_mappings (
    ticker TEXT PRIMARY KEY,
    location_code TEXT NOT NULL,
    weather_type TEXT NOT NULL,             -- 'temperature', 'rain', 'snow'
    threshold_value REAL,                   -- e.g., 90 for "above 90F"
    threshold_direction TEXT NOT NULL,      -- 'above', 'below', 'between'
    event_date DATE NOT NULL,
    agent_probability REAL,                 -- Agent's probability estimate 0-1
    agent_confidence REAL,                  -- Confidence in estimate 0-1
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_weather_mappings_location
    ON weather_market_mappings(location_code);

CREATE INDEX IF NOT EXISTS idx_weather_mappings_date
    ON weather_market_mappings(event_date);


-- Signal candidate tracking for Signal Tester Agent
CREATE TABLE IF NOT EXISTS signal_candidates (
    signal_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    feature_formula TEXT NOT NULL,          -- Python expression for feature
    category TEXT DEFAULT 'general',        -- Category of signal
    status TEXT DEFAULT 'proposed',         -- 'proposed', 'testing', 'approved', 'rejected'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signal_candidates_status
    ON signal_candidates(status);


-- Signal backtest results
CREATE TABLE IF NOT EXISTS signal_backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id TEXT NOT NULL,
    backtest_run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sample_size INTEGER,                    -- Number of trades in backtest
    win_rate REAL,                          -- Win rate 0-1
    information_coefficient REAL,           -- IC correlation
    sharpe_ratio REAL,
    max_drawdown REAL,
    p_value REAL,                           -- Statistical significance
    recommended_action TEXT,                -- 'approve', 'reject', 'needs_more_data'
    notes TEXT,
    FOREIGN KEY (signal_id) REFERENCES signal_candidates(signal_id)
);

CREATE INDEX IF NOT EXISTS idx_signal_backtest_signal
    ON signal_backtest_results(signal_id, backtest_run_at DESC);


-- Weather correlation tracking for Risk Agent
CREATE TABLE IF NOT EXISTS weather_correlations (
    location_pair TEXT PRIMARY KEY,         -- e.g., "NYC:BOS"
    correlation_coefficient REAL NOT NULL,  -- Historical correlation
    sample_size INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Weather exposure tracking
CREATE TABLE IF NOT EXISTS weather_exposure_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_weather_exposure REAL,            -- Total $ in weather markets
    exposure_by_location TEXT,              -- JSON: {"NYC": 100, "CHI": 50}
    exposure_by_type TEXT,                  -- JSON: {"temperature": 100, "rain": 50}
    portfolio_pct REAL                      -- Weather exposure as % of portfolio
);

CREATE INDEX IF NOT EXISTS idx_weather_exposure_time
    ON weather_exposure_snapshots(snapshot_at DESC)
