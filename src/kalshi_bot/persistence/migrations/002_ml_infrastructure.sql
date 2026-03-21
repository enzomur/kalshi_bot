-- ML Infrastructure Schema for Self-Learning Trading Bot
-- This migration adds tables for data collection, model training, and performance tracking

-- Market snapshots: Price history polled every 5 minutes
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    event_ticker TEXT NOT NULL,

    -- Price data
    yes_bid INTEGER,
    yes_ask INTEGER,
    no_bid INTEGER,
    no_ask INTEGER,
    last_price INTEGER,

    -- Liquidity data
    volume INTEGER NOT NULL DEFAULT 0,
    open_interest INTEGER NOT NULL DEFAULT 0,

    -- Computed spread
    spread INTEGER,

    -- Market metadata
    status TEXT NOT NULL DEFAULT 'open',
    close_time TIMESTAMP,
    expiration_time TIMESTAMP,

    -- Snapshot timestamp
    snapshot_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_market_snapshots_ticker ON market_snapshots(ticker);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_event_ticker ON market_snapshots(event_ticker);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_snapshot_at ON market_snapshots(snapshot_at);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_ticker_snapshot ON market_snapshots(ticker, snapshot_at);

-- Market settlements: Outcomes for resolved markets
CREATE TABLE IF NOT EXISTS market_settlements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    event_ticker TEXT NOT NULL,

    -- Outcome: 'yes' or 'no'
    outcome TEXT NOT NULL CHECK (outcome IN ('yes', 'no')),

    -- Final price that confirmed settlement (99-100 for YES, 0-1 for NO)
    final_price INTEGER NOT NULL,

    -- Timestamps
    first_detected_at TIMESTAMP NOT NULL,
    confirmed_at TIMESTAMP,

    -- Number of snapshots we had for this market before settlement
    snapshot_count INTEGER NOT NULL DEFAULT 0,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_market_settlements_ticker ON market_settlements(ticker);
CREATE INDEX IF NOT EXISTS idx_market_settlements_event_ticker ON market_settlements(event_ticker);
CREATE INDEX IF NOT EXISTS idx_market_settlements_outcome ON market_settlements(outcome);
CREATE INDEX IF NOT EXISTS idx_market_settlements_confirmed_at ON market_settlements(confirmed_at);

-- ML models: Metadata for trained models
CREATE TABLE IF NOT EXISTS ml_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,

    -- Model type: 'logistic', 'gradient_boost', 'ensemble'
    model_type TEXT NOT NULL,

    -- Model file path (relative to models directory)
    model_path TEXT NOT NULL,

    -- Training metadata
    training_samples INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,

    -- Performance metrics (JSON)
    metrics TEXT NOT NULL,  -- {"accuracy": 0.58, "auc": 0.62, "log_loss": 0.68, ...}

    -- Cross-validation results (JSON)
    cv_results TEXT,  -- {"fold_scores": [0.55, 0.58, ...], "mean": 0.57, "std": 0.02}

    -- Feature importance (JSON)
    feature_importance TEXT,  -- {"feature_name": importance_value, ...}

    -- Model status: 'training', 'active', 'retired', 'failed'
    status TEXT NOT NULL DEFAULT 'training',

    -- Active model flag (only one can be active per model_type)
    is_active BOOLEAN NOT NULL DEFAULT FALSE,

    -- Timestamps
    trained_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    activated_at TIMESTAMP,
    retired_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_models_model_id ON ml_models(model_id);
CREATE INDEX IF NOT EXISTS idx_ml_models_model_type ON ml_models(model_type);
CREATE INDEX IF NOT EXISTS idx_ml_models_status ON ml_models(status);
CREATE INDEX IF NOT EXISTS idx_ml_models_is_active ON ml_models(is_active);

-- ML predictions: Log of predictions for accuracy tracking
CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,

    -- Model that made the prediction
    model_id TEXT NOT NULL,

    -- Market being predicted
    ticker TEXT NOT NULL,
    event_ticker TEXT NOT NULL,

    -- Prediction details
    predicted_prob_yes REAL NOT NULL,  -- Model's P(YES)
    market_price INTEGER NOT NULL,     -- Market price at prediction time (cents)
    edge REAL NOT NULL,                -- predicted_prob - market_price/100

    -- Confidence and features used (JSON)
    confidence REAL NOT NULL,
    features_used TEXT,  -- JSON snapshot of features at prediction time

    -- Trading decision
    trade_side TEXT,  -- 'yes', 'no', or NULL if no trade
    trade_quantity INTEGER,

    -- Outcome (filled in when market settles)
    actual_outcome TEXT,  -- 'yes' or 'no'
    was_correct BOOLEAN,

    -- Timestamps
    predicted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    settled_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_id ON ml_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_ticker ON ml_predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_predicted_at ON ml_predictions(predicted_at);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_was_correct ON ml_predictions(was_correct);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_actual_outcome ON ml_predictions(actual_outcome);

-- Strategy performance: Rolling performance metrics
CREATE TABLE IF NOT EXISTS strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name TEXT NOT NULL,  -- 'ml_edge', 'arbitrage', etc.

    -- Performance window
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,

    -- Prediction accuracy (ML strategies)
    total_predictions INTEGER NOT NULL DEFAULT 0,
    correct_predictions INTEGER NOT NULL DEFAULT 0,
    accuracy REAL,

    -- Trading performance
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    win_rate REAL,

    -- P&L metrics
    total_pnl REAL NOT NULL DEFAULT 0,
    total_fees REAL NOT NULL DEFAULT 0,
    net_pnl REAL NOT NULL DEFAULT 0,

    -- Risk metrics
    max_drawdown REAL NOT NULL DEFAULT 0,
    sharpe_ratio REAL,

    -- Position sizing multiplier (from self-correction)
    kelly_multiplier REAL NOT NULL DEFAULT 1.0,

    -- Strategy status
    is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    disabled_reason TEXT,
    disabled_at TIMESTAMP,

    -- Consecutive metrics
    consecutive_wins INTEGER NOT NULL DEFAULT 0,
    consecutive_losses INTEGER NOT NULL DEFAULT 0,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_strategy_performance_strategy_name ON strategy_performance(strategy_name);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_window_end ON strategy_performance(window_end);
CREATE INDEX IF NOT EXISTS idx_strategy_performance_is_enabled ON strategy_performance(is_enabled);

-- Training queue: Markets queued for training data preparation
CREATE TABLE IF NOT EXISTS training_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE NOT NULL,
    event_ticker TEXT NOT NULL,

    -- When to include in training
    include_after TIMESTAMP NOT NULL,

    -- Status: 'pending', 'included', 'excluded'
    status TEXT NOT NULL DEFAULT 'pending',
    exclusion_reason TEXT,

    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_queue_status ON training_queue(status);
CREATE INDEX IF NOT EXISTS idx_training_queue_include_after ON training_queue(include_after);
