# Kalshi Arbitrage Trading Bot

An automated trading bot for [Kalshi](https://kalshi.com) prediction markets that detects arbitrage opportunities and uses machine learning for edge detection.

## Features

### Trading Strategies
- **Single-Market Arbitrage**: Detects when YES + NO prices sum to less than $1.00
- **Multi-Outcome Arbitrage**: Exploits mispricing across mutually exclusive outcomes
- **Cross-Market Arbitrage**: Identifies logical inconsistencies between related markets
- **ML Edge Detection**: Uses trained models to find mispriced markets

### Risk Management
- Multiple circuit breakers (drawdown, daily loss, consecutive failures)
- Position limits and concentration controls
- Automatic strategy disabling on poor performance
- Paper trading mode for safe testing

### Machine Learning
- Automated data collection (market snapshots, settlements)
- Feature engineering pipeline
- Logistic regression and gradient boosting models
- Self-correcting position sizing based on accuracy
- Historical data backfill for bootstrapping

### Backtesting
- Historical data replay engine
- Position tracking and settlement simulation
- Comprehensive metrics (Sharpe, drawdown, profit factor)
- Report generation (text, JSON, Markdown, HTML)

### Infrastructure
- Real-time dashboard with API endpoints
- SQLite persistence for trades, positions, and audit logs
- Desktop notifications for trade execution
- Performance monitoring script
- Comprehensive logging

## Installation

```bash
# Clone the repository
git clone https://github.com/enzomur/kalshi-bot.git
cd kalshi-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Add your Kalshi API credentials to `.env`:
```
KALSHI_API_KEY_ID=your_api_key
KALSHI_PRIVATE_KEY_PATH=/path/to/your/private_key.pem
KALSHI_ENVIRONMENT=demo  # or 'production'
```

3. Adjust settings in `settings.yaml` as needed.

## Usage

### Run the Bot
```bash
# Paper trading mode (default, safe)
python -m kalshi_bot.main

# With dashboard enabled
python -m kalshi_bot.main  # Dashboard at http://localhost:8000
```

### ML Commands
```bash
# Backfill historical data for ML training
python -m kalshi_bot.main --backfill-history --days 90

# Check ML system status
python -m kalshi_bot.main --ml-status

# Train ML model
python -m kalshi_bot.main --train-model
```

### Backtesting
```bash
# Run backtest on historical data
python -m kalshi_bot.main --backtest

# With custom parameters
python -m kalshi_bot.main --backtest --backtest-balance 5000 --backtest-output report.html
```

### Performance Monitoring
```bash
# Check paper trading performance
python check_performance.py
```

### Export Opportunities
```bash
# Export market opportunities for analysis
python -m kalshi_bot.main --export-opportunities -o opportunities.md
```

## Project Structure

```
src/kalshi_bot/
├── api/                 # Kalshi API client and authentication
├── arbitrage/           # Arbitrage detection strategies
├── backtesting/         # Historical replay and simulation
├── config/              # Settings and configuration
├── core/                # Core types and exceptions
├── dashboard/           # FastAPI web dashboard
├── execution/           # Order execution and paper trading
├── ml/                  # Machine learning pipeline
│   ├── models/          # ML model implementations
│   ├── training/        # Training and scheduling
│   ├── inference/       # Prediction and edge detection
│   └── self_correction/ # Performance monitoring
├── optimization/        # Position sizing (Kelly criterion)
├── persistence/         # Database and migrations
├── portfolio/           # Portfolio management
├── risk/                # Risk management and circuit breakers
└── utils/               # Logging and notifications
```

## Dashboard API

When running, the dashboard provides these endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Bot status overview |
| `GET /api/portfolio` | Portfolio and positions |
| `GET /api/opportunities` | Detected opportunities |
| `GET /api/ml/status` | ML system status |
| `POST /api/controls` | Control actions (stop, reset, etc.) |

## Safety Features

- **Paper trading enabled by default** - No real trades until explicitly configured
- **Multiple circuit breakers** - Automatic halt on excessive losses
- **Position limits** - Prevents over-concentration
- **Emergency stop** - Manual kill switch via dashboard

## Requirements

- Python 3.11+
- Kalshi API credentials
- See `pyproject.toml` for dependencies

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational purposes. Trading prediction markets involves financial risk. Use at your own risk and never trade more than you can afford to lose.
