# Kalshi Trading Bot - Claude Code Context

## Project Overview
Automated trading bot for Kalshi prediction markets. Uses:
1. **Weather Trading** - NWS forecast-based edge detection (primary strategy)
2. **ML Trading** - Gradient boost model for general markets
3. **Arbitrage** - YES+NO < $1 opportunities

## Quick Start
```bash
# Run bot in paper trading mode (safe)
python3 src/kalshi_bot/main.py --paper

# Run bot in live mode
python3 src/kalshi_bot/main.py

# Check performance
python3 check_performance.py
```

## Project Structure
```
src/kalshi_bot/
├── bot.py              # Main bot loop - orchestrates everything
├── main.py             # Entry point, CLI args
├── api/
│   └── client.py       # Kalshi API client
├── ml/
│   ├── feature_engineer.py   # 17 base features + 5 weather features
│   ├── inference/predictor.py # Edge prediction, trading opportunities
│   ├── training/trainer.py    # Model training (logistic, gradient_boost)
│   ├── models/               # Model implementations
│   └── self_correction/      # Strategy disabler, accuracy monitoring
├── arbitrage/
│   ├── detector.py           # Finds arbitrage opportunities
│   └── strategies/           # Single, multi-outcome, cross-market
├── execution/
│   ├── executor.py           # Live order execution
│   └── paper_trading.py      # Paper trading simulator
├── agents/
│   ├── weather/
│   │   ├── agent.py          # Fetches NWS forecasts
│   │   ├── weather_trader.py # Direct NWS-based trading (NEW)
│   │   ├── probability_calc.py # NWS to probability conversion
│   │   ├── market_mapper.py  # Ticker parsing
│   │   └── nws_client.py     # NWS API client
│   ├── signal_tester/        # Signal backtesting (disabled)
│   └── risk/                 # Correlation tracking (disabled)
├── persistence/database.py   # SQLite wrapper
└── config/settings.py        # YAML config loading
```

## Weather Trading Pipeline (Primary Strategy)
The weather trading strategy uses NWS forecasts directly - no ML model needed.

### How It Works
1. `WeatherResearchAgent` fetches NWS forecasts every 15 min
2. `probability_calc.py` converts forecasts to probabilities using:
   - Normal CDF for temperature (with uncertainty by days out)
   - Precipitation probability from NWS
3. `WeatherTrader` compares NWS probability to market price
4. Trades when NWS significantly disagrees with market (>10% edge)

### Key Files
- `agents/weather/weather_trader.py` - Trading logic
- `agents/weather/probability_calc.py` - Forecast to probability
- `agents/weather/agent.py` - NWS data collection

### Parameters (in weather_trader.py)
```python
MIN_EDGE = 0.10      # 10% minimum edge to trade
MIN_CONFIDENCE = 0.60 # NWS confidence threshold
MIN_PRICE = 20       # Only trade 20-80c markets
MAX_PRICE = 80
MAX_HOURS_TO_EVENT = 72  # Prefer near-term (3 days)
KELLY_FRACTION = 0.15    # Conservative sizing
```

## ML Trading (Secondary Strategy)
Uses gradient boost model on 17 features. Less reliable than weather.

### Price Filters (predictor.py)
```python
MIN_PRICE_FOR_YES_BET = 15  # Only trade 16-84c markets
MAX_PRICE_FOR_YES_BET = 85
MIN_PRICE_FOR_NO_BET = 15
MAX_PRICE_FOR_NO_BET = 85
```

## Database
SQLite at `data/kalshi_bot.db`. Key tables:
- `ml_predictions` - ML prediction history
- `weather_forecasts` - NWS forecast history
- `weather_market_mappings` - Weather probability estimates
- `market_snapshots` - Historical price data
- `trades`, `orders` - Executed trades

## Useful Queries
```sql
-- Weather predictions
SELECT ticker, agent_probability, agent_confidence
FROM weather_market_mappings ORDER BY rowid DESC LIMIT 10;

-- ML accuracy
SELECT COUNT(*) as total, SUM(was_correct) as wins
FROM ml_predictions WHERE actual_outcome IS NOT NULL;

-- Strategy status
SELECT strategy_name, is_enabled, consecutive_losses FROM strategy_performance;
```

## Configuration
`settings.yaml` - Main config. Key sections:
```yaml
agents:
  weather_research:
    enabled: true  # Required for weather trading
    enabled_locations: ["NYC", "CHI", "LAX", "MIA"]
  signal_tester:
    enabled: false  # Not integrated
  weather_risk:
    enabled: false  # Not needed
```

## Paper Trading Mode
- Pass `--paper` flag to main.py
- Starts with $1000 simulated balance
- Access via `executor.get_status()["current_balance"]`

## Common Issues

### No Weather Opportunities
- Check weather agent is enabled in settings.yaml
- Check NWS forecasts are being fetched (look for "Weather agent: fetching forecasts")
- Weather markets must exist and be in enabled_locations

### ML Strategy Auto-Disabled
```sql
UPDATE strategy_performance SET consecutive_losses=0, is_enabled=1 WHERE strategy_name='ml_edge'
```

## Dashboard
Runs on port 8001. Access at `http://localhost:8001`.
