# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a vector-scalping trading strategy project using Python 3.13+ and the `uv` package manager. The project follows strict coding standards for production-grade Python applications with async-first architecture, type safety, and comprehensive testing requirements.

## Vector Scalping Trading Strategy

This section documents the mathematical foundation and implementation details of the vector-based scalping strategy used in this project.

### Key Vector Concepts

- **Price Vector**: Represents price movement over time
  - **Magnitude**: Absolute price displacement over N candles
  - **Direction**: Angle of price movement (upward/downward trend)
  - Used to filter signals based on directional strength

- **Momentum Vector**: Combines price action with volatility
  - **Price Momentum**: Rate of price change with directional bias
  - **Volatility Component**: Market uncertainty measure
  - **Magnitude**: Combined strength of momentum and volatility
  - **Direction**: Overall market bias (bullish/bearish)

### Mathematical Formulas

#### Price Vector Calculation

```python
# Price displacement over N candles
displacement = close[n] - close[0]

# Price vector magnitude (absolute movement)
price_magnitude = abs(displacement)

# Price vector direction (normalized to -1 to 1)
price_direction = displacement / max(high[0:n].max() - low[0:n].min(), 0.0001)
```

#### Momentum Vector Calculation

```python
# Price momentum (rate of change with bias)
price_momentum = (close[n] - close[0]) / n

# Volatility component (average true range)
volatility = sum(max(high[i] - low[i], 
                     abs(high[i] - close[i-1]), 
                     abs(low[i] - close[i-1])) for i in range(1, n+1)) / n

# Momentum magnitude
momentum_magnitude = sqrt(price_momentum**2 + volatility**2)

# Momentum direction
momentum_direction = price_momentum / momentum_magnitude if momentum_magnitude > 0 else 0
```

#### Multi-Timeframe Vector Combination

```python
# Combine 5-min and 15-min vectors
combined_magnitude = (tf5_magnitude * 0.7) + (tf15_magnitude * 0.3)
combined_direction = (tf5_direction * 0.6) + (tf15_direction * 0.4)

# Signal strength (0-100 percentile)
signal_strength = percentile_rank(combined_magnitude, lookback_period=100)
```

#### Entry Signal Conditions

**LONG Entry:**
```python
long_signal = (
    combined_direction > 0.3 and           # Bullish bias
    signal_strength > 60 and               # Above 60th percentile
    tf5_direction > 0.2 and               # 5-min confirmation
    tf15_direction > 0.1                   # 15-min trend alignment
)
```

**SHORT Entry:**
```python
short_signal = (
    combined_direction < -0.3 and          # Bearish bias
    signal_strength > 60 and               # Above 60th percentile
    tf5_direction < -0.2 and              # 5-min confirmation
    tf15_direction < -0.1                  # 15-min trend alignment
)
```

#### Take Profit and Stop Loss Calculations

**For 4-decimal pairs (e.g., EUR/USD):**
```python
# Take profit: 20 pips (0.0020)
tp_long = entry_price + 0.0020
tp_short = entry_price - 0.0020

# Stop loss: time-based (Friday 5 PM) or 30 pips
sl_long = entry_price - 0.0030
sl_short = entry_price + 0.0030
```

**For 2-decimal pairs (e.g., USD/JPY):**
```python
# Take profit: 20 pips (0.20)
tp_long = entry_price + 0.20
tp_short = entry_price - 0.20

# Stop loss: time-based (Friday 5 PM) or 30 pips
sl_long = entry_price - 0.30
sl_short = entry_price + 0.30
```

#### Vector Divergence Detection

```python
# Price vs momentum divergence
price_trend = (close[n] - close[n-5]) / close[n-5]
momentum_trend = (momentum_magnitude[n] - momentum_magnitude[n-5]) / momentum_magnitude[n-5]

# Bullish divergence: price down, momentum up
bullish_divergence = price_trend < -0.001 and momentum_trend > 0.001

# Bearish divergence: price up, momentum down
bearish_divergence = price_trend > 0.001 and momentum_trend < -0.001
```

### Practical Implementation Tips

- **Vector Period**: Use 5-candle vectors (25 minutes on 5-min chart) for quick scalping signals
- **Trend Confirmation**: Always combine with 15-minute timeframe for trend alignment
- **Signal Filtering**: Only trade when vector strength > 60th percentile to avoid weak signals
- **Risk Management**: 
  - Target: 20-pip take profit for all pairs
  - Stop loss: Time-based exit every Friday 5 PM GMT
  - Avoid trading during low vector strength periods (< 40th percentile)
- **Market Conditions**: Strategy works best during trending markets with moderate volatility

### Worked Example (EUR/USD)

Sample 5-minute EUR/USD prices:
```
Candle 0: Open=1.0850, High=1.0860, Low=1.0845, Close=1.0855
Candle 1: Open=1.0855, High=1.0865, Low=1.0850, Close=1.0862
Candle 2: Open=1.0862, High=1.0870, Low=1.0858, Close=1.0868
Candle 3: Open=1.0868, High=1.0875, Low=1.0865, Close=1.0872
Candle 4: Open=1.0872, High=1.0880, Low=1.0870, Close=1.0878
```

**Step 1: Price Vector**
```
displacement = 1.0878 - 1.0855 = 0.0023
price_magnitude = 0.0023
price_direction = 0.0023 / (1.0880 - 1.0845) = 0.0023 / 0.0035 = 0.657
```

**Step 2: Momentum Vector**
```
price_momentum = (1.0878 - 1.0855) / 5 = 0.00046
volatility = avg(0.0015, 0.0015, 0.0012, 0.0010, 0.0010) = 0.00124
momentum_magnitude = sqrt(0.00046² + 0.00124²) = 0.00132
momentum_direction = 0.00046 / 0.00132 = 0.348
```

**Step 3: Signal Assessment**
```
If signal_strength = 75th percentile and tf15_direction = 0.25:
LONG signal triggered (direction > 0.3, strength > 60, confirmations positive)
Entry: 1.0878, TP: 1.0898, SL: 1.0848
```

### Data Source Integration

This strategy uses the [tvkit Python library](https://github.com/lumduan/tvkit) for fetching real-time 5-minute forex data from TradingView. The library provides:

- Real-time OHLCV data for major forex pairs
- Historical data for backtesting
- WebSocket connections for live trading
- Built-in data validation and error handling

```python
# Example tvkit integration
from tvkit import TvDatafeed, Interval

tv = TvDatafeed()
data = tv.get_hist(symbol='EURUSD', exchange='FX_IDC', 
                   interval=Interval.in_5_minute, n_bars=1000)
```

## Development Commands

### Package Management (using uv)
- Install dependencies: `uv pip install -r requirements.txt`
- Add dependencies: `uv pip add <package>`
- Remove dependencies: `uv pip remove <package>`
- Lock dependencies: `uv pip freeze > requirements.txt`
- Run scripts: `uv pip run python <script.py>`

### Testing
- Run all tests: `uv run python -m pytest tests/ -v`
- Run specific test: `uv run python -m pytest tests/test_specific.py -v`
- Coverage requirement: minimum 90% code coverage

### Code Quality (Required before commits)
- Format code: `uv run ruff format .`
- Check linting: `uv run ruff check .`
- Type checking: `uv run mypy line_api/`

### Running the Application
- Main entry point: `uv run python main.py`

## Architecture & Standards

### Core Requirements
- **Python Version**: 3.13+
- **Async-First**: All I/O operations must be async
- **Type Safety**: Full type annotations required (mypy strict mode)
- **Testing**: TDD approach with pytest-asyncio for async patterns
- **Error Handling**: Proper exception handling with retry mechanisms

### Directory Structure
```
/tests/          # All pytest tests
/examples/       # Real-world usage examples  
/docs/           # API documentation
/debug/          # Debug scripts (gitignored)
/scripts/        # Utility scripts
```

### Development Workflow
1. Create feature branch: `git checkout -b feature/description`
2. Write tests first (TDD)
3. Implement with full type hints and docstrings
4. Run quality checks before committing:
   - `uv run python -m pytest tests/ -v`
   - `uv run mypy line_api/`
   - `uv run ruff check . && uv run ruff format .`

### Prohibited Actions
- No bare `except:` clauses
- No hardcoded credentials (use environment variables)
- No synchronous I/O for external calls
- No wildcard imports
- No disabled linting rules without justification

### Security & Environment
- All credentials via environment variables
- Use Pydantic SecretStr for sensitive data
- Secure defaults for all configuration
