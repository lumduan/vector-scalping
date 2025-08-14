# Vector Scalping Trading Strategy

A production-grade Python implementation of a vector-based scalping trading strategy for forex markets. This strategy uses mathematical vector analysis of price and momentum data across multiple timeframes to generate high-frequency trading signals.

## 🎯 Overview

The Vector Scalping Strategy combines price displacement vectors with momentum analysis to identify short-term trading opportunities in forex markets. It operates on 5-minute and 15-minute timeframes, targeting 20-pip profits with time-based risk management.

### Key Features

- **Vector-Based Analysis**: Mathematical price and momentum vector calculations
- **Multi-Timeframe Fusion**: Combines 5-min and 15-min data with weighted algorithms
- **Async Architecture**: Production-ready async/await patterns for real-time trading
- **Type Safety**: Full type annotations with Pydantic models
- **Polars Integration**: High-performance data processing with Polars DataFrames
- **Risk Management**: Automated take-profit, stop-loss, and time-based exits
- **Comprehensive Testing**: 49+ unit tests covering all core functionality

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd vector-scalping

# Install dependencies using uv
uv pip install -r requirements.txt

# Install package in development mode
uv pip install -e .
```

### Basic Usage

```python
import asyncio
from vector_scalping import (
    DataService, SignalGenerator, StrategyConfig, 
    RiskManagement, TimeFrame, SignalType
)

async def main():
    # Configure strategy for EUR/USD
    config = StrategyConfig(
        symbol="EURUSD",
        exchange="FX_IDC",
        vector_period=5,
        signal_threshold=60.0,
        risk_management=RiskManagement(
            symbol="EURUSD",
            pip_size=0.0001,
            take_profit_pips=20,
            stop_loss_pips=30,
            is_decimal_4=True
        )
    )
    
    # Fetch market data and generate signals
    async with DataService(config) as service:
        data = await service.fetch_multi_timeframe_data(100)
        
        generator = SignalGenerator(config)
        current_price = await service.get_latest_price()
        
        signal = generator.generate_signal(
            data[TimeFrame.MIN_5], 
            data[TimeFrame.MIN_15], 
            current_price
        )
        
        if signal.signal_type != SignalType.NO_SIGNAL:
            print(f"🎯 {signal.signal_type}: Entry={signal.entry_price:.5f}")
            print(f"📈 Take Profit: {signal.take_profit:.5f}")
            print(f"📉 Stop Loss: {signal.stop_loss:.5f}")
            print(f"💪 Confidence: {signal.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Run the Demo

```bash
# Run the complete strategy demonstration
uv run python examples/strategy_demo.py
```

## 📊 Strategy Mathematics

### Price Vector Calculation
```python
# Price displacement over N candles
displacement = close[n] - close[0]
magnitude = abs(displacement)
direction = displacement / max(high_range, 0.0001)
```

### Momentum Vector Calculation
```python
# Price momentum with volatility
price_momentum = (close[n] - close[0]) / n
volatility = average_true_range(period)
magnitude = sqrt(price_momentum² + volatility²)
direction = price_momentum / magnitude
```

### Multi-Timeframe Combination
```python
# Weighted combination of timeframes
combined_magnitude = (tf5_magnitude * 0.7) + (tf15_magnitude * 0.3)
combined_direction = (tf5_direction * 0.6) + (tf15_direction * 0.4)
```

### Entry Conditions

**LONG Signal:**
- Combined direction > 0.3 (bullish bias)
- Signal strength > 60th percentile
- 5-min direction > 0.2
- 15-min direction > 0.1

**SHORT Signal:**
- Combined direction < -0.3 (bearish bias)
- Signal strength > 60th percentile
- 5-min direction < -0.2
- 15-min direction < -0.1

## 🏗️ Architecture

### Core Modules

- **`models.py`** - Pydantic data models and validation
- **`calculations.py`** - Vector calculation algorithms
- **`signals.py`** - Signal generation and risk management
- **`data_service.py`** - Async data fetching with tvkit

### Project Structure

```
vector-scalping/
├── vector_scalping/           # Core strategy package
│   ├── __init__.py           # Package exports
│   ├── models.py             # Data models and validation
│   ├── calculations.py       # Vector calculations
│   ├── signals.py            # Signal generation
│   └── data_service.py       # Data fetching service
├── tests/                    # Test suite (49+ tests)
│   ├── test_models.py        # Model validation tests
│   ├── test_calculations.py  # Vector calculation tests
│   ├── test_signals.py       # Signal generation tests
│   └── test_data_service.py  # Data service tests
├── examples/                 # Usage examples
│   └── strategy_demo.py      # Complete demo script
├── docs/                     # Documentation
│   ├── modules/              # Module-specific docs
│   └── api/                  # API reference
└── pyproject.toml           # Project configuration
```

## 🧪 Development

### Testing

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run with coverage
uv run python -m pytest tests/ --cov=vector_scalping --cov-report=html
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Type checking
uv run mypy vector_scalping/
```

### Development Commands

```bash
# Install dependencies
uv pip install -r requirements.txt

# Add new dependency
uv add <package>

# Run in development mode
uv run python <script.py>

# Run tests
uv run python -m pytest tests/ -v
```

## 📈 Performance

- **Timeframes**: 5-minute and 15-minute analysis
- **Target**: 20-pip take profit
- **Risk Management**: Time-based exits (Friday 5 PM GMT)
- **Signal Filtering**: Only trades above 60th percentile strength
- **Async Performance**: Concurrent data fetching for multiple timeframes

## 🔒 Security

- Environment-based configuration
- No hardcoded credentials
- Secure data validation with Pydantic
- Defensive programming patterns
- Input sanitization and error handling

## 🎛️ Configuration

### Strategy Parameters

- `vector_period`: Number of candles for vector calculation (default: 5)
- `signal_threshold`: Minimum percentile for signal generation (default: 60.0)
- `direction_threshold`: Minimum direction strength (default: 0.3)
- `tf5_weight` / `tf15_weight`: Timeframe magnitude weights (default: 0.7/0.3)

### Risk Management

- `take_profit_pips`: Target profit in pips (default: 20)
- `stop_loss_pips`: Maximum loss in pips (default: 30)
- `pip_size`: Pip size for symbol (0.0001 for 4-decimal, 0.01 for 2-decimal)

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Module Documentation](docs/modules/)** - Detailed module usage guides
- **[API Reference](docs/api/)** - Complete API documentation
- **[Examples](examples/)** - Working code examples

## 🤝 Contributing

1. Follow the coding standards in `CLAUDE.md`
2. Write tests for all new functionality
3. Ensure 100% type annotation coverage
4. Pass all quality checks before submitting PRs

### Quality Gates

- ✅ All tests passing (pytest)
- ✅ Type checking (mypy)
- ✅ Code formatting (ruff)
- ✅ 90%+ test coverage

## 📄 License

MIT License - see LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss. Past performance does not guarantee future results. Always conduct thorough testing and risk assessment before live trading.

## 🔗 Links

- **tvkit Library**: [GitHub Repository](https://github.com/lumduan/tvkit)
- **Polars Documentation**: [User Guide](https://pola-rs.github.io/polars/)
- **Pydantic Documentation**: [User Guide](https://docs.pydantic.dev/)

---

**Built with Python 3.13+ | Async/Await | Type Safety | Production Ready**