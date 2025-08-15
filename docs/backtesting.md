# Vector Scalping Backtesting Guide

This guide explains how to use the vector scalping backtesting framework to test and validate your trading strategies.

## 🚀 Quick Start

### Basic Usage

```bash
# Run the backtest with default settings
uv run python backtest/vector_scalping_vectorbt.py

# Or run as a module
uv run python -m backtest.vector_scalping_vectorbt
```

### Sample Output

```
🚀 Starting Vector Scalping Backtest
============================================================
📊 Generating sample EURUSD data...
📈 Data period: 2024-01-01 to 2024-01-04 (3 days)
📊 Total 5-min bars: 1,000
🔍 Calculating vector signals...

🎯 Signal Statistics:
  📈 Long signals: 20
  📉 Short signals: 17
  📊 Total signals: 37
  📊 Signal frequency: 3.70%

💰 Performance Metrics:
  💵 Initial capital: $10,000.00
  📈 Long strategy return: +1.10%
  📉 Short strategy return: -0.60%
  🎯 Combined return: +0.25%
  📉 Max drawdown (Long): 0.30%
  📉 Max drawdown (Short): 0.80%

📈 Trading Statistics:
  🎯 Win rate (Long): 87.5%
  🎯 Win rate (Short): 42.9%
  🎯 Combined win rate: 65.2%
  📊 Total trades: 15
  📊 Avg trade return (Long): $13.75
  📊 Sharpe ratio (Long): 4.65
```

## 🛠️ Configuration

### BacktestConfig Parameters

```python
from backtest.vector_scalping_vectorbt import BacktestConfig

config = BacktestConfig(
    symbol="EURUSD",                    # Trading symbol
    initial_cash=10000.0,               # Starting capital
    pip_value=1.0,                      # Value per pip
    take_profit_pips=20,                # Take profit target
    stop_loss_pips=30,                  # Stop loss distance
    pip_size=0.0001,                    # Pip size (0.0001 for 4-decimal)
    vector_period=5,                    # Vector calculation period
    percentile_threshold=60.0,          # Signal strength threshold
    direction_threshold=0.0005,         # Minimum direction threshold
    lookback_period=100                 # Percentile calculation window
)
```

### Symbol Configuration

#### 4-Decimal Pairs (EUR/USD, GBP/USD, AUD/USD, etc.)
```python
config = BacktestConfig(
    symbol="EURUSD",
    pip_size=0.0001,        # 1 pip = 0.0001
    take_profit_pips=20,    # 20 pips = 0.0020
    stop_loss_pips=30       # 30 pips = 0.0030
)
```

#### 2-Decimal Pairs (USD/JPY, EUR/JPY, GBP/JPY, etc.)
```python
config = BacktestConfig(
    symbol="USDJPY", 
    pip_size=0.01,          # 1 pip = 0.01
    take_profit_pips=20,    # 20 pips = 0.20
    stop_loss_pips=30       # 30 pips = 0.30
)
```

## 🧪 Custom Backtesting

### Using Your Own Data

```python
import asyncio
import pandas as pd
from backtest.vector_scalping_vectorbt import VectorBacktester, BacktestConfig

async def run_custom_backtest():
    # Load your OHLCV data
    df = pd.read_csv('your_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Configure backtest
    config = BacktestConfig(
        symbol="EURUSD",
        initial_cash=50000.0,
        vector_period=5,
        percentile_threshold=65.0  # Stricter signal filtering
    )
    
    # Run backtest
    backtester = VectorBacktester(config)
    results = await backtester.run_backtest(data=df)
    
    return results

# Run the backtest
results = asyncio.run(run_custom_backtest())
```

### Data Format Requirements

Your CSV data should have the following format:

```csv
datetime,open,high,low,close,volume
2024-01-01 09:00:00,1.0850,1.0860,1.0845,1.0855,1500
2024-01-01 09:05:00,1.0855,1.0865,1.0850,1.0862,1200
2024-01-01 09:10:00,1.0862,1.0870,1.0858,1.0868,1800
...
```

**Requirements:**
- **datetime**: Timestamp in any pandas-compatible format
- **open, high, low, close**: Price values (float)
- **volume**: Trading volume (float)
- **Frequency**: 5-minute bars for optimal performance
- **Timezone**: UTC preferred for consistency

## 📊 Strategy Logic

### Vector Calculations

The backtest implements the mathematical formulas from CLAUDE.md:

#### Price Vector
```python
# Individual price differences
Δ₁ = P₂ - P₁, Δ₂ = P₃ - P₂, Δ₃ = P₄ - P₃, Δ₄ = P₅ - P₄

# Vector Magnitude
magnitude = √(Δ₁² + Δ₂² + Δ₃² + Δ₄²)

# Vector Direction  
direction = (Δ₁ + Δ₂ + Δ₃ + Δ₄) ÷ 4
```

#### Momentum Vector
```python
# Price momentum components
PM₁ = C₂ - C₁, PM₂ = C₃ - C₂, PM₃ = C₄ - C₃, PM₄ = C₅ - C₄

# Volatility components
V₁ = H₁ - L₁, V₂ = H₂ - L₂, V₃ = H₃ - L₃, V₄ = H₄ - L₄, V₅ = H₅ - L₅

# Momentum magnitude
magnitude = √[(PM₁² + PM₂² + PM₃² + PM₄²) + (V₁² + V₂² + V₃² + V₄² + V₅²)]

# Momentum direction
direction = (PM₁ + PM₂ + PM₃ + PM₄) ÷ 4
```

### Signal Generation

#### LONG Entry Conditions
- Combined direction > direction_threshold (default: 0.0005)
- Vector strength > percentile_threshold (default: 60th percentile)
- 5-minute direction > 0.4 × direction_threshold
- 15-minute direction > 0.2 × direction_threshold

#### SHORT Entry Conditions  
- Combined direction < -direction_threshold
- Vector strength > percentile_threshold
- 5-minute direction < -0.4 × direction_threshold
- 15-minute direction < -0.2 × direction_threshold

### Risk Management

#### Take Profit & Stop Loss
- **Take Profit**: Fixed at 20 pips for all pairs
- **Stop Loss**: Fixed at 30 pips for all pairs
- **Time-based Exit**: All positions closed Friday 5 PM GMT

#### Position Sizing
- Each trade uses fixed pip value (configurable)
- No compounding (fixed position sizes)
- Separate tracking for long and short strategies

## 📈 Performance Metrics

### Return Calculations
- **Total Return**: (Final Value - Initial Capital) / Initial Capital
- **Combined Return**: Average of long and short strategy returns
- **Win Rate**: Percentage of profitable trades
- **Average Trade Return**: Mean profit/loss per trade

### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return metric (annualized)
- **Trade Duration**: Average time in position

### Signal Analysis
- **Signal Frequency**: Percentage of bars generating signals
- **Vector Strength Distribution**: Percentile analysis
- **Direction Alignment**: Multi-timeframe confirmation rates

## 🔧 Advanced Configuration

### Optimization Parameters

```python
# High-frequency scalping (more signals)
config = BacktestConfig(
    percentile_threshold=50.0,      # Lower threshold
    direction_threshold=0.0003,     # More sensitive
    vector_period=3,                # Shorter period
    take_profit_pips=15             # Smaller targets
)

# Conservative approach (fewer, higher-quality signals)  
config = BacktestConfig(
    percentile_threshold=75.0,      # Higher threshold
    direction_threshold=0.001,      # Less sensitive
    vector_period=7,                # Longer period
    take_profit_pips=25             # Larger targets
)
```

### Multi-Timeframe Weights

```python
config = BacktestConfig(
    # Favor 5-minute signals (more aggressive)
    tf5_weight=0.8,                 # 5-min magnitude weight  
    tf15_weight=0.2,                # 15-min magnitude weight
    tf5_dir_weight=0.7,             # 5-min direction weight
    tf15_dir_weight=0.3             # 15-min direction weight
)

config = BacktestConfig(
    # Balanced approach
    tf5_weight=0.6,
    tf15_weight=0.4,
    tf5_dir_weight=0.5,
    tf15_dir_weight=0.5
)
```

## 🛠️ Technical Implementation

### Architecture
- **Custom Backtesting Engine**: Pure pandas/numpy implementation
- **Async Support**: Fully async-compatible for real-time integration
- **Type Safety**: Complete mypy compliance with type annotations
- **Error Handling**: Graceful handling of calculation errors
- **Memory Efficient**: Streaming calculations for large datasets

### Dependencies
```bash
# Core requirements (automatically installed)
pandas>=2.0.0
numpy>=1.20.0
pydantic>=2.0.0

# Optional: vectorbt support (compatibility issues with Python 3.13)
# uv pip install "numpy<2.1" "numba<0.61" vectorbt
```

### File Structure
```
backtest/
└── vector_scalping_vectorbt.py    # Main backtesting script
    ├── BacktestConfig              # Configuration model
    ├── VectorBacktester           # Main backtester class
    ├── _simulate_trades()         # Trade simulation engine
    ├── _calculate_performance_metrics()  # Performance analysis
    └── main()                     # Entry point
```

## 🚨 Limitations & Considerations

### Known Limitations
1. **Vectorbt Compatibility**: Uses custom engine due to numpy version conflicts
2. **Slippage**: Not modeled (assumes perfect fills at TP/SL levels)  
3. **Spread**: Fixed 1-pip spread assumption
4. **Market Hours**: No trading hour restrictions implemented
5. **Holidays**: No holiday calendar filtering

### Performance Considerations
- **Memory Usage**: ~100MB for 10,000 5-minute bars
- **Computation Time**: ~2-5 seconds for 1,000 bars on modern hardware
- **Scalability**: Linear scaling with data size

### Accuracy Notes
- **Historical Bias**: Uses perfect hindsight for TP/SL exits
- **Execution**: Assumes instant execution at signal generation
- **Data Quality**: Results dependent on input data accuracy

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, ensure the module is in Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/vector-scalping"

# Or use absolute imports
uv run python -m backtest.vector_scalping_vectorbt
```

#### Memory Issues
```python
# For large datasets, process in chunks
def chunk_backtest(data, chunk_size=5000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        result = await backtester.run_backtest(chunk)
        results.append(result)
    return combine_results(results)
```

#### Performance Optimization
```python
# Reduce lookback period for faster computation
config = BacktestConfig(
    lookback_period=50,     # Default: 100
    vector_period=3         # Default: 5
)
```

## 📚 Examples

See the `examples/backtesting/` directory for complete working examples:
- Basic backtesting workflow
- Custom data integration
- Parameter optimization
- Multi-symbol backtesting
- Performance analysis and visualization

## ⚡ Next Steps

1. **Parameter Optimization**: Use grid search to find optimal parameters
2. **Walk-Forward Analysis**: Implement rolling window backtesting
3. **Risk Management**: Add position sizing and portfolio management
4. **Real-time Integration**: Connect to live data feeds
5. **Visualization**: Add performance charts and analysis plots

For more advanced usage and customization, refer to the source code and inline documentation in `backtest/vector_scalping_vectorbt.py`.