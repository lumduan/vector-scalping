# Core API Reference

This document provides the complete API reference for the core components of the Vector Scalping Strategy.

## 📦 Package Structure

```python
from vector_scalping import (
    # Core Classes
    DataService,
    SignalGenerator, 
    VectorCalculations,
    
    # Models
    OHLCVData,
    TradingSignal,
    StrategyConfig,
    RiskManagement,
    
    # Enums
    SignalType,
    TimeFrame
)
```

## 🏗️ VectorCalculations

Static methods for vector mathematical calculations.

### Methods

#### `calculate_price_vector(data, period=5)`

Calculate price vector from OHLCV data.

**Parameters:**
- `data` (List[OHLCVData]): OHLCV candlestick data
- `period` (int): Number of candles for calculation (default: 5)

**Returns:**
- `PriceVector`: Price vector calculation results

**Raises:**
- `ValueError`: If insufficient data or invalid period

**Example:**
```python
from vector_scalping.calculations import VectorCalculations

vector = VectorCalculations.calculate_price_vector(ohlcv_data, period=5)
print(f"Direction: {vector.direction:.3f}")
print(f"Magnitude: {vector.magnitude:.5f}")
```

#### `calculate_momentum_vector(data, period=5)`

Calculate momentum vector combining price and volatility.

**Parameters:**
- `data` (List[OHLCVData]): OHLCV candlestick data  
- `period` (int): Number of candles for calculation (default: 5)

**Returns:**
- `MomentumVector`: Momentum vector calculation results

**Example:**
```python
momentum = VectorCalculations.calculate_momentum_vector(ohlcv_data, period=5)
print(f"Price Momentum: {momentum.price_momentum:.5f}")
print(f"Volatility: {momentum.volatility:.5f}")
```

#### `combine_vectors(tf5_price, tf5_momentum, tf15_price, tf15_momentum, **weights)`

Combine multi-timeframe vectors using weighted averages.

**Parameters:**
- `tf5_price` (PriceVector): 5-minute price vector
- `tf5_momentum` (MomentumVector): 5-minute momentum vector
- `tf15_price` (PriceVector): 15-minute price vector
- `tf15_momentum` (MomentumVector): 15-minute momentum vector
- `tf5_weight` (float): 5-minute magnitude weight (default: 0.7)
- `tf15_weight` (float): 15-minute magnitude weight (default: 0.3)
- `tf5_dir_weight` (float): 5-minute direction weight (default: 0.6)
- `tf15_dir_weight` (float): 15-minute direction weight (default: 0.4)

**Returns:**
- `CombinedVector`: Combined vector results

**Example:**
```python
combined = VectorCalculations.combine_vectors(
    tf5_price, tf5_momentum, tf15_price, tf15_momentum,
    tf5_weight=0.8, tf15_weight=0.2
)
```

#### `calculate_percentile_rank(current_value, historical_values)`

Calculate percentile rank of current value in historical data.

**Parameters:**
- `current_value` (float): Current vector magnitude
- `historical_values` (List[float]): Historical magnitude values

**Returns:**
- `float`: Percentile rank (0-100)

**Example:**
```python
rank = VectorCalculations.calculate_percentile_rank(0.0035, historical_data)
print(f"Signal strength: {rank:.1f}th percentile")
```

#### `detect_divergence(current_data, comparison_data, period=5)`

Detect price vs momentum divergence between two periods.

**Parameters:**
- `current_data` (List[OHLCVData]): Recent period data
- `comparison_data` (List[OHLCVData]): Earlier period data
- `period` (int): Number of candles to compare (default: 5)

**Returns:**
- `DivergenceSignal`: Divergence analysis results

**Example:**
```python
divergence = VectorCalculations.detect_divergence(
    recent_candles, earlier_candles, period=5
)
print(f"Bullish divergence: {divergence.is_bullish_divergence}")
```

## 🎯 SignalGenerator

Generates trading signals based on vector analysis.

### Constructor

#### `SignalGenerator(config)`

Initialize signal generator with strategy configuration.

**Parameters:**
- `config` (StrategyConfig): Strategy configuration parameters

**Example:**
```python
from vector_scalping.signals import SignalGenerator

generator = SignalGenerator(strategy_config)
```

### Methods

#### `generate_signal(tf5_data, tf15_data, current_price)`

Generate trading signal based on vector analysis.

**Parameters:**
- `tf5_data` (List[OHLCVData]): 5-minute OHLCV data
- `tf15_data` (List[OHLCVData]): 15-minute OHLCV data
- `current_price` (float): Current market price

**Returns:**
- `TradingSignal`: Complete trading signal with entry/exit information

**Example:**
```python
signal = generator.generate_signal(tf5_candles, tf15_candles, 1.0850)

if signal.signal_type == SignalType.LONG:
    print(f"Entry: {signal.entry_price}")
    print(f"Take Profit: {signal.take_profit}")
    print(f"Stop Loss: {signal.stop_loss}")
```

#### `should_exit_time_based()`

Check if it's Friday 5 PM GMT for time-based exit.

**Returns:**
- `bool`: True if it's time to exit all positions

**Example:**
```python
if generator.should_exit_time_based():
    print("Time to close all positions!")
```

#### `reset_historical_data()`

Reset historical magnitude data (useful for backtesting).

**Example:**
```python
generator.reset_historical_data()  # Start fresh for new backtest
```

### Properties

#### `historical_magnitudes`

List of historical vector magnitudes used for percentile calculation.

**Type:** `List[float]`

**Example:**
```python
print(f"Historical data points: {len(generator.historical_magnitudes)}")
```

## 🌐 DataService

Async service for fetching and processing forex data.

### Constructor

#### `DataService(config)`

Initialize data service with strategy configuration.

**Parameters:**
- `config` (StrategyConfig): Strategy configuration

**Example:**
```python
from vector_scalping.data_service import DataService

async with DataService(config) as service:
    # Use service methods
    pass
```

### Async Context Manager

The DataService must be used as an async context manager to properly manage connections.

```python
async with DataService(config) as service:
    data = await service.fetch_historical_data(TimeFrame.MIN_5, 100)
```

### Methods

#### `fetch_historical_data(timeframe, bars_count=100)`

Fetch historical OHLCV data for the configured symbol.

**Parameters:**
- `timeframe` (TimeFrame): Chart timeframe (MIN_5, MIN_15, etc.)
- `bars_count` (int): Number of bars to fetch (default: 100)

**Returns:**
- `List[OHLCVData]`: OHLCV data sorted by timestamp

**Raises:**
- `RuntimeError`: If client not initialized or data fetch fails

**Example:**
```python
data = await service.fetch_historical_data(TimeFrame.MIN_5, 200)
print(f"Fetched {len(data)} 5-minute bars")
```

#### `fetch_multi_timeframe_data(bars_count=100)`

Fetch data for both 5-minute and 15-minute timeframes concurrently.

**Parameters:**
- `bars_count` (int): Number of bars to fetch for each timeframe

**Returns:**
- `Dict[TimeFrame, List[OHLCVData]]`: Dictionary mapping timeframes to data

**Example:**
```python
data = await service.fetch_multi_timeframe_data(100)
tf5_data = data[TimeFrame.MIN_5]
tf15_data = data[TimeFrame.MIN_15]
```

#### `convert_to_polars_dataframe(data)`

Convert OHLCV data to Polars DataFrame for analysis.

**Parameters:**
- `data` (List[OHLCVData]): OHLCV data to convert

**Returns:**
- `polars.DataFrame`: Polars DataFrame with OHLCV data

**Example:**
```python
df = service.convert_to_polars_dataframe(ohlcv_data)
print(f"DataFrame shape: {df.shape}")
```

#### `add_technical_indicators(df)`

Add technical indicators to Polars DataFrame.

**Parameters:**
- `df` (polars.DataFrame): DataFrame with OHLCV data

**Returns:**
- `polars.DataFrame`: DataFrame with additional technical indicators

**Added Indicators:**
- `sma_5`, `sma_10`, `sma_20`: Simple moving averages
- `price_change`, `returns`: Price changes and returns
- `atr_5`, `atr_14`: Average True Range
- `volatility_5`, `volatility_20`: Rolling volatility

**Example:**
```python
df_with_indicators = service.add_technical_indicators(df)
print(f"Added indicators: {df_with_indicators.columns}")
```

#### `get_latest_price()`

Get the latest price for the configured symbol.

**Returns:**
- `float`: Latest close price

**Example:**
```python
latest_price = await service.get_latest_price()
print(f"Current price: {latest_price:.5f}")
```

#### `stream_real_time_data(timeframe, callback=None)`

Stream real-time data updates (for live trading).

**Parameters:**
- `timeframe` (TimeFrame): Chart timeframe for streaming
- `callback` (Optional[Callable]): Callback function for processing updates

**Returns:**
- `None`: This is an async generator that yields updates

**Example:**
```python
async def process_bar(bar):
    print(f"New bar: {bar.close}")

await service.stream_real_time_data(TimeFrame.MIN_5, process_bar)
```

#### `validate_symbol_format(symbol)`

Validate if symbol is in correct format for the exchange.

**Parameters:**
- `symbol` (str): Trading symbol to validate

**Returns:**
- `bool`: True if valid format, False otherwise

**Example:**
```python
is_valid = service.validate_symbol_format("EURUSD")  # True
is_invalid = service.validate_symbol_format("EUR/USD")  # False
```

## 🔧 Utility Functions

### Type Checking

```python
from vector_scalping.models import SignalType, TimeFrame

# Check signal types
def is_trading_signal(signal_type: SignalType) -> bool:
    return signal_type in [SignalType.LONG, SignalType.SHORT]

# Check timeframes
def is_valid_timeframe(timeframe: str) -> bool:
    try:
        TimeFrame(timeframe)
        return True
    except ValueError:
        return False
```

### Data Validation

```python
from vector_scalping.models import OHLCVData
from pydantic import ValidationError

def validate_ohlcv_data(data_dict: dict) -> bool:
    """Validate raw OHLCV data."""
    try:
        OHLCVData(**data_dict)
        return True
    except ValidationError:
        return False
```

### Error Handling Patterns

```python
import asyncio
from vector_scalping import DataService

async def safe_data_fetch(config, retries=3):
    """Safely fetch data with retries."""
    for attempt in range(retries):
        try:
            async with DataService(config) as service:
                return await service.fetch_historical_data(TimeFrame.MIN_5, 100)
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## 📊 Performance Considerations

### Memory Usage

- Use `fetch_multi_timeframe_data()` for concurrent fetching
- Process data in chunks for large datasets
- Convert to Polars DataFrame for efficient analysis
- Clear historical data periodically in long-running applications

### Network Optimization

- Fetch multiple timeframes concurrently
- Use appropriate `bars_count` to avoid over-fetching
- Implement caching for frequently accessed data
- Handle network errors with exponential backoff

### CPU Optimization

- Use Polars for data processing (faster than Pandas)
- Batch vector calculations when possible
- Avoid frequent DataFrame conversions
- Use numpy operations in custom calculations

## 🔗 Related Documentation

- **[Models API](models.md)** - Complete model reference
- **[Types API](types.md)** - Type definitions and enums
- **[Module Guides](../modules/)** - Detailed usage guides

---

**Next**: [Models API Reference](models.md) →