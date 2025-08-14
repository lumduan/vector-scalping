# Models Module Documentation

The `models.py` module provides Pydantic data models for type-safe data validation and configuration management in the vector scalping strategy.

## 📋 Overview

All data structures in the strategy use Pydantic models to ensure:
- **Type Safety**: Runtime validation of data types
- **Data Integrity**: Validation rules and constraints
- **Serialization**: JSON encoding/decoding support
- **Documentation**: Self-documenting model schemas

## 🏗️ Core Models

### OHLCVData

Represents a single candlestick with OHLCV (Open, High, Low, Close, Volume) data.

```python
from vector_scalping.models import OHLCVData

# Create OHLCV data
candle = OHLCVData(
    timestamp=1640995200,      # Unix timestamp
    open=1.0850,              # Opening price
    high=1.0860,              # High price  
    low=1.0845,               # Low price
    close=1.0855,             # Closing price
    volume=1500.0             # Trading volume
)

# Access calculated properties
print(f"DateTime: {candle.datetime}")           # Converted timestamp
print(f"True Range: {candle.true_range}")       # High - Low
```

**Properties:**
- `datetime`: Converts timestamp to Python datetime object
- `true_range`: Calculates true range (high - low)

**Validation:**
- All prices must be positive
- Volume must be non-negative
- Automatic type conversion for numeric fields

### PriceVector

Contains price vector calculation results.

```python
from vector_scalping.models import PriceVector

vector = PriceVector(
    displacement=0.0050,      # Price movement
    magnitude=0.0050,         # Absolute movement
    direction=0.5,            # Normalized direction (-1 to 1)
    period=5,                 # Number of candles
    start_price=1.0850,       # Starting price
    end_price=1.0900,         # Ending price
    price_range=0.0100        # High-low range
)
```

**Use Cases:**
- Store price vector calculation results
- Track directional movement strength
- Validate vector parameters

### MomentumVector

Contains momentum vector calculation results combining price and volatility.

```python
from vector_scalping.models import MomentumVector

momentum = MomentumVector(
    price_momentum=0.00046,   # Rate of price change
    volatility=0.00124,       # Average true range
    magnitude=0.00132,        # Combined magnitude
    direction=0.348,          # Momentum direction
    period=5                  # Calculation period
)
```

### CombinedVector

Results from multi-timeframe vector combination.

```python
from vector_scalping.models import CombinedVector

combined = CombinedVector(
    tf5_magnitude=0.00132,     # 5-minute magnitude
    tf15_magnitude=0.00098,    # 15-minute magnitude
    tf5_direction=0.348,       # 5-minute direction
    tf15_direction=0.256,      # 15-minute direction
    combined_magnitude=0.00122, # Weighted combination
    combined_direction=0.312,   # Weighted direction
    signal_strength=75.5       # Percentile rank
)
```

### TradingSignal

Complete trading signal with entry/exit information.

```python
from vector_scalping.models import TradingSignal, SignalType
import time

signal = TradingSignal(
    signal_type=SignalType.LONG,    # LONG, SHORT, or NO_SIGNAL
    entry_price=1.0850,             # Entry price
    take_profit=1.0870,             # Take profit level
    stop_loss=1.0820,               # Stop loss level
    confidence=0.75,                # Signal confidence (0-1)
    timestamp=int(time.time()),     # Signal timestamp
    reason="Strong bullish vector", # Signal reason
    vector_data=combined,           # Associated vector data
    divergence_data=None            # Optional divergence data
)

# Access datetime property
print(f"Signal time: {signal.datetime}")
```

## ⚙️ Configuration Models

### RiskManagement

Risk management parameters for different currency pairs.

```python
from vector_scalping.models import RiskManagement, SignalType

# EUR/USD (4-decimal pair)
eurusd_risk = RiskManagement(
    symbol="EURUSD",
    pip_size=0.0001,              # 4-decimal pip size
    take_profit_pips=20,          # 20-pip target
    stop_loss_pips=30,            # 30-pip stop
    is_decimal_4=True             # 4-decimal pair
)

# Calculate trade levels
entry_price = 1.0850
tp = eurusd_risk.calculate_take_profit(entry_price, SignalType.LONG)
sl = eurusd_risk.calculate_stop_loss(entry_price, SignalType.LONG)

print(f"Entry: {entry_price}")
print(f"Take Profit: {tp}")     # 1.0870
print(f"Stop Loss: {sl}")       # 1.0820

# USD/JPY (2-decimal pair)
usdjpy_risk = RiskManagement(
    symbol="USDJPY",
    pip_size=0.01,                # 2-decimal pip size
    is_decimal_4=False            # 2-decimal pair
)
```

### StrategyConfig

Complete strategy configuration.

```python
from vector_scalping.models import StrategyConfig, RiskManagement

config = StrategyConfig(
    symbol="EURUSD",              # Trading symbol
    exchange="FX_IDC",            # Exchange identifier
    vector_period=5,              # Vector calculation period
    percentile_lookback=100,      # Percentile calculation window
    signal_threshold=60.0,        # Minimum signal strength
    direction_threshold=0.3,      # Minimum directional bias
    tf5_weight=0.7,              # 5-minute weight
    tf15_weight=0.3,             # 15-minute weight
    tf5_direction_weight=0.6,    # 5-min direction weight
    tf15_direction_weight=0.4,   # 15-min direction weight
    risk_management=risk_management
)
```

## 🔧 Enums and Constants

### SignalType

Trading signal types.

```python
from vector_scalping.models import SignalType

# Available signal types
SignalType.LONG        # Buy signal
SignalType.SHORT       # Sell signal
SignalType.EXIT        # Exit signal
SignalType.NO_SIGNAL   # No trading signal
```

### TimeFrame

Supported chart timeframes.

```python
from vector_scalping.models import TimeFrame

# Available timeframes
TimeFrame.MIN_5   # "5"  - 5-minute charts
TimeFrame.MIN_15  # "15" - 15-minute charts
TimeFrame.MIN_30  # "30" - 30-minute charts
TimeFrame.HOUR_1  # "60" - 1-hour charts
```

## 📝 Validation Examples

### Data Validation

```python
from vector_scalping.models import OHLCVData
from pydantic import ValidationError

try:
    # Valid data
    valid_candle = OHLCVData(
        timestamp=1640995200,
        open=1.0850,
        high=1.0860,
        low=1.0845,
        close=1.0855,
        volume=1500.0
    )
    print("✅ Valid candle created")
    
except ValidationError as e:
    print(f"❌ Validation error: {e}")

try:
    # Invalid data (negative volume)
    invalid_candle = OHLCVData(
        timestamp=1640995200,
        open=1.0850,
        high=1.0860,
        low=1.0845,
        close=1.0855,
        volume=-100.0  # Invalid negative volume
    )
    
except ValidationError as e:
    print(f"❌ Validation caught: {e}")
```

### Configuration Validation

```python
from vector_scalping.models import StrategyConfig, RiskManagement

# Create risk management
risk_mgmt = RiskManagement(
    symbol="EURUSD",
    pip_size=0.0001,
    is_decimal_4=True
)

# Valid configuration
config = StrategyConfig(
    symbol="EURUSD",
    risk_management=risk_mgmt,
    tf5_weight=0.7,
    tf15_weight=0.3  # Weights sum to 1.0
)

print("✅ Configuration validated")
```

## 🔄 JSON Serialization

All models support JSON serialization for data persistence and API communication.

```python
from vector_scalping.models import TradingSignal, SignalType
import json

# Create a signal
signal = TradingSignal(
    signal_type=SignalType.LONG,
    entry_price=1.0850,
    confidence=0.75,
    timestamp=1640995200,
    reason="Strong bullish momentum"
)

# Serialize to JSON
json_data = signal.model_dump_json()
print(f"JSON: {json_data}")

# Deserialize from JSON
parsed_signal = TradingSignal.model_validate_json(json_data)
print(f"Parsed: {parsed_signal.signal_type}")
```

## 🧪 Testing Models

```python
import pytest
from vector_scalping.models import OHLCVData, PriceVector

def test_ohlcv_creation():
    """Test OHLCV data creation."""
    candle = OHLCVData(
        timestamp=1640995200,
        open=1.0850,
        high=1.0860,
        low=1.0845,
        close=1.0855,
        volume=1500.0
    )
    
    assert candle.open == 1.0850
    assert candle.true_range == 0.0015
    assert isinstance(candle.datetime, datetime)

def test_price_vector_validation():
    """Test price vector validation."""
    vector = PriceVector(
        displacement=0.0050,
        magnitude=0.0050,
        direction=0.5,  # Must be between -1 and 1
        period=5,
        start_price=1.0850,
        end_price=1.0900,
        price_range=0.0100
    )
    
    assert -1 <= vector.direction <= 1
    assert vector.magnitude >= 0
```

## 🎯 Best Practices

### 1. Use Type Hints

```python
from typing import List
from vector_scalping.models import OHLCVData

def process_candles(candles: List[OHLCVData]) -> float:
    """Process OHLCV candles and return average close."""
    return sum(candle.close for candle in candles) / len(candles)
```

### 2. Validate Input Data

```python
def create_safe_candle(data: dict) -> OHLCVData:
    """Safely create OHLCV candle with validation."""
    try:
        return OHLCVData(**data)
    except ValidationError as e:
        print(f"Invalid candle data: {e}")
        raise
```

### 3. Use Model Methods

```python
# Use built-in calculation methods
risk_mgmt = RiskManagement(symbol="EURUSD", pip_size=0.0001)
tp = risk_mgmt.calculate_take_profit(1.0850, SignalType.LONG)
sl = risk_mgmt.calculate_stop_loss(1.0850, SignalType.LONG)
```

## 🔗 Related Documentation

- **[Calculations Module](calculations.md)** - Using models in vector calculations
- **[Signals Module](signals.md)** - Signal generation with models
- **[Data Service](data_service.md)** - Data fetching and model conversion
- **[API Reference](../api/models.md)** - Complete model API

---

**Next**: Learn about [Vector Calculations](calculations.md) →