# Models API Reference

Complete API reference for all Pydantic models used in the Vector Scalping Strategy.

## 📊 Data Models

### OHLCVData

Represents a single candlestick with OHLCV data.

```python
class OHLCVData(BaseModel):
    timestamp: int = Field(..., description="Unix timestamp")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: float = Field(..., ge=0, description="Trading volume")
```

**Properties:**
- `datetime: datetime` - Converts timestamp to Python datetime object
- `true_range: float` - Calculates true range (high - low)

**Validation:**
- All prices must be positive (`gt=0`)
- Volume must be non-negative (`ge=0`)
- Automatic type conversion for numeric fields

**Example:**
```python
candle = OHLCVData(
    timestamp=1640995200,
    open=1.0850,
    high=1.0860,
    low=1.0845,
    close=1.0855,
    volume=1500.0
)

print(candle.datetime)      # 2022-01-01 00:00:00
print(candle.true_range)    # 0.0015
```

### PriceVector

Contains price vector calculation results.

```python
class PriceVector(BaseModel):
    displacement: float = Field(..., description="Price displacement over N candles")
    magnitude: float = Field(..., ge=0, description="Absolute price movement")
    direction: float = Field(..., ge=-1, le=1, description="Normalized direction (-1 to 1)")
    period: int = Field(..., gt=0, description="Number of candles used")
    start_price: float = Field(..., gt=0, description="Starting price")
    end_price: float = Field(..., gt=0, description="Ending price")
    price_range: float = Field(..., gt=0, description="High-low range over period")
```

**Validation:**
- `magnitude` must be non-negative
- `direction` must be between -1 and 1
- `period` must be positive
- All prices must be positive

**Example:**
```python
vector = PriceVector(
    displacement=0.0050,
    magnitude=0.0050,
    direction=0.5,
    period=5,
    start_price=1.0850,
    end_price=1.0900,
    price_range=0.0100
)
```

### MomentumVector

Contains momentum vector calculation results.

```python
class MomentumVector(BaseModel):
    price_momentum: float = Field(..., description="Rate of price change with bias")
    volatility: float = Field(..., ge=0, description="Average true range")
    magnitude: float = Field(..., ge=0, description="Combined momentum magnitude")
    direction: float = Field(..., ge=-1, le=1, description="Momentum direction (-1 to 1)")
    period: int = Field(..., gt=0, description="Number of candles used")
```

**Example:**
```python
momentum = MomentumVector(
    price_momentum=0.00046,
    volatility=0.00124,
    magnitude=0.00132,
    direction=0.348,
    period=5
)
```

### CombinedVector

Results from multi-timeframe vector combination.

```python
class CombinedVector(BaseModel):
    tf5_magnitude: float = Field(..., ge=0, description="5-minute vector magnitude")
    tf15_magnitude: float = Field(..., ge=0, description="15-minute vector magnitude")
    tf5_direction: float = Field(..., ge=-1, le=1, description="5-minute direction")
    tf15_direction: float = Field(..., ge=-1, le=1, description="15-minute direction")
    combined_magnitude: float = Field(..., ge=0, description="Weighted combined magnitude")
    combined_direction: float = Field(..., ge=-1, le=1, description="Weighted combined direction")
    signal_strength: float = Field(..., ge=0, le=100, description="Signal strength percentile")
```

**Example:**
```python
combined = CombinedVector(
    tf5_magnitude=0.00132,
    tf15_magnitude=0.00098,
    tf5_direction=0.348,
    tf15_direction=0.256,
    combined_magnitude=0.00122,
    combined_direction=0.312,
    signal_strength=75.5
)
```

### DivergenceSignal

Price vs momentum divergence detection results.

```python
class DivergenceSignal(BaseModel):
    price_trend: float = Field(..., description="Price trend over comparison period")
    momentum_trend: float = Field(..., description="Momentum trend over comparison period")
    is_bullish_divergence: bool = Field(..., description="Price down, momentum up")
    is_bearish_divergence: bool = Field(..., description="Price up, momentum down")
    divergence_strength: float = Field(..., ge=0, le=1, description="Strength of divergence")
```

**Example:**
```python
divergence = DivergenceSignal(
    price_trend=-0.0015,
    momentum_trend=0.0025,
    is_bullish_divergence=True,
    is_bearish_divergence=False,
    divergence_strength=0.65
)
```

### TradingSignal

Complete trading signal with entry/exit information.

```python
class TradingSignal(BaseModel):
    signal_type: SignalType = Field(..., description="Type of trading signal")
    entry_price: Optional[float] = Field(None, gt=0, description="Entry price for trade")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit level")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss level")
    confidence: float = Field(..., ge=0, le=1, description="Signal confidence (0-1)")
    timestamp: int = Field(..., description="Signal generation timestamp")
    reason: str = Field(..., description="Reason for signal generation")
    vector_data: Optional[CombinedVector] = Field(None, description="Associated vector data")
    divergence_data: Optional[DivergenceSignal] = Field(None, description="Divergence data")
```

**Properties:**
- `datetime: datetime` - Converts timestamp to Python datetime object

**Example:**
```python
signal = TradingSignal(
    signal_type=SignalType.LONG,
    entry_price=1.0850,
    take_profit=1.0870,
    stop_loss=1.0820,
    confidence=0.75,
    timestamp=int(time.time()),
    reason="Strong bullish momentum",
    vector_data=combined_vector,
    divergence_data=None
)

print(signal.datetime)  # Converted timestamp
```

## ⚙️ Configuration Models

### RiskManagement

Risk management parameters for different currency pairs.

```python
class RiskManagement(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    pip_size: float = Field(..., description="Pip size for the symbol")
    take_profit_pips: int = Field(20, description="Take profit in pips")
    stop_loss_pips: int = Field(30, description="Stop loss in pips")
    is_decimal_4: bool = Field(True, description="True for 4-decimal pairs, False for 2-decimal")
```

**Methods:**

#### `calculate_take_profit(entry_price, signal_type)`

Calculate take profit level.

**Parameters:**
- `entry_price` (float): Entry price for the trade
- `signal_type` (SignalType): LONG or SHORT signal type

**Returns:**
- `float`: Take profit price level

**Raises:**
- `ValueError`: If invalid signal type provided

**Example:**
```python
risk_mgmt = RiskManagement(symbol="EURUSD", pip_size=0.0001)
tp = risk_mgmt.calculate_take_profit(1.0850, SignalType.LONG)  # 1.0870
```

#### `calculate_stop_loss(entry_price, signal_type)`

Calculate stop loss level.

**Parameters:**
- `entry_price` (float): Entry price for the trade
- `signal_type` (SignalType): LONG or SHORT signal type

**Returns:**
- `float`: Stop loss price level

**Example:**
```python
sl = risk_mgmt.calculate_stop_loss(1.0850, SignalType.LONG)  # 1.0820
```

### StrategyConfig

Complete strategy configuration.

```python
class StrategyConfig(BaseModel):
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    exchange: str = Field("FX_IDC", description="Exchange identifier")
    vector_period: int = Field(5, description="Number of candles for vector calculation")
    percentile_lookback: int = Field(100, description="Lookback period for percentile ranking")
    signal_threshold: float = Field(60.0, ge=0, le=100, description="Minimum percentile for signals")
    direction_threshold: float = Field(0.3, ge=0, le=1, description="Minimum direction for signals")
    tf5_weight: float = Field(0.7, ge=0, le=1, description="Weight for 5-minute timeframe")
    tf15_weight: float = Field(0.3, ge=0, le=1, description="Weight for 15-minute timeframe")
    tf5_direction_weight: float = Field(0.6, ge=0, le=1, description="5-min direction weight")
    tf15_direction_weight: float = Field(0.4, ge=0, le=1, description="15-min direction weight")
    risk_management: RiskManagement = Field(..., description="Risk management settings")
```

**Validation:**
- Weights should sum to 1.0 (simplified validation in current implementation)
- All threshold values have appropriate ranges
- Symbol and exchange are required strings

**Example:**
```python
config = StrategyConfig(
    symbol="EURUSD",
    exchange="FX_IDC",
    vector_period=5,
    signal_threshold=65.0,
    direction_threshold=0.25,
    risk_management=RiskManagement(
        symbol="EURUSD",
        pip_size=0.0001,
        is_decimal_4=True
    )
)
```

## 🔢 Enums

### SignalType

Trading signal types.

```python
class SignalType(str, Enum):
    LONG = "LONG"          # Buy signal
    SHORT = "SHORT"        # Sell signal
    EXIT = "EXIT"          # Exit signal
    NO_SIGNAL = "NO_SIGNAL" # No trading signal
```

**Usage:**
```python
if signal.signal_type == SignalType.LONG:
    print("Buy signal generated")
elif signal.signal_type == SignalType.SHORT:
    print("Sell signal generated")
elif signal.signal_type == SignalType.NO_SIGNAL:
    print("No trading opportunity")
```

### TimeFrame

Supported chart timeframes.

```python
class TimeFrame(str, Enum):
    MIN_5 = "5"     # 5-minute charts
    MIN_15 = "15"   # 15-minute charts
    MIN_30 = "30"   # 30-minute charts
    HOUR_1 = "60"   # 1-hour charts
```

**Usage:**
```python
data_5m = await service.fetch_historical_data(TimeFrame.MIN_5, 100)
data_15m = await service.fetch_historical_data(TimeFrame.MIN_15, 50)
```

## 🔧 Model Utilities

### JSON Serialization

All models support JSON serialization for data persistence and API communication.

```python
# Serialize to JSON
signal_json = signal.model_dump_json()

# Deserialize from JSON
parsed_signal = TradingSignal.model_validate_json(signal_json)

# Convert to dictionary
signal_dict = signal.model_dump()

# Create from dictionary
new_signal = TradingSignal.model_validate(signal_dict)
```

### Model Copying

```python
# Create a copy with modifications
modified_config = config.model_copy(update={
    'signal_threshold': 70.0,
    'direction_threshold': 0.4
})

# Deep copy
deep_copy = config.model_copy(deep=True)
```

### Schema Generation

```python
# Generate JSON schema
schema = TradingSignal.model_json_schema()
print(json.dumps(schema, indent=2))

# Get field information
fields = TradingSignal.model_fields
for field_name, field_info in fields.items():
    print(f"{field_name}: {field_info.annotation}")
```

## ✅ Validation Examples

### Custom Validation

```python
from pydantic import field_validator

class CustomStrategyConfig(StrategyConfig):
    """Strategy config with custom validation."""
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol_format(cls, v: str) -> str:
        """Validate symbol is exactly 6 uppercase letters."""
        if len(v) != 6 or not v.isupper() or not v.isalpha():
            raise ValueError("Symbol must be 6 uppercase letters")
        return v
    
    @field_validator('vector_period')
    @classmethod
    def validate_vector_period(cls, v: int) -> int:
        """Validate vector period is reasonable."""
        if v < 3 or v > 20:
            raise ValueError("Vector period must be between 3 and 20")
        return v
```

### Validation Error Handling

```python
from pydantic import ValidationError

def safe_create_signal(data: dict) -> Optional[TradingSignal]:
    """Safely create trading signal with error handling."""
    try:
        return TradingSignal(**data)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage
signal_data = {
    'signal_type': 'LONG',
    'confidence': 0.85,
    'timestamp': int(time.time()),
    'reason': 'Strong momentum'
}

signal = safe_create_signal(signal_data)
if signal:
    print("Signal created successfully")
else:
    print("Failed to create signal")
```

## 🧪 Testing Models

### Model Testing Patterns

```python
import pytest
from vector_scalping.models import OHLCVData, PriceVector

def test_ohlcv_creation():
    """Test OHLCV data creation and validation."""
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
    """Test price vector field validation."""
    # Valid vector
    vector = PriceVector(
        displacement=0.0050,
        magnitude=0.0050,
        direction=0.5,
        period=5,
        start_price=1.0850,
        end_price=1.0900,
        price_range=0.0100
    )
    
    assert -1 <= vector.direction <= 1
    assert vector.magnitude >= 0
    
    # Invalid direction
    with pytest.raises(ValidationError):
        PriceVector(
            displacement=0.0050,
            magnitude=0.0050,
            direction=1.5,  # Invalid: > 1
            period=5,
            start_price=1.0850,
            end_price=1.0900,
            price_range=0.0100
        )
```

### Fixture Creation

```python
@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    return [
        OHLCVData(
            timestamp=1640995200 + i * 300,
            open=1.0850 + i * 0.0001,
            high=1.0860 + i * 0.0001,
            low=1.0845 + i * 0.0001,
            close=1.0855 + i * 0.0001,
            volume=1500 + i * 10
        )
        for i in range(10)
    ]

@pytest.fixture
def strategy_config():
    """Create test strategy configuration."""
    return StrategyConfig(
        symbol="EURUSD",
        risk_management=RiskManagement(
            symbol="EURUSD",
            pip_size=0.0001,
            is_decimal_4=True
        )
    )
```

## 🔗 Related Documentation

- **[Core API](core.md)** - Main classes and functions
- **[Types API](types.md)** - Type definitions and constants
- **[Module Guides](../modules/)** - Detailed usage examples

---

**Next**: [Types API Reference](types.md) →