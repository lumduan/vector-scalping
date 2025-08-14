# Types API Reference

Complete reference for all type definitions, enums, and constants used in the Vector Scalping Strategy.

## 🔢 Enumerations

### SignalType

Trading signal types for entry and exit decisions.

```python
from vector_scalping.models import SignalType

class SignalType(str, Enum):
    LONG = "LONG"          # Buy signal - enter long position
    SHORT = "SHORT"        # Sell signal - enter short position  
    EXIT = "EXIT"          # Exit signal - close current position
    NO_SIGNAL = "NO_SIGNAL" # No trading opportunity
```

**Usage Examples:**
```python
# Check signal type
if signal.signal_type == SignalType.LONG:
    print("📈 Long signal - Buy opportunity")
elif signal.signal_type == SignalType.SHORT:
    print("📉 Short signal - Sell opportunity")
elif signal.signal_type == SignalType.EXIT:
    print("🚪 Exit signal - Close position")
else:
    print("⏸️ No signal - Wait for opportunity")

# String representation
signal_str = str(SignalType.LONG)  # "LONG"

# Iteration
for signal_type in SignalType:
    print(f"Available signal: {signal_type}")
```

**Signal Meanings:**
- **LONG**: Bullish market conditions detected, consider buying
- **SHORT**: Bearish market conditions detected, consider selling
- **EXIT**: Time-based or risk-based exit conditions met
- **NO_SIGNAL**: Insufficient signal strength or conflicting indicators

### TimeFrame

Supported chart timeframes for data fetching and analysis.

```python
from vector_scalping.models import TimeFrame

class TimeFrame(str, Enum):
    MIN_5 = "5"     # 5-minute charts
    MIN_15 = "15"   # 15-minute charts  
    MIN_30 = "30"   # 30-minute charts
    HOUR_1 = "60"   # 1-hour charts
```

**Usage Examples:**
```python
# Fetch different timeframes
await service.fetch_historical_data(TimeFrame.MIN_5, 100)
await service.fetch_historical_data(TimeFrame.MIN_15, 50)

# Multi-timeframe analysis
timeframes = [TimeFrame.MIN_5, TimeFrame.MIN_15]
for tf in timeframes:
    data = await service.fetch_historical_data(tf, 100)
    print(f"{tf}-minute data: {len(data)} bars")

# String conversion
tf_string = str(TimeFrame.MIN_5)  # "5"
tf_minutes = int(tf_string)       # 5

# Timeframe mapping
TF_SECONDS = {
    TimeFrame.MIN_5: 300,   # 5 * 60
    TimeFrame.MIN_15: 900,  # 15 * 60
    TimeFrame.MIN_30: 1800, # 30 * 60
    TimeFrame.HOUR_1: 3600  # 60 * 60
}
```

**Timeframe Details:**
- **MIN_5**: Primary scalping timeframe, high frequency signals
- **MIN_15**: Trend confirmation timeframe, filter signals
- **MIN_30**: Medium-term trend analysis (future expansion)
- **HOUR_1**: Long-term trend context (future expansion)

## 📊 Type Aliases

### Common Type Patterns

```python
from typing import List, Dict, Optional, Union, Callable
from vector_scalping.models import OHLCVData, TradingSignal

# Data collections
OHLCVList = List[OHLCVData]
SignalList = List[TradingSignal]

# Multi-timeframe data
MultiTimeframeData = Dict[TimeFrame, List[OHLCVData]]

# Price types
Price = float
Volume = float
Timestamp = int

# Statistical types
Percentile = float  # 0-100
Confidence = float  # 0-1
Direction = float   # -1 to 1

# Callback types
BarCallback = Callable[[OHLCVData], None]
SignalCallback = Callable[[TradingSignal], None]
```

**Usage Examples:**
```python
# Type-safe function signatures
def process_ohlcv_data(data: OHLCVList) -> None:
    for bar in data:
        print(f"Price: {bar.close}")

def analyze_multi_timeframe(data: MultiTimeframeData) -> TradingSignal:
    tf5_data = data[TimeFrame.MIN_5]
    tf15_data = data[TimeFrame.MIN_15]
    # Analysis logic...

# Callback function typing
async def on_new_bar(bar: OHLCVData) -> None:
    print(f"New bar received: {bar.close}")

await service.stream_real_time_data(TimeFrame.MIN_5, on_new_bar)
```

## 🎯 Constants

### Strategy Configuration Constants

```python
# Default strategy parameters
DEFAULT_VECTOR_PERIOD = 5
DEFAULT_SIGNAL_THRESHOLD = 60.0
DEFAULT_DIRECTION_THRESHOLD = 0.3
DEFAULT_PERCENTILE_LOOKBACK = 100

# Timeframe weights
DEFAULT_TF5_WEIGHT = 0.7
DEFAULT_TF15_WEIGHT = 0.3
DEFAULT_TF5_DIRECTION_WEIGHT = 0.6
DEFAULT_TF15_DIRECTION_WEIGHT = 0.4

# Risk management defaults
DEFAULT_TAKE_PROFIT_PIPS = 20
DEFAULT_STOP_LOSS_PIPS = 30

# Currency pair pip sizes
MAJOR_PAIRS_PIP_SIZE = 0.0001  # EUR/USD, GBP/USD, etc.
JPY_PAIRS_PIP_SIZE = 0.01      # USD/JPY, EUR/JPY, etc.
```

### Validation Constants

```python
# Vector calculation limits
MIN_VECTOR_PERIOD = 3
MAX_VECTOR_PERIOD = 20
MIN_DATA_POINTS = 5

# Signal strength bounds
MIN_PERCENTILE = 0.0
MAX_PERCENTILE = 100.0

# Direction bounds
MIN_DIRECTION = -1.0
MAX_DIRECTION = 1.0

# Confidence bounds
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# Price validation
MIN_PRICE = 0.0001
MAX_PRICE = 1000000.0

# Volume validation
MIN_VOLUME = 0.0
MAX_VOLUME = 1000000000.0
```

### Time Constants

```python
from datetime import timezone, time

# Market hours (GMT)
MARKET_OPEN_HOUR = 22  # Sunday 10 PM GMT
MARKET_CLOSE_HOUR = 22  # Friday 10 PM GMT

# Exit time (Friday 5 PM GMT)
WEEKLY_EXIT_HOUR = 17
WEEKLY_EXIT_MINUTE = 0
WEEKLY_EXIT_DAY = 4  # Friday (0=Monday)

# Timezone
MARKET_TIMEZONE = timezone.utc

# Session times
ASIAN_SESSION = time(22, 0)    # 10 PM GMT
LONDON_SESSION = time(8, 0)    # 8 AM GMT  
NEW_YORK_SESSION = time(13, 0) # 1 PM GMT
```

## 🔧 Utility Types

### Custom Exceptions

```python
class VectorScalpingError(Exception):
    """Base exception for vector scalping strategy."""
    pass

class InsufficientDataError(VectorScalpingError):
    """Raised when insufficient data for calculations."""
    pass

class InvalidSignalError(VectorScalpingError):
    """Raised when signal validation fails."""
    pass

class ConfigurationError(VectorScalpingError):
    """Raised when configuration is invalid."""
    pass

# Usage
try:
    vector = VectorCalculations.calculate_price_vector(data, period=5)
except InsufficientDataError as e:
    print(f"Need more data: {e}")
```

### Generic Types

```python
from typing import TypeVar, Generic, Protocol

# Generic vector type
T = TypeVar('T')

class Vector(Generic[T]):
    magnitude: float
    direction: float
    data: T

# Protocol for data with timestamp
class TimestampedData(Protocol):
    timestamp: int
    
    def to_datetime(self) -> datetime:
        ...

# Callables
CalculationFunction = Callable[[List[OHLCVData]], float]
ValidationFunction = Callable[[Any], bool]
```

## 📐 Mathematical Types

### Vector Mathematics

```python
from typing import NamedTuple
import numpy as np

# 2D Vector representation
class Vector2D(NamedTuple):
    x: float
    y: float
    
    @property
    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    @property
    def angle(self) -> float:
        return np.arctan2(self.y, self.x)

# Statistical measures
class Statistics(NamedTuple):
    mean: float
    std: float
    min: float
    max: float
    count: int

# Range types
class PriceRange(NamedTuple):
    low: float
    high: float
    range: float
```

### Calculation Results

```python
# Vector calculation intermediate results
class VectorComponents(NamedTuple):
    displacement: float
    time_period: int
    start_price: float
    end_price: float
    price_range: float

# Momentum calculation components
class MomentumComponents(NamedTuple):
    price_change: float
    time_period: int
    true_ranges: List[float]
    average_true_range: float

# Divergence calculation components
class DivergenceComponents(NamedTuple):
    price_slope: float
    momentum_slope: float
    correlation: float
    significance: float
```

## 🎨 Display Types

### Formatting Utilities

```python
from enum import Enum
from typing import Dict, Any

class DisplayPrecision(Enum):
    PRICE_4_DECIMAL = 5      # 1.08500
    PRICE_2_DECIMAL = 3      # 110.50
    PERCENTAGE = 2           # 65.50%
    DIRECTION = 3            # 0.657
    CONFIDENCE = 2           # 0.85

# Format mappings
FORMAT_SPECS: Dict[DisplayPrecision, str] = {
    DisplayPrecision.PRICE_4_DECIMAL: ".5f",
    DisplayPrecision.PRICE_2_DECIMAL: ".3f", 
    DisplayPrecision.PERCENTAGE: ".1f",
    DisplayPrecision.DIRECTION: ".3f",
    DisplayPrecision.CONFIDENCE: ".2f"
}

def format_value(value: float, precision: DisplayPrecision) -> str:
    """Format value according to precision type."""
    spec = FORMAT_SPECS[precision]
    return f"{value:{spec}}"

# Usage
price = 1.08456
formatted = format_value(price, DisplayPrecision.PRICE_4_DECIMAL)  # "1.08456"
```

### Color Coding

```python
class SignalColor(Enum):
    LONG = "🟢"      # Green for bullish
    SHORT = "🔴"     # Red for bearish
    EXIT = "🟡"      # Yellow for exit
    NO_SIGNAL = "⚪" # White for no signal

class StrengthColor(Enum):
    VERY_STRONG = "🟣"  # Purple > 90th percentile
    STRONG = "🔵"       # Blue 70-90th percentile
    MODERATE = "🟡"     # Yellow 40-70th percentile
    WEAK = "🟠"         # Orange 20-40th percentile
    VERY_WEAK = "🔴"    # Red < 20th percentile

def get_signal_display(signal: TradingSignal) -> str:
    """Get colored display for signal."""
    color = SignalColor[signal.signal_type.name].value
    return f"{color} {signal.signal_type}"

def get_strength_display(percentile: float) -> str:
    """Get colored display for strength."""
    if percentile >= 90:
        color = StrengthColor.VERY_STRONG.value
    elif percentile >= 70:
        color = StrengthColor.STRONG.value
    elif percentile >= 40:
        color = StrengthColor.MODERATE.value
    elif percentile >= 20:
        color = StrengthColor.WEAK.value
    else:
        color = StrengthColor.VERY_WEAK.value
    
    return f"{color} {percentile:.1f}%"
```

## 🔍 Type Guards

### Runtime Type Checking

```python
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from vector_scalping.models import TradingSignal, OHLCVData

def is_trading_signal(obj: Any) -> bool:
    """Check if object is a valid trading signal."""
    return (hasattr(obj, 'signal_type') and 
            hasattr(obj, 'confidence') and
            hasattr(obj, 'timestamp'))

def is_ohlcv_data(obj: Any) -> bool:
    """Check if object is valid OHLCV data."""
    required_attrs = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    return all(hasattr(obj, attr) for attr in required_attrs)

def is_valid_signal_type(value: str) -> bool:
    """Check if string is a valid signal type."""
    return value in [e.value for e in SignalType]

def is_valid_timeframe(value: str) -> bool:
    """Check if string is a valid timeframe."""
    return value in [e.value for e in TimeFrame]

# Usage in validation
def validate_signal_data(data: Dict[str, Any]) -> bool:
    """Validate raw signal data before model creation."""
    if not isinstance(data.get('signal_type'), str):
        return False
    
    if not is_valid_signal_type(data['signal_type']):
        return False
    
    confidence = data.get('confidence')
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
        return False
    
    return True
```

## 🧪 Testing Types

### Test Utilities

```python
from typing import Generator, Callable
import pytest

# Test data generators
TestDataGenerator = Generator[OHLCVData, None, None]
SignalTestFunction = Callable[[TradingSignal], bool]

# Mock types
MockOHLCVData = Dict[str, Union[int, float]]
MockSignalData = Dict[str, Any]

# Test fixtures
@pytest.fixture
def sample_ohlcv() -> List[OHLCVData]:
    """Generate sample OHLCV data for testing."""
    return [
        OHLCVData(
            timestamp=1640995200 + i * 300,
            open=1.0850 + i * 0.0001,
            high=1.0860 + i * 0.0001,
            low=1.0845 + i * 0.0001,
            close=1.0855 + i * 0.0001,
            volume=1500 + i * 10
        ) for i in range(20)
    ]

# Assertion helpers
def assert_valid_signal(signal: TradingSignal) -> None:
    """Assert signal meets all validity requirements."""
    assert isinstance(signal.signal_type, SignalType)
    assert 0 <= signal.confidence <= 1
    assert signal.timestamp > 0
    assert len(signal.reason) > 0

def assert_valid_vector(vector: Union[PriceVector, MomentumVector]) -> None:
    """Assert vector meets validity requirements."""
    assert vector.magnitude >= 0
    assert -1 <= vector.direction <= 1
    assert vector.period > 0
```

## 🔗 Related Documentation

- **[Core API](core.md)** - Main classes and functions
- **[Models API](models.md)** - Complete model reference  
- **[Module Guides](../modules/)** - Detailed usage examples
- **[Configuration Guide](../modules/configuration.md)** - Setup and configuration

---

**Previous**: [Models API Reference](models.md) ← | **Next**: [Module Guides](../modules/) →