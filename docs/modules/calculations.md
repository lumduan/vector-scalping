# Calculations Module Documentation

The `calculations.py` module implements the core mathematical algorithms for vector-based trading analysis. It provides static methods for calculating price vectors, momentum vectors, and detecting divergences.

## 📊 Overview

The VectorCalculations class contains all mathematical functions used in the strategy:
- **Price Vector Analysis**: Directional price movement calculations
- **Momentum Vector Analysis**: Combined price and volatility analysis
- **Multi-Timeframe Fusion**: Weighted combination of different timeframes
- **Divergence Detection**: Price vs momentum divergence analysis
- **Statistical Functions**: Percentile ranking and signal strength

## 🧮 Vector Mathematics

### Price Vector Calculation

Price vectors measure directional price movement over a specified period.

```python
from vector_scalping.calculations import VectorCalculations
from vector_scalping.models import OHLCVData

# Sample OHLCV data (5 candles)
data = [
    OHLCVData(timestamp=1640995200, open=1.0850, high=1.0860, low=1.0845, close=1.0855, volume=1500),
    OHLCVData(timestamp=1640995500, open=1.0855, high=1.0865, low=1.0850, close=1.0862, volume=1600),
    OHLCVData(timestamp=1640995800, open=1.0862, high=1.0870, low=1.0858, close=1.0868, volume=1400),
    OHLCVData(timestamp=1640996100, open=1.0868, high=1.0875, low=1.0865, close=1.0872, volume=1800),
    OHLCVData(timestamp=1640996400, open=1.0872, high=1.0880, low=1.0870, close=1.0878, volume=1700),
]

# Calculate price vector
price_vector = VectorCalculations.calculate_price_vector(data, period=5)

print(f"Displacement: {price_vector.displacement:.5f}")      # 0.0023 (1.0878 - 1.0855)
print(f"Magnitude: {price_vector.magnitude:.5f}")           # 0.0023 (absolute movement)
print(f"Direction: {price_vector.direction:.3f}")           # 0.657 (normalized direction)
print(f"Price Range: {price_vector.price_range:.5f}")       # 0.0035 (high-low range)
```

**Mathematical Formula:**
```
displacement = close[n] - close[0]
magnitude = abs(displacement)
direction = displacement / max(high_range, 0.0001)
```

### Momentum Vector Calculation

Momentum vectors combine price momentum with volatility analysis using Average True Range.

```python
# Calculate momentum vector
momentum_vector = VectorCalculations.calculate_momentum_vector(data, period=5)

print(f"Price Momentum: {momentum_vector.price_momentum:.5f}")    # Rate of change
print(f"Volatility (ATR): {momentum_vector.volatility:.5f}")     # Average true range
print(f"Magnitude: {momentum_vector.magnitude:.5f}")             # Combined magnitude
print(f"Direction: {momentum_vector.direction:.3f}")             # Momentum direction
```

**Mathematical Formula:**
```
price_momentum = (close[n] - close[0]) / n
volatility = average_true_range(period)
magnitude = sqrt(price_momentum² + volatility²)
direction = price_momentum / magnitude (if magnitude > 0)
```

### True Range Calculation

True Range measures volatility by considering gaps between periods.

```python
# True Range calculation (automatically done in momentum vector)
def calculate_true_range(current_candle, previous_candle):
    """Calculate True Range for a candle."""
    return max(
        current_candle.high - current_candle.low,                    # Current range
        abs(current_candle.high - previous_candle.close),            # Gap up
        abs(current_candle.low - previous_candle.close)              # Gap down
    )
```

## 🔀 Multi-Timeframe Combination

Combine vectors from different timeframes using weighted averages.

```python
# Calculate vectors for both timeframes
tf5_price = VectorCalculations.calculate_price_vector(tf5_data, period=5)
tf5_momentum = VectorCalculations.calculate_momentum_vector(tf5_data, period=5)

tf15_price = VectorCalculations.calculate_price_vector(tf15_data, period=5)
tf15_momentum = VectorCalculations.calculate_momentum_vector(tf15_data, period=5)

# Combine vectors with custom weights
combined_vector = VectorCalculations.combine_vectors(
    tf5_price=tf5_price,
    tf5_momentum=tf5_momentum,
    tf15_price=tf15_price,
    tf15_momentum=tf15_momentum,
    tf5_weight=0.7,           # 70% weight for 5-minute data
    tf15_weight=0.3,          # 30% weight for 15-minute data
    tf5_dir_weight=0.6,       # 60% weight for 5-minute direction
    tf15_dir_weight=0.4       # 40% weight for 15-minute direction
)

print(f"Combined Magnitude: {combined_vector.combined_magnitude:.5f}")
print(f"Combined Direction: {combined_vector.combined_direction:.3f}")
```

**Mathematical Formula:**
```
combined_magnitude = (tf5_magnitude × 0.7) + (tf15_magnitude × 0.3)
combined_direction = (tf5_direction × 0.6) + (tf15_direction × 0.4)
```

## 📈 Percentile Ranking

Calculate signal strength using percentile ranking of vector magnitudes.

```python
# Historical magnitude values (last 20 calculations)
historical_magnitudes = [0.001, 0.002, 0.0015, 0.003, 0.0025, 
                        0.004, 0.0018, 0.0035, 0.002, 0.0028,
                        0.0032, 0.0022, 0.0038, 0.0026, 0.0041,
                        0.0029, 0.0036, 0.0024, 0.0033, 0.0027]

current_magnitude = 0.0035

# Calculate percentile rank
percentile = VectorCalculations.calculate_percentile_rank(
    current_value=current_magnitude,
    historical_values=historical_magnitudes
)

print(f"Signal Strength: {percentile:.1f}th percentile")
```

**Use Cases:**
- Filter weak signals (below 60th percentile)
- Identify strong momentum periods
- Compare current signal to historical performance

## 🔄 Divergence Detection

Detect divergences between price action and momentum indicators.

```python
# Current period (last 5 candles)
current_data = data[-5:]

# Comparison period (previous 5 candles) 
comparison_data = data[-10:-5]

# Detect divergence
divergence = VectorCalculations.detect_divergence(
    current_data=current_data,
    comparison_data=comparison_data,
    period=5
)

print(f"Price Trend: {divergence.price_trend:.4f}")
print(f"Momentum Trend: {divergence.momentum_trend:.4f}")
print(f"Bullish Divergence: {divergence.is_bullish_divergence}")
print(f"Bearish Divergence: {divergence.is_bearish_divergence}")
print(f"Divergence Strength: {divergence.divergence_strength:.3f}")
```

**Divergence Types:**
- **Bullish Divergence**: Price declining while momentum increasing
- **Bearish Divergence**: Price rising while momentum decreasing

## 🛠️ Practical Examples

### Complete Vector Analysis Workflow

```python
from vector_scalping.calculations import VectorCalculations
from vector_scalping.models import OHLCVData

def analyze_market_vectors(tf5_data, tf15_data, period=5):
    """Complete vector analysis workflow."""
    
    # 1. Calculate individual timeframe vectors
    tf5_price = VectorCalculations.calculate_price_vector(tf5_data, period)
    tf5_momentum = VectorCalculations.calculate_momentum_vector(tf5_data, period)
    
    tf15_price = VectorCalculations.calculate_price_vector(tf15_data, period)
    tf15_momentum = VectorCalculations.calculate_momentum_vector(tf15_data, period)
    
    # 2. Combine timeframes
    combined = VectorCalculations.combine_vectors(
        tf5_price, tf5_momentum, tf15_price, tf15_momentum
    )
    
    # 3. Calculate signal strength (needs historical data)
    historical_magnitudes = get_historical_magnitudes()  # Your implementation
    signal_strength = VectorCalculations.calculate_percentile_rank(
        combined.combined_magnitude, historical_magnitudes
    )
    
    # 4. Check for divergences
    if len(tf5_data) >= period * 2:
        divergence = VectorCalculations.detect_divergence(
            tf5_data[-period:], tf5_data[-period*2:-period], period
        )
    else:
        divergence = None
    
    return {
        'combined_vector': combined,
        'signal_strength': signal_strength,
        'divergence': divergence,
        'tf5_vector': tf5_momentum,
        'tf15_vector': tf15_momentum
    }

# Usage
analysis = analyze_market_vectors(tf5_candles, tf15_candles)
print(f"Signal Strength: {analysis['signal_strength']:.1f}%")
```

### Vector Direction Analysis

```python
def analyze_vector_direction(vector_direction):
    """Analyze vector direction strength."""
    
    if vector_direction > 0.5:
        return "Strong Bullish"
    elif vector_direction > 0.2:
        return "Moderate Bullish"
    elif vector_direction > -0.2:
        return "Neutral"
    elif vector_direction > -0.5:
        return "Moderate Bearish"
    else:
        return "Strong Bearish"

# Example usage
direction_analysis = analyze_vector_direction(combined_vector.combined_direction)
print(f"Market Direction: {direction_analysis}")
```

### Custom Vector Weights

```python
def create_custom_combination(tf5_vec, tf15_vec, market_condition="normal"):
    """Create vector combination based on market conditions."""
    
    if market_condition == "trending":
        # Give more weight to longer timeframe in trending markets
        tf5_weight, tf15_weight = 0.6, 0.4
    elif market_condition == "volatile":
        # Give more weight to shorter timeframe in volatile markets
        tf5_weight, tf15_weight = 0.8, 0.2
    else:
        # Default balanced weights
        tf5_weight, tf15_weight = 0.7, 0.3
    
    return VectorCalculations.combine_vectors(
        tf5_price, tf5_momentum, tf15_price, tf15_momentum,
        tf5_weight=tf5_weight, tf15_weight=tf15_weight
    )
```

## ⚠️ Error Handling

### Input Validation

```python
try:
    # Insufficient data
    vector = VectorCalculations.calculate_price_vector(data[:2], period=5)
except ValueError as e:
    print(f"Error: {e}")  # "Need at least 5 candles, got 2"

try:
    # Invalid period
    vector = VectorCalculations.calculate_price_vector(data, period=0)
except ValueError as e:
    print(f"Error: {e}")  # "Period must be positive"

try:
    # Invalid weights
    combined = VectorCalculations.combine_vectors(
        tf5_price, tf5_momentum, tf15_price, tf15_momentum,
        tf5_weight=0.8, tf15_weight=0.3  # Sum = 1.1
    )
except ValueError as e:
    print(f"Error: {e}")  # "Magnitude weights must sum to 1.0"
```

### Safe Calculations

```python
def safe_vector_calculation(data, period=5):
    """Safely calculate vectors with error handling."""
    try:
        if len(data) < period:
            raise ValueError(f"Insufficient data: need {period}, got {len(data)}")
        
        price_vector = VectorCalculations.calculate_price_vector(data, period)
        momentum_vector = VectorCalculations.calculate_momentum_vector(data, period)
        
        return price_vector, momentum_vector
        
    except Exception as e:
        print(f"Vector calculation failed: {e}")
        return None, None
```

## 🎯 Performance Optimization

### Efficient Data Processing

```python
def batch_vector_calculations(data_windows, period=5):
    """Calculate vectors for multiple data windows efficiently."""
    results = []
    
    for window in data_windows:
        if len(window) >= period:
            price_vec = VectorCalculations.calculate_price_vector(window, period)
            momentum_vec = VectorCalculations.calculate_momentum_vector(window, period)
            results.append((price_vec, momentum_vec))
        else:
            results.append((None, None))
    
    return results
```

### Memory Management

```python
def sliding_window_analysis(data_stream, window_size=100, period=5):
    """Analyze vectors using sliding window for memory efficiency."""
    window = []
    
    for new_candle in data_stream:
        window.append(new_candle)
        
        # Maintain window size
        if len(window) > window_size:
            window.pop(0)
        
        # Calculate vectors when enough data
        if len(window) >= period:
            vectors = VectorCalculations.calculate_price_vector(
                window[-period:], period
            )
            yield vectors
```

## 🧪 Testing Calculations

```python
import pytest
from vector_scalping.calculations import VectorCalculations

def test_price_vector_calculation():
    """Test price vector calculation accuracy."""
    # Known test data
    test_data = create_test_ohlcv_data()
    
    vector = VectorCalculations.calculate_price_vector(test_data, period=5)
    
    # Verify calculations
    expected_displacement = test_data[-1].close - test_data[0].close
    assert vector.displacement == pytest.approx(expected_displacement, abs=1e-6)
    assert vector.magnitude == abs(expected_displacement)
    assert -1 <= vector.direction <= 1

def test_momentum_vector_accuracy():
    """Test momentum vector calculation."""
    test_data = create_test_ohlcv_data()
    
    vector = VectorCalculations.calculate_momentum_vector(test_data, period=5)
    
    assert vector.volatility >= 0
    assert vector.magnitude >= 0
    assert -1 <= vector.direction <= 1
```

## 🔗 Related Documentation

- **[Models Module](models.md)** - Understanding data structures
- **[Signals Module](signals.md)** - Using vectors for signal generation
- **[Data Service](data_service.md)** - Getting data for calculations
- **[API Reference](../api/calculations.md)** - Complete function reference

---

**Next**: Learn about [Signal Generation](signals.md) →