# Signals Module Documentation

The `signals.py` module implements the signal generation logic for the vector scalping strategy. It analyzes vector data to produce trading signals with confidence scoring and risk management.

## 🎯 Overview

The SignalGenerator class provides:
- **Signal Generation**: LONG/SHORT/NO_SIGNAL decisions based on vector analysis
- **Risk Management**: Automated take-profit and stop-loss calculations
- **Confidence Scoring**: Multi-factor confidence assessment
- **Time-Based Exits**: Weekend and session-based exit logic
- **Historical Tracking**: Percentile-based signal strength calculation

## 🚦 Signal Logic

### Entry Conditions

The strategy generates signals based on multi-timeframe vector alignment:

```python
from vector_scalping.signals import SignalGenerator
from vector_scalping.models import StrategyConfig, RiskManagement

# Configure strategy
config = StrategyConfig(
    symbol="EURUSD",
    vector_period=5,
    signal_threshold=60.0,        # Minimum 60th percentile
    direction_threshold=0.3,      # Minimum directional bias
    risk_management=RiskManagement(
        symbol="EURUSD",
        pip_size=0.0001,
        take_profit_pips=20,
        stop_loss_pips=30,
        is_decimal_4=True
    )
)

# Create signal generator
generator = SignalGenerator(config)

# Generate signal
signal = generator.generate_signal(
    tf5_data=five_minute_candles,     # 5-minute OHLCV data
    tf15_data=fifteen_minute_candles, # 15-minute OHLCV data
    current_price=1.0850              # Current market price
)

print(f"Signal: {signal.signal_type}")
print(f"Confidence: {signal.confidence:.2f}")
print(f"Reason: {signal.reason}")
```

### LONG Signal Conditions

A LONG signal is generated when:

```python
# LONG Entry Conditions:
if (combined_direction > 0.3 and           # Bullish bias
    signal_strength > 60 and               # Above 60th percentile
    tf5_direction > 0.2 and               # 5-min confirmation
    tf15_direction > 0.1):                # 15-min trend alignment
    
    return SignalType.LONG
```

**Example:**
```python
# Check if current conditions meet LONG criteria
if signal.signal_type == SignalType.LONG:
    print(f"🟢 LONG Signal Generated")
    print(f"   Entry: {signal.entry_price:.5f}")
    print(f"   Take Profit: {signal.take_profit:.5f}")
    print(f"   Stop Loss: {signal.stop_loss:.5f}")
    print(f"   Confidence: {signal.confidence:.2f}")
```

### SHORT Signal Conditions

A SHORT signal is generated when:

```python
# SHORT Entry Conditions:
if (combined_direction < -0.3 and          # Bearish bias
    signal_strength > 60 and               # Above 60th percentile
    tf5_direction < -0.2 and              # 5-min confirmation
    tf15_direction < -0.1):               # 15-min trend alignment
    
    return SignalType.SHORT
```

### NO_SIGNAL Conditions

No signal is generated when:
- Signal strength below threshold (< 60th percentile)
- Insufficient directional bias
- Conflicting timeframe directions
- Insufficient historical data

## 💪 Confidence Scoring

The confidence score (0-1) is calculated using multiple factors:

```python
def calculate_confidence_factors(signal):
    """Break down confidence calculation."""
    
    base_confidence = 0.5
    
    # Factor 1: Signal Strength
    strength_factor = min((signal.vector_data.signal_strength - 50) / 50, 0.3)
    
    # Factor 2: Direction Alignment
    direction_diff = abs(signal.vector_data.tf5_direction - signal.vector_data.tf15_direction)
    alignment_factor = max(0.2 - direction_diff, 0)
    
    # Factor 3: Directional Bias Strength
    direction_strength = abs(signal.vector_data.combined_direction)
    bias_factor = 0.1 if direction_strength > 0.5 else 0
    
    # Factor 4: Divergence Boost
    divergence_factor = 0
    if signal.divergence_data:
        if ((signal.signal_type == SignalType.LONG and signal.divergence_data.is_bullish_divergence) or
            (signal.signal_type == SignalType.SHORT and signal.divergence_data.is_bearish_divergence)):
            divergence_factor = signal.divergence_data.divergence_strength * 0.15
    
    total_confidence = min(
        base_confidence + strength_factor + alignment_factor + bias_factor + divergence_factor,
        1.0
    )
    
    return {
        'base': base_confidence,
        'strength': strength_factor,
        'alignment': alignment_factor,
        'bias': bias_factor,
        'divergence': divergence_factor,
        'total': total_confidence
    }
```

## 💰 Risk Management

### Take Profit and Stop Loss Calculation

```python
# Automatic calculation based on symbol configuration
entry_price = 1.0850

# For 4-decimal pairs (EUR/USD, GBP/USD, etc.)
risk_mgmt = RiskManagement(
    symbol="EURUSD",
    pip_size=0.0001,              # 4-decimal pip size
    take_profit_pips=20,          # 20-pip target
    stop_loss_pips=30,            # 30-pip stop
    is_decimal_4=True
)

# Calculate levels
tp_long = risk_mgmt.calculate_take_profit(entry_price, SignalType.LONG)   # 1.0870
sl_long = risk_mgmt.calculate_stop_loss(entry_price, SignalType.LONG)     # 1.0820

tp_short = risk_mgmt.calculate_take_profit(entry_price, SignalType.SHORT) # 1.0830
sl_short = risk_mgmt.calculate_stop_loss(entry_price, SignalType.SHORT)   # 1.0880

print(f"LONG - TP: {tp_long:.5f}, SL: {sl_long:.5f}")
print(f"SHORT - TP: {tp_short:.5f}, SL: {sl_short:.5f}")
```

### Currency Pair Configuration

```python
# EUR/USD (4-decimal pair)
eurusd_config = RiskManagement(
    symbol="EURUSD",
    pip_size=0.0001,      # 1 pip = 0.0001
    is_decimal_4=True
)

# USD/JPY (2-decimal pair)
usdjpy_config = RiskManagement(
    symbol="USDJPY", 
    pip_size=0.01,        # 1 pip = 0.01
    is_decimal_4=False
)

# GBP/JPY (2-decimal pair)
gbpjpy_config = RiskManagement(
    symbol="GBPJPY",
    pip_size=0.01,
    is_decimal_4=False
)
```

## ⏰ Time-Based Risk Management

### Weekend Exit Logic

```python
# Check for time-based exit (Friday 5 PM GMT)
should_exit = generator.should_exit_time_based()

if should_exit:
    print("🕐 Time-based exit triggered - Friday 5 PM GMT")
    # Close all positions regardless of profit/loss
```

### Custom Time Rules

```python
from datetime import datetime, timezone

def custom_time_check():
    """Custom time-based rules."""
    now = datetime.now(timezone.utc)
    
    # Avoid trading during major news events
    if now.hour == 14 and now.minute < 30:  # 2:00-2:30 PM GMT
        return True, "Major news event"
    
    # Avoid trading during low liquidity
    if now.weekday() == 6:  # Sunday
        return True, "Weekend - low liquidity"
    
    # Avoid trading during market open gaps
    if now.hour == 22 and now.minute < 15:  # Market open
        return True, "Market open gap risk"
    
    return False, "Normal trading hours"

exit_required, reason = custom_time_check()
if exit_required:
    print(f"⏰ Exit required: {reason}")
```

## 📊 Signal Strength Analysis

### Historical Magnitude Tracking

```python
# The generator automatically tracks historical magnitudes
generator = SignalGenerator(config)

# Generate multiple signals to build history
for i, (tf5_data, tf15_data, price) in enumerate(data_stream):
    signal = generator.generate_signal(tf5_data, tf15_data, price)
    
    print(f"Signal {i+1}:")
    print(f"  Type: {signal.signal_type}")
    print(f"  Strength: {signal.vector_data.signal_strength:.1f}th percentile")
    print(f"  Historical count: {len(generator.historical_magnitudes)}")
```

### Signal Quality Metrics

```python
def analyze_signal_quality(signal):
    """Analyze signal quality metrics."""
    
    if signal.signal_type == SignalType.NO_SIGNAL:
        return {"quality": "No Signal", "score": 0}
    
    quality_score = 0
    factors = []
    
    # High confidence
    if signal.confidence > 0.8:
        quality_score += 30
        factors.append("High Confidence")
    elif signal.confidence > 0.6:
        quality_score += 20
        factors.append("Good Confidence")
    
    # Strong signal strength
    if signal.vector_data.signal_strength > 80:
        quality_score += 25
        factors.append("Very Strong Signal")
    elif signal.vector_data.signal_strength > 60:
        quality_score += 15
        factors.append("Strong Signal")
    
    # Direction alignment
    direction_diff = abs(signal.vector_data.tf5_direction - signal.vector_data.tf15_direction)
    if direction_diff < 0.1:
        quality_score += 20
        factors.append("Perfect Alignment")
    elif direction_diff < 0.2:
        quality_score += 10
        factors.append("Good Alignment")
    
    # Divergence confirmation
    if signal.divergence_data and (
        (signal.signal_type == SignalType.LONG and signal.divergence_data.is_bullish_divergence) or
        (signal.signal_type == SignalType.SHORT and signal.divergence_data.is_bearish_divergence)
    ):
        quality_score += 25
        factors.append("Divergence Confirmation")
    
    # Determine quality level
    if quality_score >= 80:
        quality = "Excellent"
    elif quality_score >= 60:
        quality = "Good"
    elif quality_score >= 40:
        quality = "Fair"
    else:
        quality = "Poor"
    
    return {
        "quality": quality,
        "score": quality_score,
        "factors": factors
    }

# Usage
quality = analyze_signal_quality(signal)
print(f"Signal Quality: {quality['quality']} ({quality['score']}/100)")
print(f"Factors: {', '.join(quality['factors'])}")
```

## 🔄 Divergence Integration

### Divergence-Enhanced Signals

```python
# Signals are automatically enhanced with divergence analysis
if signal.divergence_data:
    print(f"📈 Divergence Analysis:")
    print(f"  Price Trend: {signal.divergence_data.price_trend:.4f}")
    print(f"  Momentum Trend: {signal.divergence_data.momentum_trend:.4f}")
    print(f"  Bullish Divergence: {signal.divergence_data.is_bullish_divergence}")
    print(f"  Bearish Divergence: {signal.divergence_data.is_bearish_divergence}")
    print(f"  Strength: {signal.divergence_data.divergence_strength:.3f}")
    
    # Divergence gives confidence boost
    if ((signal.signal_type == SignalType.LONG and signal.divergence_data.is_bullish_divergence) or
        (signal.signal_type == SignalType.SHORT and signal.divergence_data.is_bearish_divergence)):
        print("✅ Divergence confirms signal direction!")
```

## 🛠️ Advanced Usage

### Custom Signal Filters

```python
class AdvancedSignalGenerator(SignalGenerator):
    """Extended signal generator with custom filters."""
    
    def __init__(self, config, custom_filters=None):
        super().__init__(config)
        self.custom_filters = custom_filters or []
    
    def generate_signal(self, tf5_data, tf15_data, current_price):
        """Generate signal with custom filtering."""
        
        # Get base signal
        signal = super().generate_signal(tf5_data, tf15_data, current_price)
        
        # Apply custom filters
        for filter_func in self.custom_filters:
            if not filter_func(signal, tf5_data, tf15_data):
                return TradingSignal(
                    signal_type=SignalType.NO_SIGNAL,
                    confidence=0.0,
                    timestamp=int(time.time()),
                    reason="Filtered by custom filter"
                )
        
        return signal

# Custom filter examples
def volatility_filter(signal, tf5_data, tf15_data):
    """Filter out signals during high volatility."""
    recent_ranges = [candle.high - candle.low for candle in tf5_data[-5:]]
    avg_range = sum(recent_ranges) / len(recent_ranges)
    return avg_range < 0.005  # Max 50 pips average range

def volume_filter(signal, tf5_data, tf15_data):
    """Filter out signals during low volume."""
    recent_volume = sum(candle.volume for candle in tf5_data[-3:])
    return recent_volume > 10000  # Minimum volume threshold

# Usage
custom_generator = AdvancedSignalGenerator(
    config=config,
    custom_filters=[volatility_filter, volume_filter]
)
```

### Portfolio Signal Management

```python
class PortfolioSignalManager:
    """Manage signals across multiple currency pairs."""
    
    def __init__(self, symbols_config):
        self.generators = {
            symbol: SignalGenerator(config) 
            for symbol, config in symbols_config.items()
        }
        self.active_signals = {}
    
    def update_signals(self, market_data):
        """Update signals for all symbols."""
        new_signals = {}
        
        for symbol, generator in self.generators.items():
            if symbol in market_data:
                data = market_data[symbol]
                signal = generator.generate_signal(
                    data['tf5'], data['tf15'], data['current_price']
                )
                new_signals[symbol] = signal
        
        self.active_signals = new_signals
        return new_signals
    
    def get_best_signal(self):
        """Get the highest confidence signal."""
        if not self.active_signals:
            return None
        
        valid_signals = [
            (symbol, signal) for symbol, signal in self.active_signals.items()
            if signal.signal_type != SignalType.NO_SIGNAL
        ]
        
        if not valid_signals:
            return None
        
        return max(valid_signals, key=lambda x: x[1].confidence)

# Usage
portfolio_manager = PortfolioSignalManager({
    'EURUSD': eurusd_config,
    'GBPUSD': gbpusd_config,
    'USDJPY': usdjpy_config
})

signals = portfolio_manager.update_signals(market_data)
best_signal = portfolio_manager.get_best_signal()

if best_signal:
    symbol, signal = best_signal
    print(f"Best signal: {symbol} - {signal.signal_type} (confidence: {signal.confidence:.2f})")
```

## 🧪 Testing Signals

### Unit Testing

```python
import pytest
from vector_scalping.signals import SignalGenerator
from vector_scalping.models import SignalType

def test_long_signal_generation():
    """Test LONG signal generation."""
    generator = SignalGenerator(config)
    
    # Create bullish test data
    bullish_tf5 = create_bullish_test_data()
    bullish_tf15 = create_bullish_test_data()
    
    # Pre-populate with high historical values to ensure strong signal
    generator.historical_magnitudes = [0.001] * 15 + [0.005] * 5
    
    signal = generator.generate_signal(bullish_tf5, bullish_tf15, 1.0850)
    
    assert signal.signal_type == SignalType.LONG
    assert signal.confidence > 0.0
    assert signal.entry_price == 1.0850
    assert signal.take_profit > signal.entry_price
    assert signal.stop_loss < signal.entry_price

def test_no_signal_weak_strength():
    """Test NO_SIGNAL for weak signal strength."""
    generator = SignalGenerator(config)
    
    # Pre-populate with very high historical values
    generator.historical_magnitudes = [0.1] * 20  # Very high values
    
    signal = generator.generate_signal(test_data, test_data, 1.0850)
    
    assert signal.signal_type == SignalType.NO_SIGNAL
    assert "below threshold" in signal.reason
```

### Integration Testing

```python
async def test_full_signal_workflow():
    """Test complete signal generation workflow."""
    
    # Setup
    config = create_test_config()
    generator = SignalGenerator(config)
    
    # Get real market data
    async with DataService(config) as service:
        data = await service.fetch_multi_timeframe_data(100)
        current_price = await service.get_latest_price()
    
    # Generate signal
    signal = generator.generate_signal(
        data[TimeFrame.MIN_5],
        data[TimeFrame.MIN_15],
        current_price
    )
    
    # Verify signal integrity
    assert isinstance(signal.signal_type, SignalType)
    assert 0 <= signal.confidence <= 1
    assert signal.timestamp > 0
    assert len(signal.reason) > 0
    
    if signal.signal_type != SignalType.NO_SIGNAL:
        assert signal.entry_price is not None
        assert signal.take_profit is not None
        assert signal.stop_loss is not None
```

## 🔗 Related Documentation

- **[Models Module](models.md)** - Understanding signal data structures
- **[Calculations Module](calculations.md)** - Vector calculations used in signals
- **[Data Service](data_service.md)** - Getting market data for signals
- **[API Reference](../api/signals.md)** - Complete signal API reference

---

**Next**: Learn about [Data Service](data_service.md) →