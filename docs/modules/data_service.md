# Data Service Module Documentation

The `data_service.py` module provides async data fetching and processing capabilities using the tvkit library for real-time forex data and Polars for high-performance data analysis.

## 🌐 Overview

The DataService class offers:
- **Async Data Fetching**: Non-blocking real-time and historical data retrieval
- **Multi-Timeframe Support**: Concurrent fetching of 5-min and 15-min data
- **Polars Integration**: High-performance DataFrame operations
- **Technical Indicators**: Built-in technical analysis calculations
- **Error Handling**: Robust error handling and retry mechanisms
- **Data Validation**: Automatic data validation using Pydantic models

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from vector_scalping.data_service import DataService
from vector_scalping.models import StrategyConfig, RiskManagement, TimeFrame

async def fetch_forex_data():
    """Basic data fetching example."""
    
    # Configure for EUR/USD
    config = StrategyConfig(
        symbol="EURUSD",
        exchange="FX_IDC",
        risk_management=RiskManagement(
            symbol="EURUSD",
            pip_size=0.0001,
            is_decimal_4=True
        )
    )
    
    # Use async context manager
    async with DataService(config) as service:
        # Fetch 5-minute data
        data_5m = await service.fetch_historical_data(TimeFrame.MIN_5, bars_count=100)
        print(f"Fetched {len(data_5m)} 5-minute bars")
        
        # Fetch 15-minute data  
        data_15m = await service.fetch_historical_data(TimeFrame.MIN_15, bars_count=50)
        print(f"Fetched {len(data_15m)} 15-minute bars")
        
        # Get latest price
        latest_price = await service.get_latest_price()
        print(f"Latest price: {latest_price:.5f}")

# Run the example
asyncio.run(fetch_forex_data())
```

### Multi-Timeframe Data Fetching

```python
async def fetch_multi_timeframe():
    """Fetch multiple timeframes concurrently."""
    
    async with DataService(config) as service:
        # Fetch both timeframes concurrently for better performance
        data = await service.fetch_multi_timeframe_data(bars_count=100)
        
        tf5_data = data[TimeFrame.MIN_5]
        tf15_data = data[TimeFrame.MIN_15]
        
        print(f"5-min data: {len(tf5_data)} bars")
        print(f"15-min data: {len(tf15_data)} bars")
        
        # Data is automatically sorted by timestamp
        print(f"Latest 5-min bar: {tf5_data[-1].close:.5f}")
        print(f"Latest 15-min bar: {tf15_data[-1].close:.5f}")

asyncio.run(fetch_multi_timeframe())
```

## 📊 Polars DataFrame Integration

### Converting to Polars DataFrames

```python
import polars as pl

async def polars_analysis():
    """Analyze data using Polars DataFrames."""
    
    async with DataService(config) as service:
        # Fetch OHLCV data
        ohlcv_data = await service.fetch_historical_data(TimeFrame.MIN_5, 200)
        
        # Convert to Polars DataFrame
        df = service.convert_to_polars_dataframe(ohlcv_data)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns}")
        
        # Display basic info
        print("\nDataFrame sample:")
        print(df.select(["timestamp", "open", "high", "low", "close", "volume"]).head())
        
        # Basic statistics
        stats = df.select([
            pl.col("close").mean().alias("avg_close"),
            pl.col("close").std().alias("close_volatility"),
            pl.col("volume").mean().alias("avg_volume"),
            pl.col("high").max().alias("max_high"),
            pl.col("low").min().alias("min_low")
        ])
        
        print(f"\nStatistics:")
        print(stats)

asyncio.run(polars_analysis())
```

### Technical Indicators with Polars

```python
async def technical_analysis():
    """Add technical indicators using Polars."""
    
    async with DataService(config) as service:
        ohlcv_data = await service.fetch_historical_data(TimeFrame.MIN_5, 200)
        
        # Convert to DataFrame
        df = service.convert_to_polars_dataframe(ohlcv_data)
        
        # Add technical indicators
        df_with_indicators = service.add_technical_indicators(df)
        
        # Display indicators
        indicators = df_with_indicators.select([
            "timestamp", "close", "sma_5", "sma_20", "atr_5", "volatility_5"
        ]).tail(10)
        
        print("Technical Indicators (last 10 bars):")
        print(indicators)
        
        # Calculate signal conditions
        latest = df_with_indicators.tail(1)
        close = latest["close"][0]
        sma_5 = latest["sma_5"][0]
        sma_20 = latest["sma_20"][0]
        
        if sma_5 and sma_20:  # Check for None values
            if close > sma_5 > sma_20:
                print("📈 Bullish trend: Price > SMA(5) > SMA(20)")
            elif close < sma_5 < sma_20:
                print("📉 Bearish trend: Price < SMA(5) < SMA(20)")

asyncio.run(technical_analysis())
```

## 🔧 Advanced Polars Operations

### Custom Technical Indicators

```python
def add_custom_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Add custom technical indicators."""
    
    return df.with_columns([
        # Bollinger Bands
        (pl.col("close").rolling_mean(20) + (pl.col("close").rolling_std(20) * 2)).alias("bb_upper"),
        (pl.col("close").rolling_mean(20) - (pl.col("close").rolling_std(20) * 2)).alias("bb_lower"),
        
        # RSI approximation
        pl.col("returns").rolling_mean(14).alias("rsi_approx"),
        
        # Price position within range
        ((pl.col("close") - pl.col("low").rolling_min(20)) / 
         (pl.col("high").rolling_max(20) - pl.col("low").rolling_min(20))).alias("price_position"),
        
        # Volatility ratio
        (pl.col("true_range") / pl.col("atr_14")).alias("volatility_ratio"),
        
        # Trend strength
        (pl.col("close") / pl.col("sma_20")).alias("trend_strength")
    ])

# Usage
async def custom_analysis():
    async with DataService(config) as service:
        data = await service.fetch_historical_data(TimeFrame.MIN_5, 300)
        df = service.convert_to_polars_dataframe(data)
        df = service.add_technical_indicators(df)
        df = add_custom_indicators(df)
        
        # Filter for strong trend conditions
        strong_trends = df.filter(
            (pl.col("trend_strength") > 1.02) |  # Price > 2% above SMA(20)
            (pl.col("trend_strength") < 0.98)    # Price > 2% below SMA(20)
        )
        
        print(f"Strong trend periods: {len(strong_trends)} out of {len(df)}")
```

### Data Filtering and Analysis

```python
async def advanced_filtering():
    """Advanced data filtering with Polars."""
    
    async with DataService(config) as service:
        data = await service.fetch_historical_data(TimeFrame.MIN_5, 500)
        df = service.convert_to_polars_dataframe(data)
        df = service.add_technical_indicators(df)
        
        # Filter for high-volume periods
        high_volume = df.filter(
            pl.col("volume") > pl.col("volume").quantile(0.8)  # Top 20% volume
        )
        
        # Filter for high volatility
        high_volatility = df.filter(
            pl.col("volatility_5") > pl.col("volatility_5").quantile(0.9)  # Top 10% volatility
        )
        
        # Combine conditions
        significant_moves = df.filter(
            (pl.col("volume") > pl.col("volume").quantile(0.8)) &
            (pl.col("volatility_5") > pl.col("volatility_5").quantile(0.8)) &
            (pl.col("price_change").abs() > pl.col("price_change").abs().quantile(0.9))
        )
        
        print(f"High volume periods: {len(high_volume)}")
        print(f"High volatility periods: {len(high_volatility)}")
        print(f"Significant moves: {len(significant_moves)}")
        
        # Analyze significant moves
        if len(significant_moves) > 0:
            move_stats = significant_moves.select([
                pl.col("price_change").mean().alias("avg_move"),
                pl.col("price_change").std().alias("move_volatility"),
                pl.col("volume").mean().alias("avg_volume"),
                pl.col("volatility_5").mean().alias("avg_volatility")
            ])
            print("\nSignificant move statistics:")
            print(move_stats)

asyncio.run(advanced_filtering())
```

## 🔄 Real-Time Data Streaming

### Live Data Processing

```python
async def live_data_stream():
    """Process live data stream."""
    
    async def process_new_bar(bar):
        """Callback for processing new bars."""
        print(f"📊 New bar: {bar.datetime} - Close: {bar.close:.5f}")
        
        # Add your real-time processing logic here
        # For example: update indicators, check signals, etc.
    
    async with DataService(config) as service:
        # Start real-time streaming
        await service.stream_real_time_data(
            timeframe=TimeFrame.MIN_5,
            callback=process_new_bar
        )

# Note: This would run indefinitely in a real application
# asyncio.run(live_data_stream())
```

### Streaming with Signal Generation

```python
from vector_scalping.signals import SignalGenerator

async def live_trading_signals():
    """Generate signals from live data stream."""
    
    signal_generator = SignalGenerator(config)
    recent_5m_data = []
    recent_15m_data = []
    
    async def process_5m_bar(bar):
        """Process new 5-minute bar."""
        recent_5m_data.append(bar)
        
        # Keep only recent data
        if len(recent_5m_data) > 100:
            recent_5m_data.pop(0)
        
        # Generate signal if enough data
        if len(recent_5m_data) >= 20 and len(recent_15m_data) >= 10:
            signal = signal_generator.generate_signal(
                recent_5m_data, recent_15m_data, bar.close
            )
            
            if signal.signal_type != SignalType.NO_SIGNAL:
                print(f"🚨 {signal.signal_type} signal - Confidence: {signal.confidence:.2f}")
    
    async with DataService(config) as service:
        # In a real implementation, you'd need to handle both timeframes
        await service.stream_real_time_data(TimeFrame.MIN_5, process_5m_bar)
```

## 📈 Data Validation and Quality

### Symbol Format Validation

```python
# Validate symbol formats
valid_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
invalid_symbols = ["EUR/USD", "eur_usd", "EURUSA", "EUR123"]

async with DataService(config) as service:
    for symbol in valid_symbols:
        is_valid = service.validate_symbol_format(symbol)
        print(f"{symbol}: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    for symbol in invalid_symbols:
        is_valid = service.validate_symbol_format(symbol)
        print(f"{symbol}: {'✅ Valid' if is_valid else '❌ Invalid'}")
```

### Data Quality Checks

```python
async def data_quality_analysis():
    """Analyze data quality."""
    
    async with DataService(config) as service:
        data = await service.fetch_historical_data(TimeFrame.MIN_5, 1000)
        df = service.convert_to_polars_dataframe(data)
        
        # Check for data gaps
        time_diffs = df.select(
            (pl.col("timestamp").diff() / 300).alias("time_gap_minutes")  # 5-min = 300 seconds
        ).drop_nulls()
        
        gaps = time_diffs.filter(pl.col("time_gap_minutes") > 5.5)  # Allow small tolerance
        print(f"Data gaps found: {len(gaps)}")
        
        # Check for price anomalies
        price_changes = df.select(
            (pl.col("close").pct_change() * 100).alias("price_change_pct")
        ).drop_nulls()
        
        large_moves = price_changes.filter(pl.col("price_change_pct").abs() > 1.0)  # >1% moves
        print(f"Large price moves (>1%): {len(large_moves)}")
        
        # Check for zero volume
        zero_volume = df.filter(pl.col("volume") == 0)
        print(f"Zero volume bars: {len(zero_volume)}")
        
        # Data completeness
        total_expected = 1000
        actual_received = len(df)
        completeness = (actual_received / total_expected) * 100
        print(f"Data completeness: {completeness:.1f}%")

asyncio.run(data_quality_analysis())
```

## ⚡ Performance Optimization

### Efficient Data Processing

```python
async def optimized_data_processing():
    """Demonstrate optimized data processing techniques."""
    
    async with DataService(config) as service:
        # Fetch data once, process multiple ways
        start_time = time.time()
        
        data = await service.fetch_multi_timeframe_data(1000)
        fetch_time = time.time() - start_time
        
        # Convert to Polars for fast processing
        conversion_start = time.time()
        df_5m = service.convert_to_polars_dataframe(data[TimeFrame.MIN_5])
        df_15m = service.convert_to_polars_dataframe(data[TimeFrame.MIN_15])
        conversion_time = time.time() - conversion_start
        
        # Add indicators
        indicator_start = time.time()
        df_5m = service.add_technical_indicators(df_5m)
        df_15m = service.add_technical_indicators(df_15m)
        indicator_time = time.time() - indicator_start
        
        print(f"Data fetch: {fetch_time:.3f}s")
        print(f"Conversion: {conversion_time:.3f}s")
        print(f"Indicators: {indicator_time:.3f}s")
        print(f"Total: {fetch_time + conversion_time + indicator_time:.3f}s")

asyncio.run(optimized_data_processing())
```

### Memory Management

```python
async def memory_efficient_processing():
    """Process large datasets efficiently."""
    
    async with DataService(config) as service:
        # Process data in chunks to manage memory
        chunk_size = 500
        total_bars = 5000
        
        results = []
        
        for i in range(0, total_bars, chunk_size):
            chunk_data = await service.fetch_historical_data(
                TimeFrame.MIN_5, 
                bars_count=min(chunk_size, total_bars - i)
            )
            
            # Process chunk
            df = service.convert_to_polars_dataframe(chunk_data)
            df = service.add_technical_indicators(df)
            
            # Extract only what we need
            summary = df.select([
                pl.col("close").mean().alias("avg_close"),
                pl.col("volume").sum().alias("total_volume"),
                pl.col("volatility_5").mean().alias("avg_volatility")
            ])
            
            results.append(summary)
            
            print(f"Processed chunk {i//chunk_size + 1}")
        
        # Combine results
        final_summary = pl.concat(results)
        print(f"Final summary: {final_summary}")
```

## 🛠️ Error Handling

### Robust Error Handling

```python
async def robust_data_fetching():
    """Demonstrate robust error handling."""
    
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            async with DataService(config) as service:
                data = await service.fetch_historical_data(TimeFrame.MIN_5, 100)
                print(f"✅ Data fetched successfully: {len(data)} bars")
                break
                
        except RuntimeError as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                print("🔥 All attempts failed")
                raise
        
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            raise

asyncio.run(robust_data_fetching())
```

### Custom Error Handling

```python
class DataServiceError(Exception):
    """Custom exception for data service errors."""
    pass

class EnhancedDataService(DataService):
    """Enhanced data service with custom error handling."""
    
    async def safe_fetch_data(self, timeframe, bars_count, max_retries=3):
        """Safely fetch data with retries."""
        
        for attempt in range(max_retries):
            try:
                return await self.fetch_historical_data(timeframe, bars_count)
                
            except RuntimeError as e:
                if "No data received" in str(e):
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        raise DataServiceError(f"Failed to fetch data after {max_retries} attempts")
                else:
                    raise DataServiceError(f"Data fetch error: {e}")
            
            except Exception as e:
                raise DataServiceError(f"Unexpected error: {e}")
        
        raise DataServiceError("Max retries exceeded")

# Usage
async def safe_data_usage():
    config = create_config()
    
    async with EnhancedDataService(config) as service:
        try:
            data = await service.safe_fetch_data(TimeFrame.MIN_5, 100)
            print(f"Data fetched: {len(data)} bars")
        except DataServiceError as e:
            print(f"Data service error: {e}")
```

## 🧪 Testing Data Service

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_data_service_initialization():
    """Test data service initialization."""
    service = DataService(config)
    assert service.config == config
    assert service._client is None

@pytest.mark.asyncio 
async def test_fetch_historical_data():
    """Test historical data fetching."""
    with patch('vector_scalping.data_service.OHLCV') as mock_ohlcv:
        mock_client = AsyncMock()
        mock_bars = create_mock_bars()
        mock_client.get_historical_ohlcv.return_value = mock_bars
        mock_ohlcv.return_value = mock_client
        
        service = DataService(config)
        service._client = mock_client
        
        result = await service.fetch_historical_data(TimeFrame.MIN_5, 10)
        
        assert len(result) == len(mock_bars)
        assert all(isinstance(bar, OHLCVData) for bar in result)
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_data_workflow():
    """Test complete data workflow."""
    
    async with DataService(config) as service:
        # Test multi-timeframe fetch
        data = await service.fetch_multi_timeframe_data(50)
        
        assert TimeFrame.MIN_5 in data
        assert TimeFrame.MIN_15 in data
        assert len(data[TimeFrame.MIN_5]) > 0
        assert len(data[TimeFrame.MIN_15]) > 0
        
        # Test DataFrame conversion
        df = service.convert_to_polars_dataframe(data[TimeFrame.MIN_5])
        assert isinstance(df, pl.DataFrame)
        assert len(df) == len(data[TimeFrame.MIN_5])
        
        # Test technical indicators
        df_with_indicators = service.add_technical_indicators(df)
        expected_columns = ["sma_5", "sma_10", "atr_5", "volatility_5"]
        assert all(col in df_with_indicators.columns for col in expected_columns)
```

## 🔗 Related Documentation

- **[Models Module](models.md)** - Understanding data structures
- **[Calculations Module](calculations.md)** - Processing fetched data
- **[Signals Module](signals.md)** - Using data for signal generation
- **[API Reference](../api/data_service.md)** - Complete API reference

---

**Next**: Learn about [Configuration](configuration.md) →