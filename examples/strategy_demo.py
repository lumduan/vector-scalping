#!/usr/bin/env python3
"""
Vector Scalping Strategy Demo Script

This script demonstrates the complete vector scalping trading strategy implementation.
It fetches real EUR/USD data, computes vectors, and generates trading signals.

Usage:
    uv run python examples/strategy_demo.py

Features demonstrated:
- Real-time data fetching using tvkit
- Vector calculations (price and momentum)
- Multi-timeframe analysis (5-min and 15-min)
- Signal generation with entry/exit conditions
- Risk management calculations
- Polars DataFrame integration for analysis
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List
import polars as pl

from vector_scalping import (
    DataService,
    SignalGenerator,
    VectorCalculations,
    StrategyConfig,
    RiskManagement,
    OHLCVData,
    TimeFrame,
    SignalType
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_strategy_config() -> StrategyConfig:
    """
    Create strategy configuration for EUR/USD.
    
    Returns:
        Configured StrategyConfig for EUR/USD trading
    """
    # Risk management for EUR/USD (4-decimal pair)
    risk_mgmt = RiskManagement(
        symbol="EURUSD",
        pip_size=0.0001,
        take_profit_pips=20,
        stop_loss_pips=30,
        is_decimal_4=True
    )
    
    # Strategy configuration
    config = StrategyConfig(
        symbol="EURUSD",
        exchange="FX_IDC",
        vector_period=5,
        percentile_lookback=100,
        signal_threshold=60.0,
        direction_threshold=0.3,
        tf5_weight=0.7,
        tf15_weight=0.3,
        tf5_direction_weight=0.6,
        tf15_direction_weight=0.4,
        risk_management=risk_mgmt
    )
    
    logger.info(f"Created strategy config for {config.symbol}")
    logger.info(f"Vector period: {config.vector_period} candles")
    logger.info(f"Signal threshold: {config.signal_threshold}th percentile")
    logger.info(f"Take profit: {config.risk_management.take_profit_pips} pips")
    
    return config


async def fetch_market_data(
    data_service: DataService, 
    bars_count: int = 100
) -> Dict[TimeFrame, List[OHLCVData]]:
    """
    Fetch multi-timeframe market data.
    
    Args:
        data_service: Data service instance
        bars_count: Number of bars to fetch for each timeframe
        
    Returns:
        Dictionary with 5-min and 15-min data
    """
    logger.info(f"Fetching {bars_count} bars for both 5-min and 15-min timeframes...")
    
    start_time = time.time()
    data = await data_service.fetch_multi_timeframe_data(bars_count)
    fetch_time = time.time() - start_time
    
    logger.info(f"Data fetch completed in {fetch_time:.2f} seconds")
    logger.info(f"5-min data: {len(data[TimeFrame.MIN_5])} bars")
    logger.info(f"15-min data: {len(data[TimeFrame.MIN_15])} bars")
    
    # Show data summary
    if data[TimeFrame.MIN_5]:
        latest_5m = data[TimeFrame.MIN_5][-1]
        earliest_5m = data[TimeFrame.MIN_5][0]
        logger.info(f"5-min range: {datetime.fromtimestamp(earliest_5m.timestamp)} to {datetime.fromtimestamp(latest_5m.timestamp)}")
        logger.info(f"Latest 5-min close: {latest_5m.close}")
    
    return data


def analyze_data_with_polars(
    data_service: DataService,
    ohlcv_data: List[OHLCVData]
) -> pl.DataFrame:
    """
    Analyze data using Polars DataFrame with technical indicators.
    
    Args:
        data_service: Data service instance
        ohlcv_data: OHLCV data to analyze
        
    Returns:
        Polars DataFrame with technical indicators
    """
    logger.info("Converting data to Polars DataFrame and adding technical indicators...")
    
    # Convert to Polars DataFrame
    df = data_service.convert_to_polars_dataframe(ohlcv_data)
    
    # Add technical indicators
    df_with_indicators = data_service.add_technical_indicators(df)
    
    logger.info(f"DataFrame shape: {df_with_indicators.shape}")
    logger.info(f"Columns: {df_with_indicators.columns}")
    
    # Show some statistics
    if not df_with_indicators.is_empty():
        stats = df_with_indicators.select([
            pl.col("close").mean().alias("avg_close"),
            pl.col("close").std().alias("close_volatility"),
            pl.col("volume").mean().alias("avg_volume"),
            pl.col("atr_5").mean().alias("avg_atr_5")
        ]).row(0, named=True)
        
        logger.info(f"Average close: {stats['avg_close']:.5f}")
        logger.info(f"Close volatility: {stats['close_volatility']:.5f}")
        logger.info(f"Average volume: {stats['avg_volume']:.0f}")
        logger.info(f"Average ATR(5): {stats['avg_atr_5']:.5f}")
    
    return df_with_indicators


def demonstrate_vector_calculations(
    tf5_data: List[OHLCVData],
    tf15_data: List[OHLCVData],
    config: StrategyConfig
) -> None:
    """
    Demonstrate vector calculations step by step.
    
    Args:
        tf5_data: 5-minute OHLCV data
        tf15_data: 15-minute OHLCV data
        config: Strategy configuration
    """
    logger.info("=" * 60)
    logger.info("VECTOR CALCULATIONS DEMONSTRATION")
    logger.info("=" * 60)
    
    # Calculate 5-minute vectors
    tf5_price = VectorCalculations.calculate_price_vector(tf5_data, config.vector_period)
    tf5_momentum = VectorCalculations.calculate_momentum_vector(tf5_data, config.vector_period)
    
    logger.info("5-MINUTE TIMEFRAME VECTORS:")
    logger.info("  Price Vector:")
    logger.info(f"    Displacement: {tf5_price.displacement:.5f}")
    logger.info(f"    Magnitude: {tf5_price.magnitude:.5f}")
    logger.info(f"    Direction: {tf5_price.direction:.3f}")
    logger.info(f"    Price range: {tf5_price.price_range:.5f}")
    
    logger.info("  Momentum Vector:")
    logger.info(f"    Price momentum: {tf5_momentum.price_momentum:.5f}")
    logger.info(f"    Volatility (ATR): {tf5_momentum.volatility:.5f}")
    logger.info(f"    Magnitude: {tf5_momentum.magnitude:.5f}")
    logger.info(f"    Direction: {tf5_momentum.direction:.3f}")
    
    # Calculate 15-minute vectors
    tf15_price = VectorCalculations.calculate_price_vector(tf15_data, config.vector_period)
    tf15_momentum = VectorCalculations.calculate_momentum_vector(tf15_data, config.vector_period)
    
    logger.info("\n15-MINUTE TIMEFRAME VECTORS:")
    logger.info("  Price Vector:")
    logger.info(f"    Displacement: {tf15_price.displacement:.5f}")
    logger.info(f"    Magnitude: {tf15_price.magnitude:.5f}")
    logger.info(f"    Direction: {tf15_price.direction:.3f}")
    
    logger.info("  Momentum Vector:")
    logger.info(f"    Price momentum: {tf15_momentum.price_momentum:.5f}")
    logger.info(f"    Volatility (ATR): {tf15_momentum.volatility:.5f}")
    logger.info(f"    Magnitude: {tf15_momentum.magnitude:.5f}")
    logger.info(f"    Direction: {tf15_momentum.direction:.3f}")
    
    # Combine vectors
    combined = VectorCalculations.combine_vectors(
        tf5_price, tf5_momentum, tf15_price, tf15_momentum,
        config.tf5_weight, config.tf15_weight,
        config.tf5_direction_weight, config.tf15_direction_weight
    )
    
    logger.info("\nCOMBINED MULTI-TIMEFRAME VECTOR:")
    logger.info(f"  Combined magnitude: {combined.combined_magnitude:.5f}")
    logger.info(f"  Combined direction: {combined.combined_direction:.3f}")
    logger.info(f"  Magnitude weights: 5m={config.tf5_weight}, 15m={config.tf15_weight}")
    logger.info(f"  Direction weights: 5m={config.tf5_direction_weight}, 15m={config.tf15_direction_weight}")


async def demonstrate_signal_generation(
    tf5_data: List[OHLCVData],
    tf15_data: List[OHLCVData],
    config: StrategyConfig,
    data_service: DataService
) -> None:
    """
    Demonstrate signal generation process.
    
    Args:
        tf5_data: 5-minute OHLCV data
        tf15_data: 15-minute OHLCV data
        config: Strategy configuration
        data_service: Data service for getting current price
    """
    logger.info("=" * 60)
    logger.info("SIGNAL GENERATION DEMONSTRATION")
    logger.info("=" * 60)
    
    # Create signal generator
    signal_generator = SignalGenerator(config)
    
    # Build some historical magnitude data for percentile calculation
    logger.info("Building historical magnitude data for percentile calculation...")
    for i in range(20):
        # Use sliding windows to build historical data
        if len(tf5_data) > i + config.vector_period:
            end_idx = -i if i > 0 else None
            window_5m = tf5_data[-(config.vector_period + i):end_idx]
            window_15m = tf15_data[-(config.vector_period + i):end_idx]
            if len(window_5m) >= config.vector_period and len(window_15m) >= config.vector_period:
                try:
                    # Get current price (latest close)
                    current_price = await data_service.get_latest_price()
                    signal = signal_generator.generate_signal(window_5m, window_15m, current_price)
                except Exception as e:
                    logger.warning(f"Error building historical data: {e}")
                    # Use latest close from data as fallback
                    current_price = tf5_data[-1].close
                    signal = signal_generator.generate_signal(window_5m, window_15m, current_price)
    
    logger.info(f"Built {len(signal_generator.historical_magnitudes)} historical magnitude values")
    
    # Generate final signal with current data
    try:
        current_price = await data_service.get_latest_price()
    except Exception:
        current_price = tf5_data[-1].close
        logger.warning(f"Using latest close price from data: {current_price}")
    
    logger.info(f"Generating signal with current price: {current_price}")
    
    signal = signal_generator.generate_signal(tf5_data, tf15_data, current_price)
    
    # Display signal results
    logger.info("\nSIGNAL ANALYSIS RESULTS:")
    logger.info(f"  Signal Type: {signal.signal_type.value}")
    logger.info(f"  Confidence: {signal.confidence:.2f}")
    logger.info(f"  Timestamp: {signal.datetime}")
    logger.info(f"  Reason: {signal.reason}")
    
    if signal.vector_data:
        logger.info("\nVECTOR DATA:")
        logger.info(f"  Combined magnitude: {signal.vector_data.combined_magnitude:.5f}")
        logger.info(f"  Combined direction: {signal.vector_data.combined_direction:.3f}")
        logger.info(f"  Signal strength: {signal.vector_data.signal_strength:.1f}th percentile")
        logger.info(f"  5m direction: {signal.vector_data.tf5_direction:.3f}")
        logger.info(f"  15m direction: {signal.vector_data.tf15_direction:.3f}")
    
    if signal.divergence_data:
        logger.info("\nDIVERGENCE ANALYSIS:")
        logger.info(f"  Price trend: {signal.divergence_data.price_trend:.4f}")
        logger.info(f"  Momentum trend: {signal.divergence_data.momentum_trend:.4f}")
        logger.info(f"  Bullish divergence: {signal.divergence_data.is_bullish_divergence}")
        logger.info(f"  Bearish divergence: {signal.divergence_data.is_bearish_divergence}")
        logger.info(f"  Divergence strength: {signal.divergence_data.divergence_strength:.3f}")
    
    if signal.signal_type != SignalType.NO_SIGNAL:
        logger.info("\nTRADE SETUP:")
        logger.info(f"  Entry price: {signal.entry_price:.5f}")
        logger.info(f"  Take profit: {signal.take_profit:.5f}")
        logger.info(f"  Stop loss: {signal.stop_loss:.5f}")
        
        # Calculate pip values
        if signal.entry_price and signal.take_profit and signal.stop_loss:
            if signal.signal_type == SignalType.LONG:
                tp_pips = (signal.take_profit - signal.entry_price) / config.risk_management.pip_size
                sl_pips = (signal.entry_price - signal.stop_loss) / config.risk_management.pip_size
            else:
                tp_pips = (signal.entry_price - signal.take_profit) / config.risk_management.pip_size
                sl_pips = (signal.stop_loss - signal.entry_price) / config.risk_management.pip_size
            
            logger.info(f"  Take profit: {tp_pips:.0f} pips")
            logger.info(f"  Stop loss: {sl_pips:.0f} pips")
            logger.info(f"  Risk/Reward ratio: 1:{tp_pips/sl_pips:.2f}")
    
    # Check time-based exit
    should_exit = signal_generator.should_exit_time_based()
    logger.info("\nTIME-BASED EXIT CHECK:")
    logger.info(f"  Should exit (Friday 5 PM GMT): {should_exit}")


def display_data_summary(
    tf5_data: List[OHLCVData],
    tf15_data: List[OHLCVData]
) -> None:
    """
    Display summary of fetched data.
    
    Args:
        tf5_data: 5-minute OHLCV data
        tf15_data: 15-minute OHLCV data
    """
    logger.info("=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)
    
    if tf5_data:
        logger.info(f"5-MINUTE DATA ({len(tf5_data)} bars):")
        logger.info(f"  Time range: {datetime.fromtimestamp(tf5_data[0].timestamp)} to {datetime.fromtimestamp(tf5_data[-1].timestamp)}")
        logger.info(f"  Price range: {min(bar.low for bar in tf5_data):.5f} - {max(bar.high for bar in tf5_data):.5f}")
        logger.info(f"  Latest: O={tf5_data[-1].open:.5f} H={tf5_data[-1].high:.5f} L={tf5_data[-1].low:.5f} C={tf5_data[-1].close:.5f}")
        logger.info(f"  Average volume: {sum(bar.volume for bar in tf5_data) / len(tf5_data):.0f}")
    
    if tf15_data:
        logger.info(f"\n15-MINUTE DATA ({len(tf15_data)} bars):")
        logger.info(f"  Time range: {datetime.fromtimestamp(tf15_data[0].timestamp)} to {datetime.fromtimestamp(tf15_data[-1].timestamp)}")
        logger.info(f"  Price range: {min(bar.low for bar in tf15_data):.5f} - {max(bar.high for bar in tf15_data):.5f}")
        logger.info(f"  Latest: O={tf15_data[-1].open:.5f} H={tf15_data[-1].high:.5f} L={tf15_data[-1].low:.5f} C={tf15_data[-1].close:.5f}")
        logger.info(f"  Average volume: {sum(bar.volume for bar in tf15_data) / len(tf15_data):.0f}")


async def main():
    """
    Main demonstration function.
    """
    logger.info("🚀 Starting Vector Scalping Strategy Demo")
    logger.info("=" * 60)
    
    try:
        # 1. Create strategy configuration
        config = await create_strategy_config()
        
        # 2. Initialize data service and fetch market data
        async with DataService(config) as data_service:
            
            # Validate symbol format
            if not data_service.validate_symbol_format(config.symbol):
                raise ValueError(f"Invalid symbol format: {config.symbol}")
            
            # Fetch multi-timeframe data
            market_data = await fetch_market_data(data_service, bars_count=50)
            tf5_data = market_data[TimeFrame.MIN_5]
            tf15_data = market_data[TimeFrame.MIN_15]
            
            # 3. Display data summary
            display_data_summary(tf5_data, tf15_data)
            
            # 4. Analyze data with Polars
            df_5m = analyze_data_with_polars(data_service, tf5_data)
            
            # Display latest technical indicators
            if not df_5m.is_empty():
                latest_indicators = df_5m.select([
                    "timestamp", "close", "sma_5", "sma_20", "atr_5", "volatility_5"
                ]).tail(1).row(0, named=True)
                
                logger.info("\nLATEST TECHNICAL INDICATORS:")
                logger.info(f"  Close: {latest_indicators['close']:.5f}")
                sma_5_val = latest_indicators['sma_5']
                sma_20_val = latest_indicators['sma_20']
                atr_5_val = latest_indicators['atr_5']
                vol_5_val = latest_indicators['volatility_5']
                
                logger.info(f"  SMA(5): {f'{sma_5_val:.5f}' if sma_5_val is not None else 'N/A'}")
                logger.info(f"  SMA(20): {f'{sma_20_val:.5f}' if sma_20_val is not None else 'N/A'}")
                logger.info(f"  ATR(5): {f'{atr_5_val:.5f}' if atr_5_val is not None else 'N/A'}")
                logger.info(f"  Volatility(5): {f'{vol_5_val:.5f}' if vol_5_val is not None else 'N/A'}")
            
            # 5. Demonstrate vector calculations
            demonstrate_vector_calculations(tf5_data, tf15_data, config)
            
            # 6. Demonstrate signal generation
            await demonstrate_signal_generation(tf5_data, tf15_data, config, data_service)
        
        logger.info("=" * 60)
        logger.info("✅ Vector Scalping Strategy Demo Completed Successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())