#!/usr/bin/env python3
"""
Custom data backtesting example.

This example demonstrates:
1. Loading custom OHLCV data from CSV
2. Data validation and preprocessing  
3. Running backtests with real market data
4. Advanced result analysis
"""

import asyncio
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtest.vector_scalping_vectorbt import VectorBacktester, BacktestConfig


def create_sample_data(symbol: str = "EURUSD", n_bars: int = 2000) -> pd.DataFrame:
    """
    Create realistic sample OHLCV data for demonstration.
    
    In practice, replace this with your actual data loading logic:
    df = pd.read_csv('your_data.csv', parse_dates=['datetime'])
    """
    
    print(f"📊 Generating sample {symbol} data ({n_bars} bars)...")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Base parameters for realistic forex data
    if symbol.startswith("JPY") or symbol.endswith("JPY"):
        initial_price = 150.0  # USD/JPY style
        volatility = 0.5
        pip_size = 0.01
    else:
        initial_price = 1.0850  # EUR/USD style  
        volatility = 0.0005
        pip_size = 0.0001
    
    # Generate realistic price series with trends and volatility clustering
    returns = []
    price = initial_price
    
    for i in range(n_bars):
        # Add trend component (cycles every 500 bars)
        trend = volatility * 0.2 * np.sin(2 * np.pi * i / 500)
        
        # Add random walk with GARCH-like volatility clustering
        if i > 10:
            recent_vol = np.std(returns[-10:])
            volatility_multiplier = 1 + 0.5 * recent_vol / volatility
        else:
            volatility_multiplier = 1
            
        random_return = np.random.normal(0, volatility * volatility_multiplier)
        
        # Mean reversion component
        deviation = (price - initial_price) / initial_price
        mean_reversion = -0.05 * deviation * volatility
        
        # News/event spikes (rare but large moves)
        if np.random.random() < 0.005:  # 0.5% chance
            spike = np.random.choice([-1, 1]) * volatility * 5
            random_return += spike
        
        # Combine components
        total_return = trend + random_return + mean_reversion
        returns.append(total_return)
        price += total_return
    
    # Create OHLCV bars from price series
    prices = np.array([initial_price + np.sum(returns[:i+1]) for i in range(n_bars)])
    
    data = []
    for i, close in enumerate(prices):
        # Realistic intrabar price action
        bar_volatility = volatility * np.random.uniform(0.3, 2.0)
        
        # Open price (previous close for continuous series)
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] 
        
        # High and low with realistic distribution
        high_offset = np.random.exponential(bar_volatility * 0.7)
        low_offset = np.random.exponential(bar_volatility * 0.7)
        
        high = max(open_price, close) + high_offset
        low = min(open_price, close) - low_offset
        
        # Volume correlated with volatility and time of day
        base_volume = np.random.uniform(1000, 3000)
        volatility_factor = 1 + abs(returns[i]) / volatility
        time_factor = 1 + 0.5 * np.sin(2 * np.pi * (i % 288) / 288)  # Daily pattern
        volume = base_volume * volatility_factor * time_factor
        
        data.append({
            'open': round(open_price, 5 if pip_size == 0.0001 else 3),
            'high': round(high, 5 if pip_size == 0.0001 else 3),
            'low': round(low, 5 if pip_size == 0.0001 else 3),
            'close': round(close, 5 if pip_size == 0.0001 else 3),
            'volume': round(volume, 0)
        })
    
    # Create DataFrame with 5-minute intervals
    start_time = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
    dates = pd.date_range(start=start_time, periods=n_bars, freq='5T')
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'datetime'
    
    print(f"✅ Generated {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Validate OHLCV data quality."""
    
    print("🔍 Validating data quality...")
    
    issues = []
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for NaN values
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        issues.append(f"NaN values in columns: {nan_cols}")
    
    # Check OHLC logic
    ohlc_issues = (
        (df['high'] < df['open']) | 
        (df['high'] < df['close']) |
        (df['low'] > df['open']) | 
        (df['low'] > df['close']) |
        (df['high'] < df['low'])
    ).sum()
    
    if ohlc_issues > 0:
        issues.append(f"OHLC logic violations: {ohlc_issues} bars")
    
    # Check for negative volumes
    negative_volume = (df['volume'] < 0).sum()
    if negative_volume > 0:
        issues.append(f"Negative volumes: {negative_volume} bars")
    
    # Check for extreme price gaps (>5% moves)
    price_changes = df['close'].pct_change().abs()
    extreme_moves = (price_changes > 0.05).sum()
    if extreme_moves > len(df) * 0.01:  # More than 1% of bars
        issues.append(f"Excessive extreme moves: {extreme_moves} (>{len(df)*0.01:.0f})")
    
    if issues:
        print("❌ Data validation issues found:")
        for issue in issues:
            print(f"  • {issue}")
        return False
    else:
        print("✅ Data validation passed")
        return True


def analyze_data_characteristics(df: pd.DataFrame) -> None:
    """Analyze data characteristics for strategy optimization."""
    
    print("\n📈 Data Characteristics:")
    
    # Basic statistics
    print(f"  📊 Total bars: {len(df):,}")
    print(f"  📅 Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  💰 Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
    
    # Volatility analysis
    returns = df['close'].pct_change().dropna()
    daily_vol = returns.std() * np.sqrt(288)  # 288 5-min bars per day
    print(f"  📊 Daily volatility: {daily_vol*100:.2f}%")
    
    # Trend analysis
    overall_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"  📈 Overall return: {overall_return:+.2f}%")
    
    # Volume analysis
    avg_volume = df['volume'].mean()
    volume_cv = df['volume'].std() / avg_volume
    print(f"  📊 Average volume: {avg_volume:,.0f} (CV: {volume_cv:.2f})")
    
    # Market regime analysis
    upward_moves = (returns > 0).mean() * 100
    print(f"  ⬆️  Upward moves: {upward_moves:.1f}%")
    
    # Optimal parameters suggestion
    print("\n💡 Suggested Parameters:")
    if daily_vol > 0.015:  # High volatility
        print("  🌊 High volatility detected - consider conservative settings")
        print("     percentile_threshold=70, take_profit_pips=15")
    elif daily_vol < 0.008:  # Low volatility  
        print("  😴 Low volatility detected - consider aggressive settings")
        print("     percentile_threshold=50, take_profit_pips=25")
    else:
        print("  ⚖️  Normal volatility - default settings should work well")


async def run_custom_data_backtest(df: pd.DataFrame, symbol: str = "EURUSD"):
    """Run backtest with custom data."""
    
    print(f"\n🚀 Running backtest for {symbol}")
    print("="*50)
    
    # Determine pip size based on symbol
    if symbol.endswith("JPY"):
        pip_size = 0.01
    else:
        pip_size = 0.0001
    
    # Configure backtest
    config = BacktestConfig(
        symbol=symbol,
        initial_cash=50000.0,           # Larger account for more realistic results
        pip_value=10.0,                 # $10 per pip position sizing
        take_profit_pips=20,
        stop_loss_pips=30,
        pip_size=pip_size,
        vector_period=5,
        percentile_threshold=60.0,
        direction_threshold=0.0005,
        lookback_period=100
    )
    
    # Run backtest
    backtester = VectorBacktester(config)
    results = await backtester.run_backtest(data=df)
    
    # Additional analysis specific to custom data
    print("\n🔍 Custom Data Analysis:")
    
    # Trade distribution
    if 'trade_details' in results:
        long_trades = results['trade_details']['long_trades']
        short_trades = results['trade_details']['short_trades']
        
        if long_trades:
            avg_long_duration = np.mean([t['duration_minutes'] for t in long_trades])
            print(f"  ⏱️  Average long trade duration: {avg_long_duration:.1f} minutes")
            
            long_tp_rate = len([t for t in long_trades if t['exit_reason'] == 'take_profit']) / len(long_trades)
            print(f"  🎯 Long take-profit rate: {long_tp_rate*100:.1f}%")
        
        if short_trades:
            avg_short_duration = np.mean([t['duration_minutes'] for t in short_trades])
            print(f"  ⏱️  Average short trade duration: {avg_short_duration:.1f} minutes")
            
            short_tp_rate = len([t for t in short_trades if t['exit_reason'] == 'take_profit']) / len(short_trades)
            print(f"  🎯 Short take-profit rate: {short_tp_rate*100:.1f}%")
        
        # Exit reason analysis
        all_trades = long_trades + short_trades
        if all_trades:
            exit_reasons = {}
            for trade in all_trades:
                reason = trade['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            print("  📊 Exit reasons:")
            for reason, count in exit_reasons.items():
                pct = count / len(all_trades) * 100
                print(f"     {reason}: {count} ({pct:.1f}%)")
    
    return results


async def main():
    """Main execution function demonstrating custom data backtesting."""
    
    print("🧪 Custom Data Backtesting Example")
    print("="*50)
    
    # Create sample data (replace with your data loading logic)
    eurusd_data = create_sample_data("EURUSD", n_bars=2000)
    
    # Validate data quality
    if not validate_data(eurusd_data):
        print("❌ Data validation failed. Please fix data issues before proceeding.")
        return
    
    # Analyze data characteristics
    analyze_data_characteristics(eurusd_data)
    
    # Run backtest with custom data
    await run_custom_data_backtest(eurusd_data, "EURUSD")
    
    print("\n" + "="*50)
    print("💡 Next Steps:")
    print("="*50)
    print("1. Replace create_sample_data() with your actual data loading")
    print("2. Experiment with different BacktestConfig parameters")
    print("3. Try multiple symbols and timeframes")
    print("4. Implement walk-forward analysis for robust validation")
    print("5. Add visualization of equity curves and drawdowns")
    
    print("\n✅ Custom data backtest completed!")


if __name__ == "__main__":
    asyncio.run(main())