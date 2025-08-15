#!/usr/bin/env python3
"""
Basic backtesting example using the vector scalping strategy.

This example demonstrates:
1. Basic configuration setup
2. Running a backtest with sample data
3. Analyzing results
"""

import asyncio
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtest.vector_scalping_vectorbt import VectorBacktester, BacktestConfig


async def run_basic_backtest():
    """Run a basic backtest with default settings."""
    
    print("🚀 Basic Vector Scalping Backtest Example")
    print("=" * 50)
    
    # Configure the backtest with default EUR/USD settings
    config = BacktestConfig(
        symbol="EURUSD",
        initial_cash=10000.0,           # Start with $10,000
        pip_value=1.0,                  # $1 per pip
        take_profit_pips=20,            # 20-pip target
        stop_loss_pips=30,              # 30-pip stop
        pip_size=0.0001,                # 4-decimal pair
        vector_period=5,                # 5-candle vectors
        percentile_threshold=60.0,      # 60th percentile strength
        direction_threshold=0.0005,     # Minimum direction
        lookback_period=100             # Rolling window
    )
    
    # Create backtester and run with sample data
    backtester = VectorBacktester(config)
    results = await backtester.run_backtest()
    
    # Additional analysis
    print("\n🔍 Additional Analysis:")
    perf = results['performance_metrics']
    
    # Calculate some additional metrics
    total_trades = perf['combined_total_trades']
    if total_trades > 0:
        avg_return_per_trade = perf['combined_return'] / total_trades * 100
        print(f"  📊 Average return per trade: {avg_return_per_trade:.3f}%")
        
        # Risk-reward ratio
        winning_avg = perf['avg_trade_return_long'] if perf['avg_trade_return_long'] > 0 else 0
        losing_avg = abs(perf['avg_trade_return_short']) if perf['avg_trade_return_short'] < 0 else 0
        
        if losing_avg > 0:
            risk_reward = winning_avg / losing_avg
            print(f"  ⚖️  Risk-reward ratio: {risk_reward:.2f}:1")
    
    # Signal quality analysis
    vector = results['vector_analysis']
    print(f"  🎯 Signal quality score: {vector['strong_signals_pct']:.1f}% above threshold")
    
    return results


async def run_conservative_backtest():
    """Run a more conservative backtest with stricter parameters."""
    
    print("\n" + "="*50)
    print("🛡️  Conservative Strategy Backtest")
    print("="*50)
    
    # Conservative configuration
    config = BacktestConfig(
        symbol="EURUSD",
        initial_cash=10000.0,
        pip_value=1.0,
        take_profit_pips=25,            # Larger targets
        stop_loss_pips=25,              # Tighter stops
        pip_size=0.0001,
        vector_period=7,                # Longer period for stability
        percentile_threshold=75.0,      # Higher threshold (fewer signals)
        direction_threshold=0.001,      # Less sensitive
        lookback_period=150             # Longer history
    )
    
    backtester = VectorBacktester(config)
    results = await backtester.run_backtest()
    
    return results


async def run_aggressive_backtest():
    """Run an aggressive backtest with more frequent signals."""
    
    print("\n" + "="*50)
    print("⚡ Aggressive Strategy Backtest")
    print("="*50)
    
    # Aggressive configuration
    config = BacktestConfig(
        symbol="EURUSD", 
        initial_cash=10000.0,
        pip_value=1.0,
        take_profit_pips=15,            # Smaller targets
        stop_loss_pips=20,              # Tighter stops
        pip_size=0.0001,
        vector_period=3,                # Shorter period for speed
        percentile_threshold=45.0,      # Lower threshold (more signals)
        direction_threshold=0.0003,     # More sensitive
        lookback_period=50              # Shorter history
    )
    
    backtester = VectorBacktester(config)
    results = await backtester.run_backtest()
    
    return results


def compare_strategies(basic_results, conservative_results, aggressive_results):
    """Compare the results of different strategy configurations."""
    
    print("\n" + "="*60)
    print("📊 STRATEGY COMPARISON")
    print("="*60)
    
    strategies = [
        ("Basic", basic_results),
        ("Conservative", conservative_results), 
        ("Aggressive", aggressive_results)
    ]
    
    print(f"{'Strategy':<12} {'Return':<8} {'Win Rate':<8} {'Trades':<7} {'Sharpe':<7}")
    print("-" * 50)
    
    for name, results in strategies:
        perf = results['performance_metrics']
        return_pct = perf['combined_return'] * 100
        win_rate = perf['combined_win_rate'] * 100
        trades = perf['combined_total_trades']
        sharpe = (perf['sharpe_ratio_long'] + perf['sharpe_ratio_short']) / 2
        
        print(f"{name:<12} {return_pct:>6.2f}% {win_rate:>6.1f}% {trades:>6d} {sharpe:>6.2f}")
    
    print("\n🏆 Best Performance Summary:")
    
    # Find best performing strategies
    returns = [(name, results['performance_metrics']['combined_return']) for name, results in strategies]
    best_return = max(returns, key=lambda x: x[1])
    print(f"  📈 Highest Return: {best_return[0]} ({best_return[1]*100:.2f}%)")
    
    sharpes = [(name, (results['performance_metrics']['sharpe_ratio_long'] + 
                      results['performance_metrics']['sharpe_ratio_short'])/2) for name, results in strategies]
    best_sharpe = max(sharpes, key=lambda x: x[1])  
    print(f"  🎯 Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1]:.2f})")
    
    win_rates = [(name, results['performance_metrics']['combined_win_rate']) for name, results in strategies]
    best_win_rate = max(win_rates, key=lambda x: x[1])
    print(f"  🎲 Highest Win Rate: {best_win_rate[0]} ({best_win_rate[1]*100:.1f}%)")


async def main():
    """Main execution function."""
    
    print("🧪 Vector Scalping Strategy Comparison")
    print("Testing multiple configurations with sample data")
    print("="*60)
    
    # Run all three backtests
    basic_results = await run_basic_backtest()
    conservative_results = await run_conservative_backtest()
    aggressive_results = await run_aggressive_backtest()
    
    # Compare results
    compare_strategies(basic_results, conservative_results, aggressive_results)
    
    print("\n✅ All backtests completed successfully!")
    print("\n💡 Tip: Try running with your own data using:")
    print("   df = pd.read_csv('your_data.csv')")
    print("   results = await backtester.run_backtest(data=df)")


if __name__ == "__main__":
    asyncio.run(main())