#!/usr/bin/env python3
"""
Vector-based scalping strategy backtesting framework.

USAGE INSTRUCTIONS:
1. Run the script: uv run python backtest/vector_scalping_vectorbt.py
2. Or run as module: uv run python -m backtest.vector_scalping_vectorbt

DEFAULT SYMBOL: TFEX:S50U2025 (fetches real data from TradingView using tvkit)

NOTE: This implementation uses pandas/numpy for backtesting instead of vectorbt
due to compatibility issues. For vectorbt support, install compatible versions:
uv pip install "numpy<2.1" "numba<0.61" vectorbt

DATA SOURCE:
- Uses tvkit library to fetch real 5-minute OHLCV data from TradingView
- Automatically resamples to 15-minute timeframe for multi-timeframe analysis
- Falls back to synthetic data generation if real data fetching fails

STRATEGY OVERVIEW:
- Calculates price and momentum vectors for 5-min and 15-min timeframes
- Entry when both vectors align and strength > 60th percentile
- 20-pip take profit, time-based stop loss (Friday 5 PM)
- Filters weak signals using rolling quantile analysis

MATHEMATICAL FOUNDATION:
This implementation follows the corrected vector formulas from CLAUDE.md:
- Price Vector: √(Δ₁² + Δ₂² + Δ₃² + Δ₄²) with Direction = (Δ₁ + Δ₂ + Δ₃ + Δ₄) ÷ 4
- Momentum Vector: √[(PM² + V²)] with proper momentum and volatility components
- Multi-timeframe combination with weighted averages (0.7/0.3 for magnitude, 0.6/0.4 for direction)
"""

import asyncio
import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl
from pydantic import BaseModel, Field, field_validator

# Import existing models and calculations
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tvkit.api.chart.models.ohlcv import OHLCVBar
from tvkit.api.chart.ohlcv import OHLCV
from tvkit.export import DataExporter
from vector_scalping.calculations import VectorCalculations
from vector_scalping.models import (
    OHLCVData,
    RiskManagement,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BacktestConfig(BaseModel):
    """Configuration for vectorbt backtesting."""

    symbol: str = Field("TFEX:S50U2025", description="Trading symbol")
    initial_cash: float = Field(10000.0, description="Initial capital")
    pip_value: float = Field(1.0, description="Value per pip for position sizing")
    take_profit_pips: int = Field(20, description="Take profit in pips")
    stop_loss_pips: int = Field(30, description="Stop loss in pips")
    pip_size: float = Field(
        0.01, description="Pip size (0.01 for 2-decimal pairs like futures)"
    )
    vector_period: int = Field(5, description="Vector calculation period")
    percentile_threshold: float = Field(60.0, description="Signal strength threshold")
    direction_threshold: float = Field(
        0.0005, description="Minimum direction threshold"
    )
    tf5_weight: float = Field(0.7, description="5-minute timeframe weight")
    tf15_weight: float = Field(0.3, description="15-minute timeframe weight")
    tf5_dir_weight: float = Field(0.6, description="5-minute direction weight")
    tf15_dir_weight: float = Field(0.4, description="15-minute direction weight")
    lookback_period: int = Field(100, description="Percentile calculation lookback")

    @field_validator("pip_size")
    @classmethod
    def validate_pip_size(cls, v: float) -> float:
        """Validate pip size is positive."""
        if v <= 0:
            raise ValueError("Pip size must be positive")
        return v


class VectorBacktester:
    """Vector scalping strategy backtester using vectorbt."""

    def __init__(self, config: BacktestConfig):
        """Initialize backtester with configuration."""
        self.config = config
        self.risk_mgmt = RiskManagement(
            symbol=config.symbol,
            pip_size=config.pip_size,
            take_profit_pips=config.take_profit_pips,
            stop_loss_pips=config.stop_loss_pips,
            is_decimal_4=(config.pip_size == 0.0001),
        )

    async def fetch_real_data(self, symbol: str, n_bars: int = 1000) -> pd.DataFrame:
        """
        Fetch real OHLCV data from TradingView using tvkit.

        Args:
            symbol: Trading symbol (e.g., "TFEX:S50U2025")
            n_bars: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data and datetime index

        Raises:
            ValueError: If unable to fetch sufficient data
        """
        try:
            bars_5min: List[OHLCVBar] = []

            async with OHLCV() as client:
                bars_5min = await client.get_historical_ohlcv(
                    symbol, interval="5", bars_count=n_bars
                )

            if len(bars_5min) < 100:  # Minimum viable data
                raise ValueError(
                    f"Insufficient data: only {len(bars_5min)} bars fetched"
                )

            # Convert to pandas DataFrame
            exporter = DataExporter()
            df_polars: pl.DataFrame = await exporter.to_polars(
                bars_5min, add_analysis=False
            )

            # Convert polars to pandas with proper datetime index
            df_pandas = df_polars.to_pandas()

            # Convert timestamp to datetime index (handle different timestamp formats)
            if 'timestamp' in df_pandas.columns:
                # Try different timestamp parsing methods
                if df_pandas['timestamp'].dtype == 'object':
                    # String timestamp format (ISO 8601)
                    df_pandas['datetime'] = pd.to_datetime(df_pandas['timestamp'])
                else:
                    # Unix timestamp format
                    df_pandas['datetime'] = pd.to_datetime(df_pandas['timestamp'], unit='s')
            else:
                # If timestamp column not found, create sequential datetime index
                start_time = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
                df_pandas['datetime'] = pd.date_range(start=start_time, periods=len(df_pandas), freq="5T")
            
            df_pandas = df_pandas.set_index('datetime')

            # Keep only OHLCV columns (handle case where timestamp might still be present)
            available_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df_pandas.columns]
            df_pandas = df_pandas[available_cols]

            # Sort by datetime to ensure proper order
            df_pandas = df_pandas.sort_index()

            return df_pandas

        except Exception as e:
            raise ValueError(f"Failed to fetch real data for {symbol}: {e}") from e

    def generate_sample_data(self, n_bars: int = 1000) -> pd.DataFrame:
        """
        Generate sample data for backtesting (fallback method).
        This method is kept for backward compatibility and testing.

        Args:
            n_bars: Number of bars to generate

        Returns:
            DataFrame with OHLCV data and datetime index
        """
        # Create realistic price series for futures-style data
        np.random.seed(42)  # For reproducible results

        # Base parameters adapted for futures (higher prices, different volatility)
        initial_price = 2650.0  # Typical S&P 500 futures price level
        trend_strength = 5.0  # Larger moves for futures
        volatility = 15.0  # Higher volatility

        # Generate price series with trend and mean reversion
        returns = []
        price = initial_price

        for i in range(n_bars):
            # Add trend component (cycles every 200 bars)
            trend = trend_strength * np.sin(2 * np.pi * i / 200)

            # Add random component with volatility clustering
            random_return = np.random.normal(0, volatility)

            # Mean reversion component
            mean_reversion = (
                -0.01 * (price - initial_price) * volatility / initial_price
            )

            # Combine components
            total_return = trend + random_return + mean_reversion
            returns.append(total_return)
            price += total_return

        # Create OHLCV data
        prices = np.array(
            [initial_price + np.sum(returns[: i + 1]) for i in range(n_bars)]
        )

        # Generate realistic OHLC from close prices
        data = []
        for i, close in enumerate(prices):
            # Realistic intrabar volatility
            bar_volatility = volatility * np.random.uniform(0.5, 2.0)

            high = close + np.random.uniform(0, bar_volatility)
            low = close - np.random.uniform(0, bar_volatility)

            # Ensure OHLC logic
            if i == 0:
                open_price = close
            else:
                open_price = prices[i - 1]  # Open equals previous close

            high = max(open_price, high, close)
            low = min(open_price, low, close)

            # Generate volume (appropriate for futures)
            volume = np.random.uniform(10000, 50000) * (
                1 + abs(returns[i]) / volatility
            )

            data.append(
                {
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        # Create DataFrame with proper datetime index (5-minute bars)
        start_time = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
        dates = pd.date_range(start=start_time, periods=n_bars, freq="5T")

        df = pd.DataFrame(data, index=dates)
        df.index.name = "datetime"

        return df

    def resample_to_15min(self, df_5min: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 5-minute data to 15-minute timeframe.

        Args:
            df_5min: 5-minute OHLCV DataFrame

        Returns:
            15-minute OHLCV DataFrame
        """
        df_15min = (
            df_5min.resample("15T")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        return df_15min

    def calculate_vectors(
        self, df_5min: pd.DataFrame, df_15min: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate vector signals for backtesting.

        Args:
            df_5min: 5-minute OHLCV data
            df_15min: 15-minute OHLCV data

        Returns:
            DataFrame with vector signals and entry/exit conditions
        """
        signals = []
        historical_magnitudes: List[float] = []

        # Align 15-min data with 5-min data for proper timeframe combination
        df_15min_reindexed = df_15min.reindex(df_5min.index, method="ffill")

        for i in range(self.config.vector_period, len(df_5min)):
            try:
                # Get 5-minute data window
                tf5_window = df_5min.iloc[i - self.config.vector_period + 1 : i + 1]
                tf5_data = [
                    OHLCVData(
                        timestamp=int(pd.Timestamp(row.name).timestamp()),  # type: ignore[arg-type]
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                    )
                    for _, row in tf5_window.iterrows()
                ]

                # Get aligned 15-minute data window
                tf15_window = df_15min_reindexed.iloc[
                    i - self.config.vector_period + 1 : i + 1
                ]
                tf15_data = [
                    OHLCVData(
                        timestamp=int(pd.Timestamp(row.name).timestamp()),  # type: ignore[arg-type]
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row["volume"],
                    )
                    for _, row in tf15_window.iterrows()
                    if not pd.isna(row["close"])
                ]

                # Skip if insufficient 15-min data
                if len(tf15_data) < self.config.vector_period:
                    signals.append(
                        {
                            "datetime": df_5min.index[i],
                            "signal": 0,  # No signal
                            "vector_strength": 0.0,
                            "combined_direction": 0.0,
                            "tf5_direction": 0.0,
                            "tf15_direction": 0.0,
                        }
                    )
                    continue

                # Calculate vectors using existing calculation methods
                tf5_price = VectorCalculations.calculate_price_vector(
                    tf5_data, self.config.vector_period
                )
                tf5_momentum = VectorCalculations.calculate_momentum_vector(
                    tf5_data, self.config.vector_period
                )
                tf15_price = VectorCalculations.calculate_price_vector(
                    tf15_data, self.config.vector_period
                )
                tf15_momentum = VectorCalculations.calculate_momentum_vector(
                    tf15_data, self.config.vector_period
                )

                # Combine vectors
                combined = VectorCalculations.combine_vectors(
                    tf5_price,
                    tf5_momentum,
                    tf15_price,
                    tf15_momentum,
                    self.config.tf5_weight,
                    self.config.tf15_weight,
                    self.config.tf5_dir_weight,
                    self.config.tf15_dir_weight,
                )

                # Calculate percentile strength using lookback
                if len(historical_magnitudes) >= self.config.lookback_period:
                    lookback_data = historical_magnitudes[
                        -self.config.lookback_period :
                    ]
                else:
                    lookback_data = historical_magnitudes.copy()

                if lookback_data:
                    vector_strength = VectorCalculations.calculate_percentile_rank(
                        combined.combined_magnitude, lookback_data
                    )
                else:
                    vector_strength = 50.0  # Default middle percentile

                # Store magnitude for future percentile calculations
                historical_magnitudes.append(combined.combined_magnitude)

                # Generate signals based on strategy rules
                signal = self._generate_signal(
                    combined.combined_direction,
                    vector_strength,
                    tf5_momentum.direction,
                    tf15_momentum.direction,
                )

                signals.append(
                    {
                        "datetime": df_5min.index[i],
                        "signal": signal,
                        "vector_strength": vector_strength,
                        "combined_direction": combined.combined_direction,
                        "tf5_direction": tf5_momentum.direction,
                        "tf15_direction": tf15_momentum.direction,
                        "combined_magnitude": combined.combined_magnitude,
                        "close": df_5min.iloc[i]["close"],
                    }
                )

            except Exception:
                # Handle any calculation errors gracefully
                signals.append(
                    {
                        "datetime": df_5min.index[i],
                        "signal": 0,
                        "vector_strength": 0.0,
                        "combined_direction": 0.0,
                        "tf5_direction": 0.0,
                        "tf15_direction": 0.0,
                    }
                )

        return pd.DataFrame(signals).set_index("datetime")

    def _generate_signal(
        self,
        combined_direction: float,
        vector_strength: float,
        tf5_direction: float,
        tf15_direction: float,
    ) -> int:
        """
        Generate trading signal based on vector conditions.

        Args:
            combined_direction: Combined vector direction
            vector_strength: Vector strength percentile
            tf5_direction: 5-minute momentum direction
            tf15_direction: 15-minute momentum direction

        Returns:
            Signal: 1 for LONG, -1 for SHORT, 0 for NO_SIGNAL
        """
        # LONG entry conditions
        long_conditions = (
            combined_direction > self.config.direction_threshold
            and vector_strength > self.config.percentile_threshold
            and tf5_direction > (self.config.direction_threshold * 0.4)  # 0.2 adjusted
            and tf15_direction > (self.config.direction_threshold * 0.2)  # 0.1 adjusted
        )

        # SHORT entry conditions
        short_conditions = (
            combined_direction < -self.config.direction_threshold
            and vector_strength > self.config.percentile_threshold
            and tf5_direction < -(self.config.direction_threshold * 0.4)
            and tf15_direction < -(self.config.direction_threshold * 0.2)
        )

        if long_conditions:
            return 1
        elif short_conditions:
            return -1
        else:
            return 0

    def _is_friday_close(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is Friday 5 PM GMT for time-based exit."""
        return timestamp.weekday() == 4 and timestamp.hour == 17

    def _simulate_trades(
        self, prices: pd.Series, entry_signals: pd.Series, is_long: bool
    ) -> List[Dict[str, Any]]:
        """
        Simulate trades based on entry signals and exit conditions.

        Args:
            prices: Price series
            entry_signals: Boolean series indicating entry points
            is_long: True for long trades, False for short trades

        Returns:
            List of trade dictionaries with entry/exit details
        """
        trades = []
        in_trade = False
        entry_price = 0.0
        entry_time = None
        tp_price = 0.0
        sl_price = 0.0

        # Calculate TP/SL distances
        tp_distance = self.config.take_profit_pips * self.config.pip_size
        sl_distance = self.config.stop_loss_pips * self.config.pip_size

        for timestamp, price in prices.items():
            # Check for entry signal
            if not in_trade and entry_signals.get(timestamp, False):
                in_trade = True
                entry_price = price
                entry_time = timestamp

                # Set TP and SL levels
                if is_long:
                    tp_price = entry_price + tp_distance
                    sl_price = entry_price - sl_distance
                else:  # Short
                    tp_price = entry_price - tp_distance
                    sl_price = entry_price + sl_distance

            # Check for exit conditions
            elif in_trade:
                exit_reason = None
                exit_price = price

                # Check TP/SL conditions
                if is_long:
                    if price >= tp_price:
                        exit_reason = "take_profit"
                        exit_price = tp_price
                    elif price <= sl_price:
                        exit_reason = "stop_loss"
                        exit_price = sl_price
                else:  # Short
                    if price <= tp_price:
                        exit_reason = "take_profit"
                        exit_price = tp_price
                    elif price >= sl_price:
                        exit_reason = "stop_loss"
                        exit_price = sl_price

                # Check Friday 5 PM exit
                if not exit_reason and self._is_friday_close(pd.Timestamp(timestamp)):  # type: ignore[arg-type]
                    exit_reason = "time_exit"
                    exit_price = price

                # Record completed trade
                if exit_reason:
                    if is_long:
                        pnl_pips = (exit_price - entry_price) / self.config.pip_size
                        pnl_dollars = pnl_pips * self.config.pip_value
                    else:  # Short
                        pnl_pips = (entry_price - exit_price) / self.config.pip_size
                        pnl_dollars = pnl_pips * self.config.pip_value

                    trade = {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "is_long": is_long,
                        "pnl_pips": pnl_pips,
                        "pnl_dollars": pnl_dollars,
                        "exit_reason": exit_reason,
                        "duration_minutes": (
                            pd.Timestamp(timestamp) - pd.Timestamp(entry_time)  # type: ignore[arg-type]
                        ).total_seconds()
                        / 60,
                        "is_winner": pnl_dollars > 0,
                    }
                    trades.append(trade)

                    in_trade = False
                    entry_price = 0.0
                    entry_time = None

        return trades

    def _calculate_performance_metrics(
        self, trades: List[Dict[str, Any]], prices: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from trade list.

        Args:
            trades: List of trade dictionaries
            prices: Price series for context

        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "final_value": self.config.initial_cash,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "avg_trade_return": 0.0,
                "total_pnl": 0.0,
            }

        # Calculate basic metrics
        total_pnl = sum(trade["pnl_dollars"] for trade in trades)
        winning_trades = [t for t in trades if t["is_winner"]]
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        # Calculate returns
        total_return = total_pnl / self.config.initial_cash
        final_value = self.config.initial_cash + total_pnl
        avg_trade_return = total_pnl / total_trades if total_trades > 0 else 0.0

        # Calculate drawdown (simplified)
        cumulative_pnl = 0.0
        peak_value = self.config.initial_cash
        max_drawdown = 0.0

        for trade in trades:
            cumulative_pnl += trade["pnl_dollars"]
            current_value = self.config.initial_cash + cumulative_pnl

            if current_value > peak_value:
                peak_value = current_value

            drawdown = (peak_value - current_value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate Sharpe ratio (simplified using trade returns)
        if trades:
            trade_returns = [
                t["pnl_dollars"] / self.config.initial_cash for t in trades
            ]
            mean_return = np.mean(trade_returns)
            std_return = np.std(trade_returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
            # Annualize (assuming ~250 trading days, rough estimate)
            sharpe_ratio *= np.sqrt(250 / len(trades)) if len(trades) > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_return": total_return,
            "final_value": final_value,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_trade_return": avg_trade_return,
            "total_pnl": total_pnl,
        }

    async def run_backtest(
        self,
        data: Optional[pd.DataFrame] = None,
        use_real_data: bool = True,
        n_bars: int = 1000,
    ) -> Dict[str, Any]:
        """
        Run the complete vector scalping backtest.

        Args:
            data: Optional OHLCV data (if None, fetches real data or generates sample data)
            use_real_data: Whether to fetch real data from TradingView (default: True)
            n_bars: Number of bars to fetch/generate

        Returns:
            Dictionary with backtest results and performance metrics
        """
        print("🚀 Starting Vector Scalping Backtest")
        print("=" * 60)

        # Use provided data, fetch real data, or generate sample data
        if data is None:
            if use_real_data:
                print(f"📊 Fetching real data for {self.config.symbol}...")
                try:
                    df_5min = await self.fetch_real_data(self.config.symbol, n_bars)
                    print(f"✅ Successfully fetched {len(df_5min)} bars of real data")
                except ValueError as e:
                    print(f"⚠️  Failed to fetch real data: {e}")
                    print("📊 Falling back to sample data...")
                    df_5min = self.generate_sample_data(n_bars)
            else:
                print(f"📊 Generating sample data for {self.config.symbol}...")
                df_5min = self.generate_sample_data(n_bars)
        else:
            df_5min = data.copy()
            print(f"📊 Using provided data ({len(df_5min)} bars)")

        print(f"📈 Data period: {df_5min.index[0]} to {df_5min.index[-1]}")
        print(f"📊 Total 5-min bars: {len(df_5min)}")

        # Resample to 15-minute
        df_15min = self.resample_to_15min(df_5min)
        print(f"📊 Total 15-min bars: {len(df_15min)}")

        # Calculate vector signals
        print("🔍 Calculating vector signals...")
        signals_df = self.calculate_vectors(df_5min, df_15min)

        # Align with price data
        price_data = df_5min.reindex(signals_df.index)
        close_prices = price_data["close"]

        print("📊 Signal statistics:")
        long_signals = (signals_df["signal"] == 1).sum()
        short_signals = (signals_df["signal"] == -1).sum()
        print(f"  📈 Long signals: {long_signals}")
        print(f"  📉 Short signals: {short_signals}")
        print(f"  📊 Total signals: {long_signals + short_signals}")

        # Create entries and exits
        entries = signals_df["signal"] == 1  # LONG entries
        short_entries = signals_df["signal"] == -1  # SHORT entries

        # Run custom backtest engine
        print("⚡ Running custom backtest engine...")

        # Simulate trades using custom logic
        long_trades = self._simulate_trades(close_prices, entries, is_long=True)
        short_trades = self._simulate_trades(close_prices, short_entries, is_long=False)

        # Calculate performance metrics
        long_metrics = self._calculate_performance_metrics(long_trades, close_prices)
        short_metrics = self._calculate_performance_metrics(short_trades, close_prices)

        # Combined performance (weighted average)
        combined_return = (
            long_metrics["total_return"] + short_metrics["total_return"]
        ) / 2
        combined_win_rate = (long_metrics["win_rate"] + short_metrics["win_rate"]) / 2
        combined_trades = long_metrics["total_trades"] + short_metrics["total_trades"]

        # Calculate comprehensive metrics
        results = {
            "config": self.config.model_dump(),
            "data_period": {
                "start": df_5min.index[0].isoformat(),
                "end": df_5min.index[-1].isoformat(),
                "total_bars": len(df_5min),
                "total_days": (df_5min.index[-1] - df_5min.index[0]).days,
            },
            "signal_statistics": {
                "long_signals": int(long_signals),
                "short_signals": int(short_signals),
                "total_signals": int(long_signals + short_signals),
                "signal_frequency": f"{((long_signals + short_signals) / len(df_5min) * 100):.2f}%",
            },
            "performance_metrics": {
                "initial_cash": self.config.initial_cash,
                "final_value_long": long_metrics["final_value"],
                "final_value_short": short_metrics["final_value"],
                "total_return_long": long_metrics["total_return"],
                "total_return_short": short_metrics["total_return"],
                "combined_return": combined_return,
                "max_drawdown_long": long_metrics["max_drawdown"],
                "max_drawdown_short": short_metrics["max_drawdown"],
                "sharpe_ratio_long": long_metrics["sharpe_ratio"],
                "sharpe_ratio_short": short_metrics["sharpe_ratio"],
                "win_rate_long": long_metrics["win_rate"],
                "win_rate_short": short_metrics["win_rate"],
                "total_trades_long": long_metrics["total_trades"],
                "total_trades_short": short_metrics["total_trades"],
                "avg_trade_return_long": long_metrics["avg_trade_return"],
                "avg_trade_return_short": short_metrics["avg_trade_return"],
                "combined_win_rate": combined_win_rate,
                "combined_total_trades": combined_trades,
            },
            "vector_analysis": {
                "avg_vector_strength": float(signals_df["vector_strength"].mean()),
                "max_vector_strength": float(signals_df["vector_strength"].max()),
                "strong_signals_pct": float(
                    (
                        signals_df["vector_strength"] > self.config.percentile_threshold
                    ).mean()
                    * 100
                ),
            },
            "trade_details": {"long_trades": long_trades, "short_trades": short_trades},
        }

        # Display results
        self._display_results(results)

        return results

    def _display_results(self, results: Dict[str, Any]) -> None:
        """Display formatted backtest results."""
        print("\n" + "=" * 60)
        print("📊 VECTOR SCALPING BACKTEST RESULTS")
        print("=" * 60)

        # Data period
        data = results["data_period"]
        print(
            f"📅 Period: {data['start'][:10]} to {data['end'][:10]} ({data['total_days']} days)"
        )
        print(f"📊 Total bars: {data['total_bars']:,}")

        # Signal statistics
        signals = results["signal_statistics"]
        print("\n🎯 Signal Statistics:")
        print(f"  📈 Long signals: {signals['long_signals']:,}")
        print(f"  📉 Short signals: {signals['short_signals']:,}")
        print(f"  📊 Total signals: {signals['total_signals']:,}")
        print(f"  📊 Signal frequency: {signals['signal_frequency']}")

        # Performance metrics
        perf = results["performance_metrics"]
        print("\n💰 Performance Metrics:")
        print(f"  💵 Initial capital: ${perf['initial_cash']:,.2f}")
        print(f"  📈 Long strategy return: {perf['total_return_long']:+.2%}")
        print(f"  📉 Short strategy return: {perf['total_return_short']:+.2%}")
        print(f"  🎯 Combined return: {perf['combined_return']:+.2%}")
        print(f"  📉 Max drawdown (Long): {perf['max_drawdown_long']:.2%}")
        print(f"  📉 Max drawdown (Short): {perf['max_drawdown_short']:.2%}")

        print("\n📈 Trading Statistics:")
        print(f"  🎯 Win rate (Long): {perf['win_rate_long']:.1%}")
        print(f"  🎯 Win rate (Short): {perf['win_rate_short']:.1%}")
        print(f"  🎯 Combined win rate: {perf['combined_win_rate']:.1%}")
        print(f"  📊 Total trades (Long): {perf['total_trades_long']}")
        print(f"  📊 Total trades (Short): {perf['total_trades_short']}")
        print(f"  📊 Combined total trades: {perf['combined_total_trades']}")
        print(f"  📊 Avg trade return (Long): ${perf['avg_trade_return_long']:.2f}")
        print(f"  📊 Avg trade return (Short): ${perf['avg_trade_return_short']:.2f}")
        print(f"  📊 Sharpe ratio (Long): {perf['sharpe_ratio_long']:.2f}")
        print(f"  📊 Sharpe ratio (Short): {perf['sharpe_ratio_short']:.2f}")

        # Vector analysis
        vector = results["vector_analysis"]
        print("\n🔍 Vector Analysis:")
        print(f"  📊 Average vector strength: {vector['avg_vector_strength']:.1f}")
        print(f"  📊 Maximum vector strength: {vector['max_vector_strength']:.1f}")
        print(f"  🎯 Strong signals %: {vector['strong_signals_pct']:.1f}%")

        print("=" * 60)


async def main():
    """Main execution function."""
    # Default configuration for TFEX:S50U2025 futures
    config = BacktestConfig(
        symbol="TFEX:S50U2025",
        initial_cash=10000.0,
        pip_value=1.0,
        take_profit_pips=20,
        stop_loss_pips=30,
        pip_size=0.01,  # 2-decimal pair for futures
        vector_period=5,
        percentile_threshold=60.0,
        direction_threshold=0.0005,
        lookback_period=100,
    )

    # Create and run backtester
    backtester = VectorBacktester(config)
    results = await backtester.run_backtest()

    print("\n✅ Backtest completed successfully!")
    return results


if __name__ == "__main__":
    # Run the backtest
    asyncio.run(main())
