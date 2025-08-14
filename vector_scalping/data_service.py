"""Async data fetching service using tvkit and Polars for data processing."""

import asyncio
import logging
from typing import List, Optional, Dict, Callable

import polars as pl
from tvkit.api.chart.ohlcv import OHLCV
from tvkit.api.chart.models.ohlcv import OHLCVBar

from .models import OHLCVData, TimeFrame, StrategyConfig


class DataService:
    """Async service for fetching and processing forex data using tvkit."""
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize data service with strategy configuration.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._client: Optional[OHLCV] = None
        
    async def __aenter__(self) -> "DataService":
        """Async context manager entry."""
        self._client = OHLCV()
        await self._client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def fetch_historical_data(
        self,
        timeframe: TimeFrame,
        bars_count: int = 100
    ) -> List[OHLCVData]:
        """
        Fetch historical OHLCV data for the configured symbol.
        
        Args:
            timeframe: Chart timeframe (5, 15, 30, 60 minutes)
            bars_count: Number of historical bars to fetch
            
        Returns:
            List of OHLCVData objects sorted by timestamp
            
        Raises:
            RuntimeError: If client not initialized or data fetch fails
            ValueError: If invalid timeframe provided
            
        Example:
            >>> async with DataService(config) as service:
            ...     data_5m = await service.fetch_historical_data(TimeFrame.MIN_5, 100)
            ...     print(f"Fetched {len(data_5m)} 5-minute bars")
        """
        if not self._client:
            raise RuntimeError("DataService not properly initialized. Use async context manager.")
        
        try:
            # Construct exchange symbol format expected by tvkit
            exchange_symbol = f"{self.config.exchange}:{self.config.symbol}"
            
            self.logger.info(
                f"Fetching {bars_count} bars of {timeframe.value}-minute data for {exchange_symbol}"
            )
            
            # Fetch data from tvkit
            raw_bars: List[OHLCVBar] = await self._client.get_historical_ohlcv(
                exchange_symbol=exchange_symbol,
                interval=timeframe.value,
                bars_count=bars_count
            )
            
            if not raw_bars:
                raise RuntimeError(f"No data received for {exchange_symbol}")
            
            # Convert to our OHLCVData model with validation
            ohlcv_data = []
            for bar in raw_bars:
                try:
                    ohlcv = OHLCVData(
                        timestamp=int(bar.timestamp),
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume
                    )
                    ohlcv_data.append(ohlcv)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid bar: {bar} - Error: {e}")
                    continue
            
            # Sort by timestamp to ensure chronological order
            ohlcv_data.sort(key=lambda x: x.timestamp)
            
            self.logger.info(
                f"Successfully processed {len(ohlcv_data)} bars for {timeframe.value}-minute timeframe"
            )
            
            return ohlcv_data
            
        except Exception as e:
            error_msg = f"Failed to fetch {timeframe.value}-minute data: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def fetch_multi_timeframe_data(
        self,
        bars_count: int = 100
    ) -> Dict[TimeFrame, List[OHLCVData]]:
        """
        Fetch data for both 5-minute and 15-minute timeframes concurrently.
        
        Args:
            bars_count: Number of bars to fetch for each timeframe
            
        Returns:
            Dictionary mapping timeframes to OHLCV data
            
        Raises:
            RuntimeError: If any data fetch fails
            
        Example:
            >>> async with DataService(config) as service:
            ...     data = await service.fetch_multi_timeframe_data(100)
            ...     tf5_data = data[TimeFrame.MIN_5]
            ...     tf15_data = data[TimeFrame.MIN_15]
        """
        self.logger.info("Fetching multi-timeframe data concurrently")
        
        # Fetch both timeframes concurrently for better performance
        try:
            tf5_task = self.fetch_historical_data(TimeFrame.MIN_5, bars_count)
            tf15_task = self.fetch_historical_data(TimeFrame.MIN_15, bars_count)
            
            tf5_data, tf15_data = await asyncio.gather(tf5_task, tf15_task)
            
            return {
                TimeFrame.MIN_5: tf5_data,
                TimeFrame.MIN_15: tf15_data
            }
            
        except Exception as e:
            error_msg = f"Failed to fetch multi-timeframe data: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def convert_to_polars_dataframe(
        self, 
        data: List[OHLCVData]
    ) -> pl.DataFrame:
        """
        Convert OHLCV data to Polars DataFrame for analysis.
        
        Args:
            data: List of OHLCVData objects
            
        Returns:
            Polars DataFrame with OHLCV data
            
        Example:
            >>> service = DataService(config)
            >>> df = service.convert_to_polars_dataframe(ohlcv_data)
            >>> print(df.select(["timestamp", "close"]).head())
        """
        if not data:
            return pl.DataFrame()
        
        # Extract data for DataFrame
        records = []
        for bar in data:
            records.append({
                "timestamp": bar.timestamp,
                "datetime": bar.datetime,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "true_range": bar.true_range
            })
        
        # Create Polars DataFrame
        df = pl.DataFrame(records)
        
        # Ensure proper data types
        df = df.with_columns([
            pl.col("timestamp").cast(pl.Int64),
            pl.col("datetime").cast(pl.Datetime),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("true_range").cast(pl.Float64)
        ])
        
        return df
    
    def add_technical_indicators(
        self, 
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Add technical indicators to Polars DataFrame.
        
        Args:
            df: Polars DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
            
        Example:
            >>> df_with_indicators = service.add_technical_indicators(df)
            >>> print(df_with_indicators.select(["close", "sma_5", "atr_5"]).tail())
        """
        if df.is_empty():
            return df
        
        # Calculate moving averages
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=5).alias("sma_5"),
            pl.col("close").rolling_mean(window_size=10).alias("sma_10"),
            pl.col("close").rolling_mean(window_size=20).alias("sma_20")
        ])
        
        # Calculate price changes and returns
        df = df.with_columns([
            (pl.col("close") - pl.col("close").shift(1)).alias("price_change"),
            ((pl.col("close") / pl.col("close").shift(1)) - 1).alias("returns"),
            (pl.col("high") - pl.col("low")).alias("hl_range")
        ])
        
        # Calculate Average True Range (ATR)
        df = df.with_columns([
            pl.col("true_range").rolling_mean(window_size=5).alias("atr_5"),
            pl.col("true_range").rolling_mean(window_size=14).alias("atr_14")
        ])
        
        # Calculate volatility measures
        df = df.with_columns([
            pl.col("returns").rolling_std(window_size=5).alias("volatility_5"),
            pl.col("returns").rolling_std(window_size=20).alias("volatility_20")
        ])
        
        return df
    
    async def get_latest_price(self) -> float:
        """
        Get the latest price for the configured symbol.
        
        Returns:
            Latest close price
            
        Raises:
            RuntimeError: If unable to fetch latest price
        """
        try:
            # Fetch just the latest bar
            latest_data = await self.fetch_historical_data(TimeFrame.MIN_5, bars_count=1)
            
            if not latest_data:
                raise RuntimeError("No data received for latest price")
            
            return latest_data[-1].close
            
        except Exception as e:
            error_msg = f"Failed to get latest price: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def stream_real_time_data(
        self,
        timeframe: TimeFrame,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Stream real-time data updates (for live trading).
        
        Note: This is a placeholder for real-time streaming functionality.
        In a production system, this would establish a WebSocket connection
        and stream live updates.
        
        Args:
            timeframe: Chart timeframe for streaming
            callback: Optional callback function for processing updates
            
        Example:
            >>> async def process_update(bar):
            ...     print(f"New bar: {bar.close}")
            >>> await service.stream_real_time_data(TimeFrame.MIN_5, process_update)
        """
        if not self._client:
            raise RuntimeError("DataService not properly initialized")
        
        exchange_symbol = f"{self.config.exchange}:{self.config.symbol}"
        
        self.logger.info(f"Starting real-time stream for {exchange_symbol}")
        
        try:
            # Use tvkit's real-time streaming capability
            async for bar in self._client.get_ohlcv(
                exchange_symbol=exchange_symbol,
                interval=timeframe.value,
                bars_count=1
            ):
                try:
                    # Convert to our OHLCVData model
                    ohlcv = OHLCVData(
                        timestamp=int(bar.timestamp),
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume
                    )
                    
                    self.logger.debug(f"Received real-time update: {ohlcv.close}")
                    
                    # Call callback if provided
                    if callback:
                        await callback(ohlcv)
                        
                except Exception as e:
                    self.logger.warning(f"Error processing real-time update: {e}")
                    continue
                    
        except Exception as e:
            error_msg = f"Real-time streaming failed: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def validate_symbol_format(self, symbol: str) -> bool:
        """
        Validate if symbol is in correct format for the exchange.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if valid format, False otherwise
            
        Example:
            >>> service.validate_symbol_format("EURUSD")  # True
            >>> service.validate_symbol_format("EUR/USD")  # False
        """
        # Basic validation for forex symbols (6 characters, all uppercase)
        if len(symbol) != 6:
            return False
        
        if not symbol.isupper():
            return False
        
        if not symbol.isalpha():
            return False
        
        return True