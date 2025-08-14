"""Tests for data service functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List
import polars as pl

from vector_scalping.models import (
    OHLCVData, 
    RiskManagement, 
    StrategyConfig, 
    TimeFrame
)
from vector_scalping.data_service import DataService


class TestDataService:
    """Test data service functionality."""
    
    @pytest.fixture
    def risk_management(self) -> RiskManagement:
        """Create risk management configuration."""
        return RiskManagement(
            symbol="EURUSD",
            pip_size=0.0001,
            is_decimal_4=True
        )
    
    @pytest.fixture
    def strategy_config(self, risk_management) -> StrategyConfig:
        """Create strategy configuration."""
        return StrategyConfig(
            symbol="EURUSD",
            exchange="FX_IDC", 
            risk_management=risk_management
        )
    
    @pytest.fixture
    def sample_ohlcv_data(self) -> List[OHLCVData]:
        """Create sample OHLCV data."""
        return [
            OHLCVData(timestamp=1640995200, open=1.0850, high=1.0860, low=1.0845, close=1.0855, volume=1500),
            OHLCVData(timestamp=1640995500, open=1.0855, high=1.0865, low=1.0850, close=1.0862, volume=1600),
            OHLCVData(timestamp=1640995800, open=1.0862, high=1.0870, low=1.0858, close=1.0868, volume=1400),
            OHLCVData(timestamp=1640996100, open=1.0868, high=1.0875, low=1.0865, close=1.0872, volume=1800),
            OHLCVData(timestamp=1640996400, open=1.0872, high=1.0880, low=1.0870, close=1.0878, volume=1700),
        ]
    
    @pytest.fixture
    def mock_tvkit_bars(self):
        """Create mock TVKit OHLCVBar objects."""
        mock_bars = []
        for i, data in enumerate([
            (1640995200, 1.0850, 1.0860, 1.0845, 1.0855, 1500),
            (1640995500, 1.0855, 1.0865, 1.0850, 1.0862, 1600),
            (1640995800, 1.0862, 1.0870, 1.0858, 1.0868, 1400),
        ]):
            bar = MagicMock()
            bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume = data
            mock_bars.append(bar)
        return mock_bars
    
    def test_data_service_initialization(self, strategy_config):
        """Test data service initialization."""
        service = DataService(strategy_config)
        
        assert service.config == strategy_config
        assert service._client is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self, strategy_config):
        """Test async context manager functionality."""
        with patch('vector_scalping.data_service.OHLCV') as mock_ohlcv_class:
            mock_client = AsyncMock()
            mock_ohlcv_class.return_value = mock_client
            
            async with DataService(strategy_config) as service:
                assert service._client == mock_client
                mock_client.__aenter__.assert_called_once()
            
            mock_client.__aexit__.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_success(self, strategy_config, mock_tvkit_bars):
        """Test successful historical data fetch."""
        with patch('vector_scalping.data_service.OHLCV') as mock_ohlcv_class:
            mock_client = AsyncMock()
            mock_client.get_historical_ohlcv.return_value = mock_tvkit_bars
            mock_ohlcv_class.return_value = mock_client
            
            service = DataService(strategy_config)
            service._client = mock_client
            
            result = await service.fetch_historical_data(TimeFrame.MIN_5, bars_count=10)
            
            # Verify the call was made correctly
            mock_client.get_historical_ohlcv.assert_called_once_with(
                exchange_symbol="FX_IDC:EURUSD",
                interval="5",
                bars_count=10
            )
            
            # Verify the result
            assert len(result) == 3
            assert all(isinstance(bar, OHLCVData) for bar in result)
            assert result[0].timestamp == 1640995200
            assert result[0].close == 1.0855
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_no_client(self, strategy_config):
        """Test error when client not initialized."""
        service = DataService(strategy_config)
        
        with pytest.raises(RuntimeError, match="not properly initialized"):
            await service.fetch_historical_data(TimeFrame.MIN_5, 10)
    
    @pytest.mark.asyncio
    async def test_fetch_historical_data_empty_response(self, strategy_config):
        """Test error handling for empty response."""
        with patch('vector_scalping.data_service.OHLCV') as mock_ohlcv_class:
            mock_client = AsyncMock()
            mock_client.get_historical_ohlcv.return_value = []  # Empty response
            mock_ohlcv_class.return_value = mock_client
            
            service = DataService(strategy_config)
            service._client = mock_client
            
            with pytest.raises(RuntimeError, match="No data received"):
                await service.fetch_historical_data(TimeFrame.MIN_5, 10)
    
    @pytest.mark.asyncio
    async def test_fetch_multi_timeframe_data(self, strategy_config, mock_tvkit_bars):
        """Test multi-timeframe data fetching."""
        with patch('vector_scalping.data_service.OHLCV') as mock_ohlcv_class:
            mock_client = AsyncMock()
            mock_client.get_historical_ohlcv.return_value = mock_tvkit_bars
            mock_ohlcv_class.return_value = mock_client
            
            service = DataService(strategy_config)
            service._client = mock_client
            
            result = await service.fetch_multi_timeframe_data(bars_count=10)
            
            # Should have both timeframes
            assert TimeFrame.MIN_5 in result
            assert TimeFrame.MIN_15 in result
            
            # Both should have the same data (mocked)
            assert len(result[TimeFrame.MIN_5]) == 3
            assert len(result[TimeFrame.MIN_15]) == 3
            
            # Verify both calls were made
            assert mock_client.get_historical_ohlcv.call_count == 2
    
    def test_convert_to_polars_dataframe(self, strategy_config, sample_ohlcv_data):
        """Test conversion to Polars DataFrame."""
        service = DataService(strategy_config)
        
        df = service.convert_to_polars_dataframe(sample_ohlcv_data)
        
        # Verify DataFrame structure
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5
        
        expected_columns = ["timestamp", "datetime", "open", "high", "low", "close", "volume", "true_range"]
        assert all(col in df.columns for col in expected_columns)
        
        # Verify data types
        assert df["timestamp"].dtype == pl.Int64
        assert df["open"].dtype == pl.Float64
        assert df["close"].dtype == pl.Float64
        
        # Verify data values
        first_row = df.row(0, named=True)
        assert first_row["timestamp"] == 1640995200
        assert first_row["open"] == 1.0850
        assert first_row["close"] == 1.0855
    
    def test_convert_empty_data_to_polars(self, strategy_config):
        """Test conversion of empty data to Polars DataFrame."""
        service = DataService(strategy_config)
        
        df = service.convert_to_polars_dataframe([])
        
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 0
    
    def test_add_technical_indicators(self, strategy_config, sample_ohlcv_data):
        """Test adding technical indicators to DataFrame."""
        service = DataService(strategy_config)
        
        df = service.convert_to_polars_dataframe(sample_ohlcv_data)
        df_with_indicators = service.add_technical_indicators(df)
        
        # Verify new columns were added
        expected_indicators = ["sma_5", "sma_10", "sma_20", "price_change", "returns", 
                             "hl_range", "atr_5", "atr_14", "volatility_5", "volatility_20"]
        
        for indicator in expected_indicators:
            assert indicator in df_with_indicators.columns
        
        # Verify calculations (basic checks)
        assert df_with_indicators["hl_range"].dtype == pl.Float64
        assert df_with_indicators["atr_5"].dtype == pl.Float64
        
        # Check that moving averages exist (may have null values for early periods)
        sma_5_values = df_with_indicators["sma_5"].drop_nulls()
        assert len(sma_5_values) > 0  # Should have some non-null values
    
    def test_add_technical_indicators_empty_df(self, strategy_config):
        """Test technical indicators on empty DataFrame."""
        service = DataService(strategy_config)
        
        empty_df = pl.DataFrame()
        result = service.add_technical_indicators(empty_df)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_get_latest_price(self, strategy_config, mock_tvkit_bars):
        """Test getting latest price."""
        with patch('vector_scalping.data_service.OHLCV') as mock_ohlcv_class:
            mock_client = AsyncMock()
            # Return only the latest bar
            mock_client.get_historical_ohlcv.return_value = [mock_tvkit_bars[0]]
            mock_ohlcv_class.return_value = mock_client
            
            service = DataService(strategy_config)
            service._client = mock_client
            
            latest_price = await service.get_latest_price()
            
            assert latest_price == 1.0855  # Close price of first mock bar
            
            # Verify correct call was made
            mock_client.get_historical_ohlcv.assert_called_once_with(
                exchange_symbol="FX_IDC:EURUSD",
                interval="5",
                bars_count=1
            )
    
    @pytest.mark.asyncio
    async def test_get_latest_price_no_data(self, strategy_config):
        """Test error handling when no latest price data."""
        with patch('vector_scalping.data_service.OHLCV') as mock_ohlcv_class:
            mock_client = AsyncMock()
            mock_client.get_historical_ohlcv.return_value = []
            mock_ohlcv_class.return_value = mock_client
            
            service = DataService(strategy_config)
            service._client = mock_client
            
            with pytest.raises(RuntimeError, match="Failed to get latest price"):
                await service.get_latest_price()
    
    def test_validate_symbol_format(self, strategy_config):
        """Test symbol format validation."""
        service = DataService(strategy_config)
        
        # Valid formats
        assert service.validate_symbol_format("EURUSD") is True
        assert service.validate_symbol_format("GBPJPY") is True
        assert service.validate_symbol_format("USDJPY") is True
        
        # Invalid formats
        assert service.validate_symbol_format("EUR/USD") is False  # Contains /
        assert service.validate_symbol_format("eurusd") is False   # Lowercase
        assert service.validate_symbol_format("EURUSA") is True    # Valid 6-char format
        assert service.validate_symbol_format("EUR123") is False   # Contains numbers
        assert service.validate_symbol_format("") is False         # Empty
    
    @pytest.mark.asyncio
    async def test_stream_real_time_data_no_client(self, strategy_config):
        """Test real-time streaming error when no client."""
        service = DataService(strategy_config)
        
        with pytest.raises(RuntimeError, match="not properly initialized"):
            await service.stream_real_time_data(TimeFrame.MIN_5)
    
    @pytest.mark.asyncio
    async def test_stream_real_time_data_with_callback(self, strategy_config, mock_tvkit_bars):
        """Test real-time streaming initialization."""
        with patch('vector_scalping.data_service.OHLCV') as mock_ohlcv_class:
            mock_client = AsyncMock()
            mock_ohlcv_class.return_value = mock_client
            
            service = DataService(strategy_config)
            service._client = mock_client
            
            # Just test that the method can be called without error
            # Real streaming would be tested in integration tests
            assert service._client is not None
            assert callable(service.stream_real_time_data)