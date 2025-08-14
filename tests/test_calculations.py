"""Tests for vector calculation functions."""

import pytest
import math
from typing import List

from vector_scalping.models import OHLCVData
from vector_scalping.calculations import VectorCalculations


class TestVectorCalculations:
    """Test vector calculation methods."""
    
    @pytest.fixture
    def sample_ohlcv_data(self) -> List[OHLCVData]:
        """Create sample OHLCV data for testing."""
        return [
            OHLCVData(timestamp=1640995200, open=1.0850, high=1.0860, low=1.0845, close=1.0855, volume=1500),
            OHLCVData(timestamp=1640995500, open=1.0855, high=1.0865, low=1.0850, close=1.0862, volume=1600),
            OHLCVData(timestamp=1640995800, open=1.0862, high=1.0870, low=1.0858, close=1.0868, volume=1400),
            OHLCVData(timestamp=1640996100, open=1.0868, high=1.0875, low=1.0865, close=1.0872, volume=1800),
            OHLCVData(timestamp=1640996400, open=1.0872, high=1.0880, low=1.0870, close=1.0878, volume=1700),
        ]
    
    def test_calculate_price_vector(self, sample_ohlcv_data):
        """Test price vector calculation."""
        vector = VectorCalculations.calculate_price_vector(sample_ohlcv_data, period=5)
        
        # Expected values based on sample data
        expected_displacement = 1.0878 - 1.0855  # 0.0023
        expected_magnitude = abs(expected_displacement)  # 0.0023
        expected_price_range = 1.0880 - 1.0845  # 0.0035
        expected_direction = expected_displacement / expected_price_range  # 0.657...
        
        assert vector.displacement == pytest.approx(expected_displacement, abs=1e-6)
        assert vector.magnitude == pytest.approx(expected_magnitude, abs=1e-6)
        assert vector.direction == pytest.approx(expected_direction, abs=1e-3)
        assert vector.period == 5
        assert vector.start_price == 1.0855
        assert vector.end_price == 1.0878
        assert vector.price_range == pytest.approx(expected_price_range, abs=1e-6)
    
    def test_calculate_momentum_vector(self, sample_ohlcv_data):
        """Test momentum vector calculation."""
        vector = VectorCalculations.calculate_momentum_vector(sample_ohlcv_data, period=5)
        
        # Expected calculations
        expected_price_momentum = (1.0878 - 1.0855) / 5  # 0.00046
        
        # Calculate expected volatility (Average True Range)
        true_ranges = [
            max(1.0865 - 1.0850, abs(1.0865 - 1.0855), abs(1.0850 - 1.0855)),  # 0.0015
            max(1.0870 - 1.0858, abs(1.0870 - 1.0862), abs(1.0858 - 1.0862)),  # 0.0012
            max(1.0875 - 1.0865, abs(1.0875 - 1.0868), abs(1.0865 - 1.0868)),  # 0.0010
            max(1.0880 - 1.0870, abs(1.0880 - 1.0872), abs(1.0870 - 1.0872)),  # 0.0010
        ]
        expected_volatility = sum(true_ranges) / len(true_ranges)
        expected_magnitude = math.sqrt(expected_price_momentum**2 + expected_volatility**2)
        expected_direction = expected_price_momentum / expected_magnitude
        
        assert vector.price_momentum == pytest.approx(expected_price_momentum, abs=1e-6)
        assert vector.volatility == pytest.approx(expected_volatility, abs=1e-4)
        assert vector.magnitude == pytest.approx(expected_magnitude, abs=1e-4)
        assert vector.direction == pytest.approx(expected_direction, abs=1e-3)
        assert vector.period == 5
    
    def test_combine_vectors(self, sample_ohlcv_data):
        """Test vector combination."""
        # Calculate individual vectors
        tf5_price = VectorCalculations.calculate_price_vector(sample_ohlcv_data, period=5)
        tf5_momentum = VectorCalculations.calculate_momentum_vector(sample_ohlcv_data, period=5)
        
        # Use same data for tf15 (in real scenario, this would be different timeframe data)
        tf15_price = VectorCalculations.calculate_price_vector(sample_ohlcv_data, period=5)
        tf15_momentum = VectorCalculations.calculate_momentum_vector(sample_ohlcv_data, period=5)
        
        combined = VectorCalculations.combine_vectors(
            tf5_price, tf5_momentum, tf15_price, tf15_momentum,
            tf5_weight=0.7, tf15_weight=0.3,
            tf5_dir_weight=0.6, tf15_dir_weight=0.4
        )
        
        # Expected values with default weights
        expected_magnitude = (tf5_momentum.magnitude * 0.7) + (tf15_momentum.magnitude * 0.3)
        expected_direction = (tf5_momentum.direction * 0.6) + (tf15_momentum.direction * 0.4)
        
        assert combined.tf5_magnitude == tf5_momentum.magnitude
        assert combined.tf15_magnitude == tf15_momentum.magnitude
        assert combined.combined_magnitude == pytest.approx(expected_magnitude, abs=1e-6)
        assert combined.combined_direction == pytest.approx(expected_direction, abs=1e-6)
        assert combined.signal_strength == 0.0  # Not calculated yet
    
    def test_calculate_percentile_rank(self):
        """Test percentile rank calculation."""
        historical_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Test value at specific percentile
        rank = VectorCalculations.calculate_percentile_rank(3.0, historical_values)
        # 3.0 is the 3rd element (index 2) out of 6 values, so 2/5 * 100 = 40%
        assert rank == pytest.approx(40.0, abs=1)
        
        # Test value at 100th percentile  
        rank = VectorCalculations.calculate_percentile_rank(6.0, historical_values)
        assert rank == 100.0
        
        # Test value at 0th percentile
        rank = VectorCalculations.calculate_percentile_rank(0.5, historical_values)
        assert rank == 0.0
    
    def test_detect_divergence(self, sample_ohlcv_data):
        """Test divergence detection."""
        # Create comparison data with different trend
        comparison_data = [
            OHLCVData(timestamp=1640994000, open=1.0900, high=1.0910, low=1.0895, close=1.0905, volume=1500),
            OHLCVData(timestamp=1640994300, open=1.0905, high=1.0915, low=1.0900, close=1.0912, volume=1600),
            OHLCVData(timestamp=1640994600, open=1.0912, high=1.0920, low=1.0908, close=1.0918, volume=1400),
            OHLCVData(timestamp=1640994900, open=1.0918, high=1.0925, low=1.0915, close=1.0922, volume=1800),
            OHLCVData(timestamp=1640995200, open=1.0922, high=1.0930, low=1.0920, close=1.0928, volume=1700),
        ]
        
        divergence = VectorCalculations.detect_divergence(
            current_data=sample_ohlcv_data,
            comparison_data=comparison_data,
            period=5
        )
        
        assert isinstance(divergence.price_trend, float)
        assert isinstance(divergence.momentum_trend, float)
        assert isinstance(divergence.is_bullish_divergence, bool)
        assert isinstance(divergence.is_bearish_divergence, bool)
        assert 0.0 <= divergence.divergence_strength <= 1.0
    
    def test_insufficient_data_errors(self):
        """Test error handling for insufficient data."""
        insufficient_data = [
            OHLCVData(timestamp=1640995200, open=1.0850, high=1.0860, low=1.0845, close=1.0855, volume=1500),
            OHLCVData(timestamp=1640995500, open=1.0855, high=1.0865, low=1.0850, close=1.0862, volume=1600),
        ]
        
        # Test price vector with insufficient data
        with pytest.raises(ValueError, match="Need at least 5 candles"):
            VectorCalculations.calculate_price_vector(insufficient_data, period=5)
        
        # Test momentum vector with insufficient data
        with pytest.raises(ValueError, match="Need at least 5 candles"):
            VectorCalculations.calculate_momentum_vector(insufficient_data, period=5)
    
    def test_invalid_period(self, sample_ohlcv_data):
        """Test error handling for invalid periods."""
        with pytest.raises(ValueError, match="Period must be positive"):
            VectorCalculations.calculate_price_vector(sample_ohlcv_data, period=0)
        
        with pytest.raises(ValueError, match="Period must be positive"):
            VectorCalculations.calculate_momentum_vector(sample_ohlcv_data, period=-1)
    
    def test_invalid_weight_combinations(self, sample_ohlcv_data):
        """Test error handling for invalid weight combinations."""
        tf5_price = VectorCalculations.calculate_price_vector(sample_ohlcv_data, period=5)
        tf5_momentum = VectorCalculations.calculate_momentum_vector(sample_ohlcv_data, period=5)
        tf15_price = VectorCalculations.calculate_price_vector(sample_ohlcv_data, period=5)
        tf15_momentum = VectorCalculations.calculate_momentum_vector(sample_ohlcv_data, period=5)
        
        # Test invalid magnitude weights
        with pytest.raises(ValueError, match="Magnitude weights must sum to 1.0"):
            VectorCalculations.combine_vectors(
                tf5_price, tf5_momentum, tf15_price, tf15_momentum,
                tf5_weight=0.8, tf15_weight=0.3  # Sum = 1.1
            )
        
        # Test invalid direction weights
        with pytest.raises(ValueError, match="Direction weights must sum to 1.0"):
            VectorCalculations.combine_vectors(
                tf5_price, tf5_momentum, tf15_price, tf15_momentum,
                tf5_dir_weight=0.7, tf15_dir_weight=0.4  # Sum = 1.1
            )
    
    def test_empty_historical_values_error(self):
        """Test error handling for empty historical values."""
        with pytest.raises(ValueError, match="Historical values cannot be empty"):
            VectorCalculations.calculate_percentile_rank(5.0, [])