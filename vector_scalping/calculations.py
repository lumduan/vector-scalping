"""Vector calculation functions for the scalping strategy."""

import math
from typing import List

from .models import OHLCVData, PriceVector, MomentumVector, CombinedVector, DivergenceSignal


class VectorCalculations:
    """Core vector calculation methods for trading strategy."""
    
    @staticmethod
    def calculate_price_vector(
        data: List[OHLCVData], 
        period: int = 5
    ) -> PriceVector:
        """
        Calculate price vector from OHLCV data.
        
        Price Vector Calculation:
        - displacement = close[n] - close[0]
        - magnitude = abs(displacement)  
        - direction = displacement / max(high_range, 0.0001)
        
        Args:
            data: List of OHLCV data (must have at least `period` elements)
            period: Number of candles to use for calculation (default 5)
            
        Returns:
            PriceVector object with displacement, magnitude, and direction
            
        Raises:
            ValueError: If insufficient data or invalid period
            
        Example:
            >>> data = [OHLCVData(...), ...]  # 5 candles
            >>> vector = VectorCalculations.calculate_price_vector(data, period=5)
            >>> print(f"Direction: {vector.direction}, Magnitude: {vector.magnitude}")
        """
        if len(data) < period:
            raise ValueError(f"Need at least {period} candles, got {len(data)}")
        
        if period <= 0:
            raise ValueError("Period must be positive")
        
        # Use the last `period` candles
        candles = data[-period:]
        
        start_price = candles[0].close
        end_price = candles[-1].close
        
        # Calculate displacement over N candles
        displacement = end_price - start_price
        
        # Calculate price magnitude (absolute movement)
        magnitude = abs(displacement)
        
        # Calculate price range over the period
        all_highs = [candle.high for candle in candles]
        all_lows = [candle.low for candle in candles]
        price_range = max(all_highs) - min(all_lows)
        
        # Avoid division by zero
        safe_range = max(price_range, 0.0001)
        
        # Calculate direction (normalized to -1 to 1)
        direction = displacement / safe_range
        
        return PriceVector(
            displacement=displacement,
            magnitude=magnitude,
            direction=direction,
            period=period,
            start_price=start_price,
            end_price=end_price,
            price_range=price_range
        )
    
    @staticmethod
    def calculate_momentum_vector(
        data: List[OHLCVData], 
        period: int = 5
    ) -> MomentumVector:
        """
        Calculate momentum vector from OHLCV data.
        
        Momentum Vector Calculation:
        - price_momentum = (close[n] - close[0]) / n
        - volatility = average_true_range over period
        - magnitude = sqrt(price_momentum^2 + volatility^2)
        - direction = price_momentum / magnitude (if magnitude > 0)
        
        Args:
            data: List of OHLCV data (must have at least `period` elements)
            period: Number of candles to use for calculation (default 5)
            
        Returns:
            MomentumVector object with momentum and volatility components
            
        Raises:
            ValueError: If insufficient data or invalid period
            
        Example:
            >>> data = [OHLCVData(...), ...]  # 5 candles
            >>> vector = VectorCalculations.calculate_momentum_vector(data, period=5)
            >>> print(f"Momentum Direction: {vector.direction}")
        """
        if len(data) < period:
            raise ValueError(f"Need at least {period} candles, got {len(data)}")
        
        if period <= 0:
            raise ValueError("Period must be positive")
        
        # Use the last `period` candles
        candles = data[-period:]
        
        # Calculate price momentum (rate of change with bias)
        start_close = candles[0].close
        end_close = candles[-1].close
        price_momentum = (end_close - start_close) / period
        
        # Calculate volatility component (Average True Range)
        true_ranges = []
        for i in range(1, len(candles)):
            current = candles[i]
            previous = candles[i-1]
            
            # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
            tr = max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close)
            )
            true_ranges.append(tr)
        
        # If only one candle, use the high-low range
        if not true_ranges:
            volatility = candles[0].high - candles[0].low
        else:
            volatility = sum(true_ranges) / len(true_ranges)
        
        # Calculate momentum magnitude
        magnitude = math.sqrt(price_momentum**2 + volatility**2)
        
        # Calculate momentum direction
        if magnitude > 0:
            direction = price_momentum / magnitude
        else:
            direction = 0.0
        
        return MomentumVector(
            price_momentum=price_momentum,
            volatility=volatility,
            magnitude=magnitude,
            direction=direction,
            period=period
        )
    
    @staticmethod
    def combine_vectors(
        tf5_price: PriceVector,
        tf5_momentum: MomentumVector,
        tf15_price: PriceVector,
        tf15_momentum: MomentumVector,
        tf5_weight: float = 0.7,
        tf15_weight: float = 0.3,
        tf5_dir_weight: float = 0.6,
        tf15_dir_weight: float = 0.4
    ) -> CombinedVector:
        """
        Combine 5-min and 15-min vectors using weighted averages.
        
        Multi-timeframe Vector Combination:
        - combined_magnitude = (tf5_magnitude * 0.7) + (tf15_magnitude * 0.3)
        - combined_direction = (tf5_direction * 0.6) + (tf15_direction * 0.4)
        
        Args:
            tf5_price: 5-minute price vector
            tf5_momentum: 5-minute momentum vector
            tf15_price: 15-minute price vector
            tf15_momentum: 15-minute momentum vector
            tf5_weight: Weight for 5-minute magnitude (default 0.7)
            tf15_weight: Weight for 15-minute magnitude (default 0.3)
            tf5_dir_weight: Weight for 5-minute direction (default 0.6)
            tf15_dir_weight: Weight for 15-minute direction (default 0.4)
            
        Returns:
            CombinedVector with weighted combination results
            
        Raises:
            ValueError: If weights don't sum to 1.0
            
        Example:
            >>> combined = VectorCalculations.combine_vectors(
            ...     tf5_price, tf5_momentum, tf15_price, tf15_momentum
            ... )
            >>> print(f"Combined direction: {combined.combined_direction}")
        """
        # Validate weights
        if abs(tf5_weight + tf15_weight - 1.0) > 1e-6:
            raise ValueError("Magnitude weights must sum to 1.0")
        
        if abs(tf5_dir_weight + tf15_dir_weight - 1.0) > 1e-6:
            raise ValueError("Direction weights must sum to 1.0")
        
        # Use momentum vector magnitudes (they incorporate both price and volatility)
        tf5_magnitude = tf5_momentum.magnitude
        tf15_magnitude = tf15_momentum.magnitude
        
        # Use momentum vector directions (they account for momentum bias)
        tf5_direction = tf5_momentum.direction
        tf15_direction = tf15_momentum.direction
        
        # Combine magnitudes using weighted average
        combined_magnitude = (tf5_magnitude * tf5_weight) + (tf15_magnitude * tf15_weight)
        
        # Combine directions using weighted average
        combined_direction = (tf5_direction * tf5_dir_weight) + (tf15_direction * tf15_dir_weight)
        
        # Note: signal_strength will be calculated later using percentile ranking
        return CombinedVector(
            tf5_magnitude=tf5_magnitude,
            tf15_magnitude=tf15_magnitude,
            tf5_direction=tf5_direction,
            tf15_direction=tf15_direction,
            combined_magnitude=combined_magnitude,
            combined_direction=combined_direction,
            signal_strength=0.0  # Will be updated by percentile calculation
        )
    
    @staticmethod
    def calculate_percentile_rank(
        current_value: float,
        historical_values: List[float]
    ) -> float:
        """
        Calculate percentile rank of current value in historical data.
        
        Args:
            current_value: Current vector magnitude
            historical_values: List of historical magnitudes (last N values)
            
        Returns:
            Percentile rank (0-100)
            
        Raises:
            ValueError: If historical_values is empty
            
        Example:
            >>> historical = [1.0, 2.0, 3.0, 4.0, 5.0]
            >>> rank = VectorCalculations.calculate_percentile_rank(3.5, historical)
            >>> print(f"Percentile rank: {rank}")  # Should be around 70
        """
        if not historical_values:
            raise ValueError("Historical values cannot be empty")
        
        # Add current value to the dataset
        all_values = historical_values + [current_value]
        all_values.sort()
        
        # Find the rank of current value (handle duplicates)
        rank = 0
        for value in all_values:
            if value < current_value:
                rank += 1
            elif value == current_value:
                break
        
        # Convert to percentile (0-100)
        percentile = (rank / (len(all_values) - 1)) * 100
        
        return percentile
    
    @staticmethod
    def detect_divergence(
        current_data: List[OHLCVData],
        comparison_data: List[OHLCVData],
        period: int = 5
    ) -> DivergenceSignal:
        """
        Detect price vs momentum divergence between two periods.
        
        Divergence Detection:
        - Compare two 5-candle periods for price and momentum trends
        - Bullish divergence: price down, momentum up
        - Bearish divergence: price up, momentum down
        
        Args:
            current_data: Most recent period data
            comparison_data: Earlier period data for comparison
            period: Number of candles to compare (default 5)
            
        Returns:
            DivergenceSignal with divergence analysis
            
        Raises:
            ValueError: If insufficient data
            
        Example:
            >>> current = recent_5_candles
            >>> previous = earlier_5_candles  
            >>> divergence = VectorCalculations.detect_divergence(current, previous)
            >>> if divergence.is_bullish_divergence:
            ...     print("Bullish divergence detected!")
        """
        if len(current_data) < period or len(comparison_data) < period:
            raise ValueError(f"Need at least {period} candles in each dataset")
        
        # Calculate vectors for both periods
        current_momentum = VectorCalculations.calculate_momentum_vector(current_data, period)
        comparison_momentum = VectorCalculations.calculate_momentum_vector(comparison_data, period)
        
        # Calculate overall price trend between the two periods
        current_end = current_data[-1].close
        comparison_end = comparison_data[-1].close
        
        # Calculate momentum trends (magnitude change)
        momentum_trend = (current_momentum.magnitude - comparison_momentum.magnitude) / max(comparison_momentum.magnitude, 0.0001)
        
        # Calculate overall price trend between the two periods
        overall_price_trend = (current_end - comparison_end) / comparison_end
        
        # Detect divergences (using thresholds to avoid noise)
        divergence_threshold = 0.001  # 0.1% threshold
        
        is_bullish_divergence = (
            overall_price_trend < -divergence_threshold and 
            momentum_trend > divergence_threshold
        )
        
        is_bearish_divergence = (
            overall_price_trend > divergence_threshold and 
            momentum_trend < -divergence_threshold
        )
        
        # Calculate divergence strength
        if is_bullish_divergence or is_bearish_divergence:
            divergence_strength = min(abs(overall_price_trend) + abs(momentum_trend), 1.0)
        else:
            divergence_strength = 0.0
        
        return DivergenceSignal(
            price_trend=overall_price_trend,
            momentum_trend=momentum_trend,
            is_bullish_divergence=is_bullish_divergence,
            is_bearish_divergence=is_bearish_divergence,
            divergence_strength=divergence_strength
        )