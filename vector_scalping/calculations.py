"""Vector calculation functions for the scalping strategy."""

import math
from typing import List

from .models import (
    CombinedVector,
    DivergenceSignal,
    MomentumVector,
    OHLCVData,
    PriceVector,
)


class VectorCalculations:
    """Core vector calculation methods for trading strategy."""

    @staticmethod
    def calculate_price_vector(
        data: List[OHLCVData],
        period: int = 5
    ) -> PriceVector:
        """
        Calculate price vector from OHLCV data using proper mathematical formula.

        Price Vector Calculation (Following Mathematical Formula):
        Step 1: Calculate individual price differences (displacement vector)
        Δ₁ = P₂ - P₁,
        Δ₂ = P₃ - P₂,
        Δ₃ = P₄ - P₃,
        Δ₄ = P₅ - P₄

        Step 2: Vector Magnitude (Strength)
        Magnitude = √(Δ₁² + Δ₂² + Δ₃² + Δ₄²)

        Step 3: Vector Direction (Average Movement)
        Direction = (Δ₁ + Δ₂ + Δ₃ + Δ₄) ÷ 4

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

        # Step 1: Calculate individual price differences (displacement vector)
        price_differences: List[float] = []
        for i in range(1, len(candles)):
            delta: float = candles[i].close - candles[i-1].close
            price_differences.append(delta)

        # Step 2: Calculate Vector Magnitude using square root of sum of squares
        sum_of_squares: float = sum(delta**2 for delta in price_differences)
        magnitude: float = math.sqrt(sum_of_squares)

        # Step 3: Calculate Vector Direction (average movement)
        direction: float = sum(price_differences) / len(price_differences) if price_differences else 0.0

        # Calculate overall displacement for compatibility
        displacement = end_price - start_price

        # Calculate price range for reference
        all_highs = [candle.high for candle in candles]
        all_lows = [candle.low for candle in candles]
        price_range = max(all_highs) - min(all_lows)

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
        Calculate momentum vector from OHLCV data using proper mathematical formula.

        Momentum Vector Calculation (Following Mathematical Formula):
        Step 1: Price Momentum Components
        PM₁ = C₂ - C₁,
        PM₂ = C₃ - C₂,
        PM₃ = C₄ - C₃,
        PM₄ = C₅ - C₄

        Step 2: Volatility Components
        V₁ = H₁ - L₁,
        V₂ = H₂ - L₂,
        V₃ = H₃ - L₃,
        V₄ = H₄ - L₄,
        V₅ = H₅ - L₅

        Step 3: Momentum Vector Magnitude
        Magnitude = √[(PM₁² + PM₂² + PM₃² + PM₄²) + (V₁² + V₂² + V₃² + V₄² + V₅²)]

        Step 4: Momentum Direction
        Direction = (PM₁ + PM₂ + PM₃ + PM₄) ÷ 4

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

        # Step 1: Calculate Price Momentum Components (PM₁, PM₂, PM₃, PM₄)
        price_momentum_components: List[float] = []
        for i in range(1, len(candles)):
            pm: float = candles[i].close - candles[i-1].close
            price_momentum_components.append(pm)

        # Step 2: Calculate Volatility Components (V₁, V₂, V₃, V₄, V₅)
        volatility_components: List[float] = []
        for candle in candles:
            v: float = candle.high - candle.low
            volatility_components.append(v)

        # Step 3: Calculate Momentum Vector Magnitude
        # Magnitude = √[(PM₁² + PM₂² + PM₃² + PM₄²) + (V₁² + V₂² + V₃² + V₄² + V₅²)]
        pm_sum_squares: float = sum(pm**2 for pm in price_momentum_components)
        v_sum_squares: float = sum(v**2 for v in volatility_components)
        magnitude: float = math.sqrt(pm_sum_squares + v_sum_squares)

        # Step 4: Calculate Momentum Direction
        # Direction = (PM₁ + PM₂ + PM₃ + PM₄) ÷ 4
        direction: float = sum(price_momentum_components) / len(price_momentum_components) if price_momentum_components else 0.0

        # Calculate average volatility for reference (traditional ATR-like calculation)
        volatility: float = sum(volatility_components) / len(volatility_components) if volatility_components else 0.0

        # Calculate average price momentum for reference
        price_momentum: float = sum(price_momentum_components) / len(price_momentum_components) if price_momentum_components else 0.0

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
        Detect price vs momentum divergence between two periods using mathematical formula.

        Vector Divergence Detection (Following Mathematical Formula):
        Compare two 5-candle periods:
        Period 1: Candles 1-5 → Direction₁, Momentum Direction₁
        Period 2: Candles 6-10 → Direction₂, Momentum Direction₂

        Bullish Divergence Signal:
        If: Direction₂ > Direction₁ AND Momentum Direction₂ < Momentum Direction₁
        AND: |Direction₂ - Direction₁| > 0.0005
        Then: Potential LONG entry

        Bearish Divergence Signal:
        If: Direction₂ < Direction₁ AND Momentum Direction₂ > Momentum Direction₁
        AND: |Direction₂ - Direction₁| > 0.0005
        Then: Potential SHORT entry

        Args:
            current_data: Most recent period data (Period 2)
            comparison_data: Earlier period data for comparison (Period 1)
            period: Number of candles to compare (default 5)

        Returns:
            DivergenceSignal with divergence analysis

        Raises:
            ValueError: If insufficient data

        Example:
            >>> current = recent_5_candles    # Period 2 (candles 6-10)
            >>> previous = earlier_5_candles  # Period 1 (candles 1-5)
            >>> divergence = VectorCalculations.detect_divergence(current, previous)
            >>> if divergence.is_bullish_divergence:
            ...     print("Bullish divergence detected!")
        """
        if len(current_data) < period or len(comparison_data) < period:
            raise ValueError(f"Need at least {period} candles in each dataset")

        # Calculate price vectors for both periods
        current_price_vector = VectorCalculations.calculate_price_vector(current_data, period)
        comparison_price_vector = VectorCalculations.calculate_price_vector(comparison_data, period)

        # Calculate momentum vectors for both periods
        current_momentum_vector = VectorCalculations.calculate_momentum_vector(current_data, period)
        comparison_momentum_vector = VectorCalculations.calculate_momentum_vector(comparison_data, period)

        # Get direction values (Period 2 and Period 1)
        direction_2: float = current_price_vector.direction
        direction_1: float = comparison_price_vector.direction
        momentum_direction_2: float = current_momentum_vector.direction
        momentum_direction_1: float = comparison_momentum_vector.direction

        # Calculate direction differences
        direction_diff: float = direction_2 - direction_1
        momentum_direction_diff: float = momentum_direction_2 - momentum_direction_1

        # Divergence threshold (0.0005 as per formula)
        divergence_threshold: float = 0.0005

        # Bullish Divergence: Direction₂ > Direction₁ AND Momentum Direction₂ < Momentum Direction₁
        # AND |Direction₂ - Direction₁| > 0.0005
        is_bullish_divergence: bool = (
            direction_2 > direction_1 and
            momentum_direction_2 < momentum_direction_1 and
            abs(direction_diff) > divergence_threshold
        )

        # Bearish Divergence: Direction₂ < Direction₁ AND Momentum Direction₂ > Momentum Direction₁
        # AND |Direction₂ - Direction₁| > 0.0005
        is_bearish_divergence: bool = (
            direction_2 < direction_1 and
            momentum_direction_2 > momentum_direction_1 and
            abs(direction_diff) > divergence_threshold
        )

        # Calculate divergence strength based on the magnitude of differences
        if is_bullish_divergence or is_bearish_divergence:
            divergence_strength: float = min(abs(direction_diff) + abs(momentum_direction_diff), 1.0)
        else:
            divergence_strength = 0.0

        return DivergenceSignal(
            price_trend=direction_diff,
            momentum_trend=momentum_direction_diff,
            is_bullish_divergence=is_bullish_divergence,
            is_bearish_divergence=is_bearish_divergence,
            divergence_strength=divergence_strength
        )
