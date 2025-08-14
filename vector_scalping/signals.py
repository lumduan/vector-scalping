"""Signal generation logic for vector scalping strategy."""

import time
from typing import List, Optional

from .models import (
    OHLCVData, 
    SignalType, 
    TradingSignal, 
    CombinedVector, 
    DivergenceSignal,
    StrategyConfig
)
from .calculations import VectorCalculations


class SignalGenerator:
    """Generates trading signals based on vector analysis."""
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize signal generator with strategy configuration.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.historical_magnitudes: List[float] = []
    
    def generate_signal(
        self,
        tf5_data: List[OHLCVData],
        tf15_data: List[OHLCVData],
        current_price: float
    ) -> TradingSignal:
        """
        Generate trading signal based on vector analysis.
        
        Entry Signal Conditions:
        
        LONG Entry:
        - combined_direction > 0.3 (bullish bias)
        - signal_strength > 60 (above 60th percentile)  
        - tf5_direction > 0.2 (5-min confirmation)
        - tf15_direction > 0.1 (15-min trend alignment)
        
        SHORT Entry:
        - combined_direction < -0.3 (bearish bias)
        - signal_strength > 60 (above 60th percentile)
        - tf5_direction < -0.2 (5-min confirmation) 
        - tf15_direction < -0.1 (15-min trend alignment)
        
        Args:
            tf5_data: 5-minute OHLCV data
            tf15_data: 15-minute OHLCV data  
            current_price: Current market price
            
        Returns:
            TradingSignal with entry/exit decision
            
        Raises:
            ValueError: If insufficient data provided
            
        Example:
            >>> generator = SignalGenerator(config)
            >>> signal = generator.generate_signal(tf5_data, tf15_data, 1.0850)
            >>> if signal.signal_type == SignalType.LONG:
            ...     print(f"LONG signal at {signal.entry_price}")
        """
        if len(tf5_data) < self.config.vector_period:
            raise ValueError(f"Need at least {self.config.vector_period} 5-minute candles")
        
        if len(tf15_data) < self.config.vector_period:
            raise ValueError(f"Need at least {self.config.vector_period} 15-minute candles")
        
        # Calculate vectors for both timeframes
        tf5_price = VectorCalculations.calculate_price_vector(tf5_data, self.config.vector_period)
        tf5_momentum = VectorCalculations.calculate_momentum_vector(tf5_data, self.config.vector_period)
        
        tf15_price = VectorCalculations.calculate_price_vector(tf15_data, self.config.vector_period)
        tf15_momentum = VectorCalculations.calculate_momentum_vector(tf15_data, self.config.vector_period)
        
        # Combine vectors
        combined_vector = VectorCalculations.combine_vectors(
            tf5_price, tf5_momentum, tf15_price, tf15_momentum,
            self.config.tf5_weight, self.config.tf15_weight,
            self.config.tf5_direction_weight, self.config.tf15_direction_weight
        )
        
        # Calculate signal strength using percentile ranking
        signal_strength = self._calculate_signal_strength(combined_vector.combined_magnitude)
        
        # Update combined vector with signal strength
        combined_vector.signal_strength = signal_strength
        
        # Check for divergence (if enough historical data)
        divergence = self._detect_divergence(tf5_data, tf15_data)
        
        # Generate signal based on conditions
        signal_type, reason, confidence = self._evaluate_conditions(combined_vector, divergence)
        
        # Calculate entry levels if signal is valid
        entry_price = None
        take_profit = None
        stop_loss = None
        
        if signal_type in [SignalType.LONG, SignalType.SHORT]:
            entry_price = current_price
            take_profit = self.config.risk_management.calculate_take_profit(
                entry_price, signal_type
            )
            stop_loss = self.config.risk_management.calculate_stop_loss(
                entry_price, signal_type  
            )
        
        return TradingSignal(
            signal_type=signal_type,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            confidence=confidence,
            timestamp=int(time.time()),
            reason=reason,
            vector_data=combined_vector,
            divergence_data=divergence
        )
    
    def _calculate_signal_strength(self, current_magnitude: float) -> float:
        """
        Calculate signal strength using percentile ranking.
        
        Args:
            current_magnitude: Current combined vector magnitude
            
        Returns:
            Signal strength percentile (0-100)
        """
        # Add current magnitude to historical data
        self.historical_magnitudes.append(current_magnitude)
        
        # Keep only recent data for percentile calculation
        if len(self.historical_magnitudes) > self.config.percentile_lookback:
            self.historical_magnitudes = self.historical_magnitudes[-self.config.percentile_lookback:]
        
        # Calculate percentile rank (need at least 10 values for meaningful percentile)
        if len(self.historical_magnitudes) < 10:
            return 50.0  # Default to 50th percentile for insufficient data
        
        return VectorCalculations.calculate_percentile_rank(
            current_magnitude, self.historical_magnitudes[:-1]
        )
    
    def _detect_divergence(
        self, 
        tf5_data: List[OHLCVData], 
        tf15_data: List[OHLCVData]
    ) -> Optional[DivergenceSignal]:
        """
        Detect divergence if sufficient historical data is available.
        
        Args:
            tf5_data: 5-minute OHLCV data
            tf15_data: 15-minute OHLCV data
            
        Returns:
            DivergenceSignal if enough data, None otherwise
        """
        required_length = self.config.vector_period * 2
        
        if len(tf5_data) < required_length:
            return None
        
        # Use 5-minute data for divergence detection (higher resolution)
        current_period = tf5_data[-self.config.vector_period:]
        previous_period = tf5_data[-(self.config.vector_period * 2):-self.config.vector_period]
        
        try:
            return VectorCalculations.detect_divergence(
                current_period, previous_period, self.config.vector_period
            )
        except ValueError:
            return None
    
    def _evaluate_conditions(
        self, 
        combined_vector: CombinedVector, 
        divergence: Optional[DivergenceSignal]
    ) -> tuple[SignalType, str, float]:
        """
        Evaluate entry conditions and determine signal type.
        
        Args:
            combined_vector: Combined vector analysis results
            divergence: Divergence analysis (optional)
            
        Returns:
            Tuple of (signal_type, reason, confidence)
        """
        # Extract vector components
        combined_direction = combined_vector.combined_direction
        signal_strength = combined_vector.signal_strength
        tf5_direction = combined_vector.tf5_direction
        tf15_direction = combined_vector.tf15_direction
        
        # Check minimum signal strength requirement
        if signal_strength < self.config.signal_threshold:
            return (
                SignalType.NO_SIGNAL,
                f"Signal strength {signal_strength:.1f} below threshold {self.config.signal_threshold}",
                0.0
            )
        
        # Check for LONG entry conditions
        if (combined_direction > self.config.direction_threshold and
            tf5_direction > 0.2 and 
            tf15_direction > 0.1):
            
            confidence = self._calculate_confidence(
                combined_vector, divergence, SignalType.LONG
            )
            
            reason = (
                f"LONG: combined_dir={combined_direction:.3f}, "
                f"strength={signal_strength:.1f}, "
                f"tf5_dir={tf5_direction:.3f}, tf15_dir={tf15_direction:.3f}"
            )
            
            if divergence and divergence.is_bullish_divergence:
                reason += f", bullish_divergence={divergence.divergence_strength:.3f}"
                confidence = min(confidence + 0.1, 1.0)  # Boost confidence for divergence
            
            return SignalType.LONG, reason, confidence
        
        # Check for SHORT entry conditions  
        elif (combined_direction < -self.config.direction_threshold and
              tf5_direction < -0.2 and
              tf15_direction < -0.1):
            
            confidence = self._calculate_confidence(
                combined_vector, divergence, SignalType.SHORT
            )
            
            reason = (
                f"SHORT: combined_dir={combined_direction:.3f}, "
                f"strength={signal_strength:.1f}, "
                f"tf5_dir={tf5_direction:.3f}, tf15_dir={tf15_direction:.3f}"
            )
            
            if divergence and divergence.is_bearish_divergence:
                reason += f", bearish_divergence={divergence.divergence_strength:.3f}"
                confidence = min(confidence + 0.1, 1.0)  # Boost confidence for divergence
            
            return SignalType.SHORT, reason, confidence
        
        # No signal conditions met
        return (
            SignalType.NO_SIGNAL,
            f"No entry conditions met: combined_dir={combined_direction:.3f}, "
            f"strength={signal_strength:.1f}",
            0.0
        )
    
    def _calculate_confidence(
        self,
        combined_vector: CombinedVector,
        divergence: Optional[DivergenceSignal],
        signal_type: SignalType
    ) -> float:
        """
        Calculate confidence score for the signal.
        
        Args:
            combined_vector: Combined vector analysis
            divergence: Divergence analysis (optional)
            signal_type: Type of signal being evaluated
            
        Returns:
            Confidence score (0-1)
        """
        base_confidence = 0.5
        
        # Boost confidence based on signal strength
        strength_factor = min((combined_vector.signal_strength - 50) / 50, 0.3)
        base_confidence += strength_factor
        
        # Boost confidence based on direction alignment
        direction_alignment = abs(combined_vector.tf5_direction - combined_vector.tf15_direction)
        alignment_factor = max(0.2 - direction_alignment, 0)
        base_confidence += alignment_factor
        
        # Boost confidence for strong directional bias
        direction_strength = abs(combined_vector.combined_direction)
        if direction_strength > 0.5:
            base_confidence += 0.1
        
        # Additional boost for divergence
        if divergence:
            if ((signal_type == SignalType.LONG and divergence.is_bullish_divergence) or
                (signal_type == SignalType.SHORT and divergence.is_bearish_divergence)):
                base_confidence += divergence.divergence_strength * 0.15
        
        return min(base_confidence, 1.0)
    
    def should_exit_time_based(self) -> bool:
        """
        Check if it's Friday 5 PM GMT for time-based exit.
        
        Returns:
            True if it's time to exit all positions
        """
        from datetime import datetime
        
        # Get current time in GMT
        now_utc = datetime.utcnow()
        
        # Check if it's Friday (weekday 4) and after 17:00 GMT
        is_friday = now_utc.weekday() == 4
        is_after_5pm = now_utc.hour >= 17
        
        return is_friday and is_after_5pm
    
    def reset_historical_data(self) -> None:
        """Reset historical magnitude data (useful for backtesting)."""
        self.historical_magnitudes.clear()