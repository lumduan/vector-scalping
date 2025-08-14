# Vector Scalping Mathematical Formula Corrections

This document explains the corrections made to the vector calculation mathematical formulas in the vector scalping trading strategy.

## Summary of Changes

The mathematical formulas have been corrected to properly implement the vector analysis as described in the strategy documentation. The key changes involve calculating price vectors and momentum vectors using individual price differences rather than overall displacement.

## 1. Price Vector Calculation - CORRECTED

### Previous (Incorrect) Formula:
```
displacement = close[n] - close[0]
magnitude = abs(displacement)
direction = displacement / price_range
```

### New (Correct) Formula:
```
Step 1: Calculate individual price differences (displacement vector)
Δ₁ = P₂ - P₁
Δ₂ = P₃ - P₂  
Δ₃ = P₄ - P₃
Δ₄ = P₅ - P₄

Step 2: Vector Magnitude (Strength)
Magnitude = √(Δ₁² + Δ₂² + Δ₃² + Δ₄²)

Step 3: Vector Direction (Average Movement)
Direction = (Δ₁ + Δ₂ + Δ₃ + Δ₄) ÷ 4
```

### Example with EUR/USD data [1.1050, 1.1055, 1.1052, 1.1058, 1.1062]:
```
Δ₁ = 1.1055 - 1.1050 = 0.0005
Δ₂ = 1.1052 - 1.1055 = -0.0003
Δ₃ = 1.1058 - 1.1052 = 0.0006  
Δ₄ = 1.1062 - 1.1058 = 0.0004

Magnitude = √(0.0005² + (-0.0003)² + 0.0006² + 0.0004²) = 0.000927
Direction = (0.0005 + (-0.0003) + 0.0006 + 0.0004) ÷ 4 = 0.0003
```

## 2. Momentum Vector Calculation - CORRECTED

### Previous (Incorrect) Formula:
```
price_momentum = (close[n] - close[0]) / n
volatility = average_true_range
magnitude = sqrt(price_momentum² + volatility²)
direction = price_momentum / magnitude
```

### New (Correct) Formula:
```
Step 1: Price Momentum Components
PM₁ = C₂ - C₁
PM₂ = C₃ - C₂
PM₃ = C₄ - C₃  
PM₄ = C₅ - C₄

Step 2: Volatility Components
V₁ = H₁ - L₁
V₂ = H₂ - L₂
V₃ = H₃ - L₃
V₄ = H₄ - L₄
V₅ = H₅ - L₅

Step 3: Momentum Vector Magnitude
Magnitude = √[(PM₁² + PM₂² + PM₃² + PM₄²) + (V₁² + V₂² + V₃² + V₄² + V₅²)]

Step 4: Momentum Direction
Direction = (PM₁ + PM₂ + PM₃ + PM₄) ÷ 4
```

## 3. Signal Threshold Adjustments

Due to the corrected mathematical formulas, the direction values are now on a different scale. The thresholds have been adjusted accordingly:

### Previous Thresholds:
- `direction_threshold`: 0.3
- `tf5_direction`: > 0.2 (LONG), < -0.2 (SHORT)
- `tf15_direction`: > 0.1 (LONG), < -0.1 (SHORT)

### New Thresholds:
- `direction_threshold`: 0.0005 (default)
- `tf5_direction`: > 0.0002 (LONG), < -0.0002 (SHORT)
- `tf15_direction`: > 0.0001 (LONG), < -0.0001 (SHORT)

## 4. Multi-Timeframe Vector Combination - UNCHANGED

The multi-timeframe combination formula remains correct:
```
Combined Direction = (0.7 × Direction₅ₘᵢₙ) + (0.3 × Direction₁₅ₘᵢₙ)
Combined Magnitude = (0.7 × Magnitude₅ₘᵢₙ) + (0.3 × Magnitude₁₅ₘᵢₙ)
```

## 5. Divergence Detection - UPDATED

The divergence detection has been updated to use the corrected direction calculations:

### Formula:
```
Bullish Divergence:
- Direction₂ > Direction₁ AND Momentum Direction₂ < Momentum Direction₁
- AND |Direction₂ - Direction₁| > 0.0005

Bearish Divergence:
- Direction₂ < Direction₁ AND Momentum Direction₂ > Momentum Direction₁  
- AND |Direction₂ - Direction₁| > 0.0005
```

## 6. Impact on Trading Signals

### Entry Signal Conditions (Updated):

**LONG Entry:**
- combined_direction > 0.0005 (or configured threshold)
- signal_strength > 60 (above 60th percentile)
- tf5_direction > 0.0002
- tf15_direction > 0.0001

**SHORT Entry:**
- combined_direction < -0.0005 (or configured threshold)
- signal_strength > 60 (above 60th percentile)
- tf5_direction < -0.0002
- tf15_direction < -0.0001

## 7. Code Changes Made

### Files Modified:
1. `vector_scalping/calculations.py`:
   - Updated `calculate_price_vector()` method
   - Updated `calculate_momentum_vector()` method
   - Updated `detect_divergence()` method

2. `vector_scalping/signals.py`:
   - Adjusted hardcoded thresholds for signal generation
   - Updated comments to reflect new threshold values

3. `vector_scalping/models.py`:
   - Updated default `direction_threshold` from 0.3 to 0.0005

4. `tests/test_calculations.py`:
   - Updated tests to match corrected mathematical formulas

5. `tests/test_signals.py`:
   - Updated test configuration with appropriate thresholds

### New Test Files:
1. `debug/test_math_formulas.py`: Validates the corrected formulas with the EUR/USD example
2. `debug/analyze_signal_test_data.py`: Analyzes test data to understand direction values

## 8. Validation

The corrected formulas have been validated with:
- The provided EUR/USD example data
- Unit tests for all calculation methods
- Integration tests for signal generation
- Complete test suite passes (49/49 tests)

## 9. Recommendations for Production Use

1. **Review Configuration**: Update your strategy configuration to use the new default `direction_threshold` of 0.0005, or adjust based on your risk tolerance.

2. **Backtesting**: Re-run backtests with the corrected formulas to validate performance.

3. **Threshold Tuning**: The new thresholds (0.0002, 0.0001) are conservative. You may want to adjust them based on:
   - Market volatility
   - Symbol characteristics (EUR/USD vs JPY pairs)
   - Historical performance data

4. **Monitoring**: Monitor signal generation frequency with the new thresholds to ensure they generate appropriate trading opportunities.

## 10. Mathematical Accuracy

The corrected formulas now properly implement:
- ✅ Individual price difference calculations
- ✅ Vector magnitude using square root of sum of squares
- ✅ Direction as average of individual components
- ✅ Momentum vector incorporating both price momentum and volatility components
- ✅ Proper divergence detection between periods
- ✅ Appropriate threshold scaling for the new direction values

This ensures the strategy follows the mathematical principles of vector analysis as originally intended.
