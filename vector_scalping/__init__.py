"""Vector Scalping Trading Strategy Package."""

__version__ = "0.1.0"
__author__ = "Vector Scalping Strategy"
__license__ = "MIT"

from .models import (
    OHLCVData,
    PriceVector,
    MomentumVector,
    CombinedVector,
    TradingSignal,
    RiskManagement,
    StrategyConfig,
    TimeFrame,
    SignalType,
)
from .calculations import VectorCalculations
from .signals import SignalGenerator
from .data_service import DataService

__all__ = [
    "OHLCVData",
    "PriceVector", 
    "MomentumVector",
    "CombinedVector",
    "TradingSignal",
    "RiskManagement",
    "StrategyConfig",
    "TimeFrame",
    "SignalType",
    "VectorCalculations",
    "SignalGenerator",
    "DataService",
]