#!/usr/bin/env python3
"""Test script to verify tvkit library functionality."""

import asyncio
from tvkit.api.chart.ohlcv import OHLCV

async def test_tvkit():
    """Test basic tvkit functionality."""
    try:
        # Test fetching EUR/USD 5-minute data
        print("Testing EUR/USD 5-minute historical data fetch...")
        
        async with OHLCV() as client:
            # Test 5-minute data
            data_5m = await client.get_historical_ohlcv(
                exchange_symbol='FX_IDC:EURUSD',
                interval='5',
                bars_count=20
            )
            
            print(f"5-minute data: {len(data_5m)} bars received")
            print("Sample 5-minute data:")
            for i, bar in enumerate(data_5m[:3]):
                print(f"  Bar {i+1}: O={bar.open}, H={bar.high}, L={bar.low}, C={bar.close}, V={bar.volume}")
            
            # Test 15-minute data  
            print("\nTesting EUR/USD 15-minute historical data fetch...")
            data_15m = await client.get_historical_ohlcv(
                exchange_symbol='FX_IDC:EURUSD',
                interval='15',
                bars_count=10
            )
            
            print(f"15-minute data: {len(data_15m)} bars received")
            print("Sample 15-minute data:")
            for i, bar in enumerate(data_15m[:3]):
                print(f"  Bar {i+1}: O={bar.open}, H={bar.high}, L={bar.low}, C={bar.close}, V={bar.volume}")
        
        return True
        
    except Exception as e:
        print(f"Error testing tvkit: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tvkit())
    print(f"tvkit test {'passed' if success else 'failed'}")