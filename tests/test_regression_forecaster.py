"""
Test script for the regression-based forecaster
"""
import pandas as pd
from src.forecaster import get_regression_forecast, get_improved_regression_forecast
from src.globals import DATA_PATH_RTM

# Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH_RTM, parse_dates=["Time"], index_col="Time")
print(f"Data loaded: {len(data)} rows from {data.index.min()} to {data.index.max()}")

# Test parameters
current_time = pd.Timestamp("2020-02-01 12:00:00")
horizon_hours = 24
market = "RT"
training_days = 15

print(f"\nTesting regression forecaster:")
print(f"  Current time: {current_time}")
print(f"  Forecast horizon: {horizon_hours} hours")
print(f"  Market: {market}")
print(f"  Training days: {training_days}")

# Generate forecast
try:
    forecast = get_improved_regression_forecast(
        data=data,
        current_time=current_time,
        horizon_hours=horizon_hours,
        market=market,
        training_days=training_days,
        verbose=True
    )
    
    print(f"\nForecast generated successfully!")
    print(f"  Forecast length: {len(forecast)} time steps")
    print(f"  Forecast range: {forecast.index.min()} to {forecast.index.max()}")
    print(f"  Price range: ${forecast.min():.2f} to ${forecast.max():.2f} per MWh")
    print(f"  Mean price: ${forecast.mean():.2f} per MWh")
    
    # Show first few values
    print(f"\nFirst 5 forecast values:")
    print(forecast.head())
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
