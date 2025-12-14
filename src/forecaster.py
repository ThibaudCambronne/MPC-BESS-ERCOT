from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb 
from sklearn.preprocessing import StandardScaler # Still useful, but less critical
from sklearn.pipeline import Pipeline

from src.globals import FREQUENCY, PRICE_NODE, TIME_STEPS_PER_HOUR

from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Import the Huber Regressor for robust regression
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Assuming these globals are defined in src.globals
# from src.globals import FREQUENCY, PRICE_NODE, TIME_STEPS_PER_HOUR

# --- Helper Functions for Enhancements ---

def _encode_cyclical_feature(df: pd.DataFrame, column: str, max_val: int) -> pd.DataFrame:
    """Encodes a cyclical feature using sine and cosine transformations."""
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_val)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_val)
    return df

def get_improved_regression_forecast(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    market: Literal["DA", "RT"],
    price_node: str = PRICE_NODE,
    training_days: int = 15,
    verbose: bool = False,
    lag_hours: int = 1, 
) -> pd.Series:
    """
    Generate price forecast using the highly effective, non-linear XGBoost Regressor.
    
    This model is much better at capturing non-linear relationships and price spikes.
    """
    # --- (Feature setup for price_col, required_cols, etc. remains the same) ---
    if market == "DA":
        price_col = f"{price_node}_DAM"
    elif market == "RT":
        price_col = price_node
    else:
        raise ValueError(f"Unknown market: {market}")
        
    if price_col not in data.columns:
        raise ValueError(f"Price column '{price_col}' not found in data.")
    
    required_cols = ['dew_point_temperature_S', price_col]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Missing required columns in data: {', '.join([c for c in required_cols if c not in data.columns])}")

    # Ensure data is indexed by datetime and sorted
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()
        
    # Forward fill missing values
    data = data.ffill()
    
    # Define training period
    training_start = current_time - pd.Timedelta(days=training_days)
    training_end = current_time
    
    # Extract historical data
    historical_data = data.loc[:current_time].copy()
    
    # Create the lagged price feature
    lag_steps = lag_hours * TIME_STEPS_PER_HOUR
    historical_data['lagged_price'] = historical_data[price_col].shift(lag_steps)

    # Add time features
    historical_data['hour'] = historical_data.index.hour
    historical_data['day_of_week'] = historical_data.index.dayofweek
    
    # Apply cyclical encoding (Still useful for tree-based models, but less critical)
    historical_data = _encode_cyclical_feature(historical_data, 'hour', 24)
    historical_data = _encode_cyclical_feature(historical_data, 'day_of_week', 7)
    
    # Extract training data
    training_mask = (historical_data.index >= training_start) & (historical_data.index < training_end)
    training_data = historical_data.loc[training_mask].copy()

    feature_cols = [
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'dew_point_temperature_S',
        'lagged_price'
    ]
    
    X_train = training_data[feature_cols].values
    y_train = training_data[price_col].values
    
    # Remove rows with NaN values
    valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    if len(X_train) == 0:
        raise ValueError("No valid training data after feature engineering and NaN removal.")

    # --- Step 2: Model Training (XGBoost Regressor) ---
    
    # Initialize XGBoost Regressor
    # objective='reg:squarederror' is the default for regression.
    # n_estimators, max_depth, learning_rate are good starting hyperparams.
    # The 'gbtree' booster is excellent for non-linear prediction.
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.1,
        random_state=42,
        tree_method='hist' # Faster training
    )
    
    # Fit the model
    model.fit(X_train, y_train)

    # --- Step 3: Generating Forecasts (Iterative Prediction for Lagged Feature) ---
    
    forecast_index = pd.date_range(
        start=current_time,
        periods=horizon_hours * TIME_STEPS_PER_HOUR,
        freq=FREQUENCY,
    )
    
    forecast_values = []
    
    # The prices needed for the lag feature: last 'lag_steps' known historical prices.
    lag_steps_needed = lag_hours * TIME_STEPS_PER_HOUR
    price_tracker_hist = historical_data.loc[historical_data.index < current_time, price_col].iloc[-lag_steps_needed:].tolist()
    
    # Combine history and the new predictions to form the rolling lag feature
    price_tracker = price_tracker_hist
    
    for i, timestamp in enumerate(forecast_index):
        
        # 1. Prepare exogenous features (time, weather)
        temp_df = pd.DataFrame([{'hour': timestamp.hour, 'day_of_week': timestamp.dayofweek}], index=[timestamp])
        temp_df = _encode_cyclical_feature(temp_df, 'hour', 24)
        temp_df = _encode_cyclical_feature(temp_df, 'day_of_week', 7)

        # Get dew point (using last known if future is unavailable)
        dew_point = data.loc[timestamp, 'dew_point_temperature_S'] if timestamp in data.index else data['dew_point_temperature_S'].iloc[-1]
        
        # 2. Determine the lagged price feature (Crucial for iterative forecasting)
        # The required lagged price is located at index 'i' of the price_tracker
        # Note: This is simplified and assumes a perfect match between time steps.
        try:
            lagged_price = price_tracker[i]
        except IndexError:
            # Should not happen if history is set correctly, but as a fallback
            lagged_price = price_tracker[-1] 
                
        # 3. Create the feature vector for prediction
        X_i = np.array([
            temp_df['hour_sin'].iloc[0], temp_df['hour_cos'].iloc[0],
            temp_df['day_of_week_sin'].iloc[0], temp_df['day_of_week_cos'].iloc[0],
            dew_point,
            lagged_price
        ]).reshape(1, -1)
        
        # 4. Predict the price
        predicted_price = model.predict(X_i)[0]
        forecast_values.append(predicted_price)
        
        # 5. Update the price_tracker with the new predicted price for future lags
        price_tracker.append(predicted_price)

    # --- (Final Series Creation and Plotting logic remains the same) ---
    forecast = pd.Series(
        forecast_values,
        index=forecast_index,
        name=f"{market}_xgboost_regression_forecast"
    )

    if verbose:
        # Historical window for plotting
        hist_start = current_time - pd.Timedelta(hours=horizon_hours)
        historical = data.loc[
            (data.index >= hist_start) & (data.index <= current_time), price_col
        ]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(
            historical.index, historical.values, label="Historical", color="tab:blue"
        )
        plt.plot(forecast.index, forecast.values, label="XGBoost Forecast", color="tab:red")
        plt.axvline(current_time, color="k", linestyle="--", label="Current Time")
        plt.xlabel("Time")
        plt.ylabel("Price ($/MWh)")
        plt.title(f"{market} Price Forecast - XGBoost Regression\n(Trained on {training_days} days)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return forecast

# --- Update the original function to use the improved one ---
# This block replaces the old logic in your original file

def get_forecast(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    market: Literal["DA", "RT"],
    method: Literal["persistence", "perfect", "regression"],
    price_node: str = PRICE_NODE,
    verbose: bool = False,
) -> pd.Series:
    """
    Generate price forecast for the specified market.
    Returns a dummy time series with the correct number of elements and a time index.
    """
    # Create a time index starting from current_time, with hourly frequency
    time_index = pd.date_range(
        start=current_time,
        periods=horizon_hours * TIME_STEPS_PER_HOUR,
        freq=FREQUENCY,
    )

    # Determine the correct price column based on market and price_node
    if market == "DA":
        price_col = f"{price_node}_DAM"
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data.")
    elif market == "RT":
        # RTM data uses just the price_node name without suffix
        price_col = price_node
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data.")
    else:
        raise ValueError(f"Unknown market: {market}")

    if method == "persistence":
        # For each forecast timestamp, use the price from the previous day at the same time
        prev_day_times = pd.date_range(
            start=current_time - pd.Timedelta(days=1),
            periods=24 * TIME_STEPS_PER_HOUR,
            freq=FREQUENCY,
        )
        # Check if all previous day times exist in the data
        missing = prev_day_times.difference(data.index)
        if not missing.empty:
            raise ValueError(
                f"Missing historical data for persistence forecast at: {list(missing)}"
            )
        forecast_values = data.loc[prev_day_times, price_col].values
        # At this point, the forecast length is only 24 hours worth of data
        # We need to repeat this data to cover the entire horizon_hours
        repeats = (horizon_hours * TIME_STEPS_PER_HOUR) // (24 * TIME_STEPS_PER_HOUR)
        remainder = (horizon_hours * TIME_STEPS_PER_HOUR) % (24 * TIME_STEPS_PER_HOUR)
        forecast_values = list(forecast_values) * repeats + list(
            forecast_values[:remainder]
        )
        forecast = pd.Series(
            forecast_values, index=time_index, name=f"{market}_forecast"
        )
    elif method == "perfect":
        # Use the actual future prices from the data
        # Ensure data is indexed by datetime
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()

        # Check that the forecast window exists in the data
        missing = time_index.difference(data.index)
        if not missing.empty:
            raise ValueError(f"Missing data for perfect forecast at: {list(missing)}")
        # Get the prices for the forecast window
        forecast = data.loc[time_index, price_col].copy()
        forecast.name = f"{market}_forecast"
    elif method == "regression":
        # Use the actual future prices from the data
        # Ensure data is indexed by datetime
        if not data.index.is_monotonic_increasing:
            data = data.sort_index()
        # Get the prices for the forecast window
        forecast = get_improved_regression_forecast(
            data=data,
            current_time=current_time,
            horizon_hours=horizon_hours,
            market=market,
            price_node=price_node,
        )
        forecast.name = f"{market}_forecast"
    else:
        raise ValueError(f"Unknown forecast method: {method}")

    if verbose:
        # Historical window: current_time - horizon_hours to current_time
        hist_start = current_time - pd.Timedelta(hours=horizon_hours)
        hist_end = current_time + pd.Timedelta(hours=horizon_hours)
        # Select historical data
        historical = data.loc[
            (data.index >= hist_start) & (data.index <= current_time), price_col
        ]
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(
            historical.index, historical.values, label="Historical", color="tab:blue"
        )
        plt.plot(forecast.index, forecast.values, label="Forecast", color="tab:orange")
        plt.axvline(current_time, color="k", linestyle="--", label="Current Time")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title(f"{market} Price Forecast ({method})")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return forecast


def get_regression_forecast(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    market: Literal["DA", "RT"],
    price_node: str = PRICE_NODE,
    training_days: int = 15,
    verbose: bool = False,
) -> pd.Series:
    """
    Generate price forecast using linear regression based on time features and dew point temperature.
    
    Uses the last 'training_days' days of data to train a regression model with features:
    - Hour of day (0-23)
    - Day of week (0-6, where 0=Monday)
    - Dew point temperature (dew_point_temperature_S)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Historical data with datetime index
    current_time : pd.Timestamp
        Current time from which to forecast
    horizon_hours : int
        Number of hours to forecast ahead
    market : Literal["DA", "RT"]
        Market type (Day-Ahead or Real-Time)
    price_node : str
        Price node to forecast (default: PRICE_NODE from globals)
    training_days : int
        Number of days of historical data to use for training (default: 15)
    verbose : bool
        If True, plot the forecast vs historical data
        
    Returns:
    --------
    pd.Series
        Forecasted prices with datetime index
    """
    # Determine the correct price column based on market and price_node
    if market == "DA":
        price_col = f"{price_node}_DAM"
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data.")
    elif market == "RT":
        # RTM data uses just the price_node name without suffix
        price_col = price_node
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data.")
    else:
        raise ValueError(f"Unknown market: {market}")
    
    # Check if dew_point_temperature_S column exists
    if "dew_point_temperature_S" not in data.columns:
        raise ValueError("Column 'dew_point_temperature_S' not found in data.")
    
    # Just forward fill dataframe 
    data = data.ffill()

    # Define training period: last 'training_days' days before current_time
    training_start = current_time - pd.Timedelta(days=training_days)
    training_end = current_time
    
    # Extract training data
    training_mask = (data.index >= training_start) & (data.index < training_end)
    training_data = data.loc[training_mask].copy()
    
    if len(training_data) == 0:
        raise ValueError(
            f"No training data available between {training_start} and {training_end}"
        )
    
    # Create features for training data
    training_data['hour'] = training_data.index.hour
    training_data['day_of_week'] = training_data.index.dayofweek
    
    # Prepare training features and target
    feature_cols = ['hour', 'day_of_week', 'dew_point_temperature_S']
    X_train = training_data[feature_cols].values
    y_train = training_data[price_col].values
    
    # Remove any rows with NaN values
    valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    if len(X_train) == 0:
        raise ValueError("No valid training data after removing NaN values")
    
    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Create forecast time index
    forecast_index = pd.date_range(
        start=current_time,
        periods=horizon_hours * TIME_STEPS_PER_HOUR,
        freq=FREQUENCY,
    )
    
    # Prepare forecast features
    # For future timestamps, we need to get dew_point_temperature_S from the data
    # If future data is available (for testing), use it; otherwise use last known value
    forecast_features = []
    for timestamp in forecast_index:
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        # Try to get actual dew point temperature if available in data
        if timestamp in data.index:
            dew_point = data.loc[timestamp, 'dew_point_temperature_S']
        else:
            # Use the last known dew point temperature (persistence assumption)
            dew_point = data.loc[current_time - pd.Timedelta(minutes=15), 'dew_point_temperature_S']
        
        forecast_features.append([hour, day_of_week, dew_point])
    
    X_forecast = np.array(forecast_features)
    
    # Generate predictions
    forecast_values = model.predict(X_forecast)
    
    # Create forecast series
    forecast = pd.Series(
        forecast_values,
        index=forecast_index,
        name=f"{market}_regression_forecast"
    )
    
    if verbose:
        # Historical window for plotting
        hist_start = current_time - pd.Timedelta(hours=horizon_hours)
        historical = data.loc[
            (data.index >= hist_start) & (data.index <= current_time), price_col
        ]
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(
            historical.index, historical.values, label="Historical", color="tab:blue"
        )
        plt.plot(forecast.index, forecast.values, label="Regression Forecast", color="tab:green")
        plt.axvline(current_time, color="k", linestyle="--", label="Current Time")
        plt.xlabel("Time")
        plt.ylabel("Price ($/MWh)")
        plt.title(f"{market} Price Forecast - Linear Regression\n(Trained on {training_days} days)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return forecast


def get_forecasts_for_da(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    method: Literal["persistence", "perfect", "regression"],
    price_node: str = PRICE_NODE,
    verbose: bool = False,
) -> tuple[pd.Series, pd.Series]:
    assert current_time.minute == 0 and current_time.hour == 10, (
        f"For the day ahead forecast, the current time must be at 10:00 AM. Got {current_time} instead."
    )

    da_prices = get_forecast(
        data,
        current_time=current_time,
        horizon_hours=horizon_hours + 14,
        market="DA",
        method=method, # <--- **This is already correct.**
        price_node=price_node,
        verbose=verbose,
    )[-TIME_STEPS_PER_HOUR * horizon_hours :]
    rt_prices = get_forecast(
        data,
        current_time=current_time,
        horizon_hours=horizon_hours + 14,
        market="RT",
        method=method, # <--- **This is already correct.**
        price_node=price_node,
        verbose=verbose,
    )[-TIME_STEPS_PER_HOUR * horizon_hours :]
    return da_prices, rt_prices