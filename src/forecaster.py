from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression

from src.globals import FREQUENCY, PRICE_NODE, TIME_STEPS_PER_HOUR, WEATHER_FEATURES


def _add_cyclical_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encodes cyclical time features (hour and day of week) using sine and cosine transformations."""
    time_in_minutes = df.index.day * 24 * 60 + df.index.hour * 60 + df.index.minute
    # time of the hour
    df["minute_sin"] = np.sin(2 * np.pi * time_in_minutes / 60)
    df["minute_cos"] = np.cos(2 * np.pi * time_in_minutes / 60)

    # Time of the day
    df["hour_sin"] = np.sin(2 * np.pi * time_in_minutes / (24 * 60))
    df["hour_cos"] = np.cos(2 * np.pi * time_in_minutes / (24 * 60))

    # Day of the week
    df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    time_features_cols = [
        "minute_sin",
        "minute_cos",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]

    return df, time_features_cols


def _prepare_training_data(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    price_col: str,
    training_days: int,
    number_of_lags: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare training data for forecasting models.

    This function encapsulates the data preparation logic used by both the XGBoost
    and regression forecasting methods. It creates features including:
    - Cyclical encoding for hour and day of week
    - Weather features
    - Lagged price features

    Returns:
        X_train: Training features
        y_train: Training target
        feature_cols: List of feature column names
    """
    required_cols = WEATHER_FEATURES + [price_col]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(
            f"Missing required columns in data: {', '.join([c for c in required_cols if c not in data.columns])}"
        )

    # Define training period
    training_start = current_time - pd.Timedelta(days=training_days)
    training_end = current_time

    historical_data = data.loc[:current_time].copy()

    # Create the lagged price features
    lag_cols = []
    lagged_data = []
    for lag in range(1, number_of_lags + 1):
        lag_col_name = f"lagged_price_{lag}"
        lag_cols.append(lag_col_name)
        lagged_data.append(historical_data[price_col].shift(lag).rename(lag_col_name))

    # Concatenate all lagged columns at once
    historical_data = pd.concat([historical_data] + lagged_data, axis=1)

    training_data = historical_data.loc[
        (historical_data.index >= training_start)
        & (historical_data.index < training_end)
    ].copy()

    # Apply cyclical encoding
    training_data, time_features_cols = _add_cyclical_time_features(training_data)

    if verbose:
        fig, ax = plt.subplots(figsize=(10, 4))
        for time_feature in time_features_cols:
            ax.plot(
                training_data.index,
                training_data[time_feature],
                label=time_feature,
            )
        ax.set_title("Time Features")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        fig.show()

    # Define feature columns
    all_feature_cols = time_features_cols + WEATHER_FEATURES + lag_cols

    # Prepare training features and target
    X_train = training_data[all_feature_cols].values
    y_train = training_data[price_col].values

    # Remove rows with NaN values
    valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    if len(X_train) == 0:
        raise ValueError(
            "No valid training data after feature engineering and NaN removal."
        )

    return X_train, y_train, all_feature_cols


def _generate_forecast(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    price_col: str,
    model,
    number_of_lags: int,
) -> pd.Series:
    """
    Generate price forecast using a trained model.

    This unified function handles the forecasting pipeline for both XGBoost and regression models.
    It uses the iterative prediction approach from the XGBoost implementation.
    """
    forecast_index = pd.date_range(
        start=current_time,
        periods=horizon_hours * TIME_STEPS_PER_HOUR,
        freq=FREQUENCY,
    )

    forecast_values = []

    # The prices needed for the lag feature: last 'number_of_lags' known historical prices.
    lag_steps_needed = number_of_lags
    historical_data = data.loc[:current_time].copy()
    price_tracker = historical_data.loc[
        historical_data.index < current_time, price_col
    ].iloc[-lag_steps_needed:]

    for i, timestamp in enumerate(forecast_index):
        # 1. Prepare exogenous features (time, weather)
        temp_df = pd.DataFrame(
            [{"hour": timestamp.hour, "day_of_week": timestamp.dayofweek}],
            index=[timestamp],
        )
        temp_df, time_features_cols = _add_cyclical_time_features(temp_df)

        # Get weather features
        weather_features = data.loc[timestamp, WEATHER_FEATURES]

        # 2. Create the feature vector for prediction
        X_i = np.array(
            temp_df[time_features_cols].iloc[0].to_list()
            + weather_features.to_list()
            + price_tracker[
                ::-1
            ].to_list()  # reverse the price list order, to get the most recent price first
        ).reshape(1, -1)

        # 3. Predict the price
        predicted_price = model.predict(X_i)[0]
        forecast_values.append(predicted_price)

        # 4. Update the price_tracker with the new predicted price for future lags
        price_tracker.loc[timestamp] = predicted_price

        # 5. remove the oldest price to maintain the lag window size
        price_tracker = price_tracker.iloc[1:]

    # Create forecast series
    forecast = pd.Series(
        forecast_values,
        index=forecast_index,
    )
    return forecast


def _train_regression_model(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    price_col: str,
    training_days: int,
    number_of_lags: int,
):
    """
    Train a linear regression model for price forecasting.
    """

    # Prepare training data
    X_train, y_train, feature_cols = _prepare_training_data(
        data=data,
        current_time=current_time,
        price_col=price_col,
        training_days=training_days,
        number_of_lags=number_of_lags,
    )

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def _train_xgboost_model(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    price_col: str,
    training_days: int,
    number_of_lags: int,
):
    """
    Train an XGBoost model for price forecasting.
    """

    # Prepare training data
    X_train, y_train, feature_cols = _prepare_training_data(
        data=data,
        current_time=current_time,
        price_col=price_col,
        training_days=training_days,
        number_of_lags=number_of_lags,
    )

    # Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        tree_method="hist",  # Faster training
    )

    # Fit the model
    model.fit(X_train, y_train)
    return model


def get_forecast(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    market: Literal["DA", "RT"],
    method: Literal["persistence", "perfect", "xgboost", "regression"],
    price_node: str = PRICE_NODE,
    training_days: int = 70,
    number_of_lags: int = 96 + 10,
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
    if market in ["DA", "RT"]:
        price_col = f"{price_node}_{market}M"
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data.")
    else:
        raise ValueError(f"Unknown market: {market}")

    # Ensure data is indexed by datetime
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()

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
        forecast = pd.Series(forecast_values, index=time_index)

    elif method == "perfect":
        # Use the actual future prices from the data

        # Check that the forecast window exists in the data
        missing = time_index.difference(data.index)
        if not missing.empty:
            raise ValueError(f"Missing data for perfect forecast at: {list(missing)}")
        # Get the prices for the forecast window
        forecast = data.loc[time_index, price_col].copy()

    elif method == "xgboost":
        # Train the XGBoost model
        model = _train_xgboost_model(
            data=data,
            current_time=current_time,
            price_col=price_col,
            training_days=training_days,
            number_of_lags=number_of_lags,
        )
        # Generate forecast using the trained model
        forecast = _generate_forecast(
            data=data,
            current_time=current_time,
            horizon_hours=horizon_hours,
            price_col=price_col,
            model=model,
            number_of_lags=number_of_lags,
        )
    elif method == "regression":
        # Train the regression model
        model = _train_regression_model(
            data=data,
            current_time=current_time,
            price_col=price_col,
            training_days=training_days,
            number_of_lags=number_of_lags,
        )
        # Generate forecast using the trained model
        forecast = _generate_forecast(
            data=data,
            current_time=current_time,
            horizon_hours=horizon_hours,
            price_col=price_col,
            model=model,
            number_of_lags=number_of_lags,
        )

    else:
        raise ValueError(f"Unknown forecast method: {method}")

    forecast.name = f"{market}_forecast"

    if verbose:
        # Historical window: current_time - horizon_hours to current_time
        hist_start = current_time - pd.Timedelta(hours=horizon_hours)
        hist_end = current_time + pd.Timedelta(hours=horizon_hours)
        # Select historical data
        historical = data.loc[
            (data.index >= hist_start) & (data.index <= hist_end), price_col
        ]
        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(
            list(historical.index),
            list(historical.values),
            label="Historical",
            color="tab:blue",
        )
        plt.plot(
            list(forecast.index),
            list(forecast.values),
            label="Forecast",
            color="tab:orange",
        )
        plt.axvline(current_time, color="k", linestyle="--", label="Current Time")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title(f"{market} Price Forecast ({method})")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return forecast


def get_forecasts_for_da(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    method: Literal["persistence", "perfect", "xgboost", "regression"],
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
        method=method,
        price_node=price_node,
        verbose=verbose,
    )[-TIME_STEPS_PER_HOUR * horizon_hours :]
    rt_prices = get_forecast(
        data,
        current_time=current_time,
        horizon_hours=horizon_hours + 14,
        market="RT",
        method=method,
        price_node=price_node,
        verbose=verbose,
    )[-TIME_STEPS_PER_HOUR * horizon_hours :]
    return da_prices, rt_prices
