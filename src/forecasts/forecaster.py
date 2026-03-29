from typing import Literal

import numpy as np
import pandas as pd

from src.forecasts.add_cyclical_time_features import (
    add_cyclical_time_features,
)
from src.forecasts.build_forecast_vs_actual_plotly_figure import (
    build_forecast_vs_actual_plotly_figure,
)
from src.forecasts.train_regression_model import train_regression_model
from src.forecasts.train_xgboost_model import train_xgboost_model
from src.globals import (
    FREQUENCY,
    PRICE_NODE,
    TIME_STEPS_PER_HOUR,
    TYPE_FORECASTS,
    WEATHER_FEATURES,
)


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
        temp_df, time_features_cols = add_cyclical_time_features(temp_df)

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


def get_forecast(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    market: Literal["DA", "RT"],
    method: TYPE_FORECASTS,
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
        model = train_xgboost_model(
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
        model = train_regression_model(
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
        figure = build_forecast_vs_actual_plotly_figure(
            current_time=current_time,
            data=data,
            forecasts={method: forecast},
            market=market,
            price_col=price_col,
        )
        figure.show()
    return forecast


def get_forecasts_for_da(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    method: TYPE_FORECASTS,
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
