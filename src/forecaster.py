from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from src.globals import FREQUENCY, PRICE_NODE, TIME_STEPS_PER_HOUR


def get_forecast(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    market: Literal["DA", "RT"],
    method: Literal["persistence", "perfect"],
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
    if market in ["DA", "RT"]:
        price_col = f"{price_node}_{market}M"
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
        missing = [t for t in prev_day_times if t not in data.index]
        if missing:
            raise ValueError(
                f"Missing historical data for persistence forecast at: {missing}"
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
        # Get the prices for the forecast window
        forecast = data.loc[time_index, price_col].copy()
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


def get_forecasts_for_da(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    method: Literal["persistence", "perfect"],
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
