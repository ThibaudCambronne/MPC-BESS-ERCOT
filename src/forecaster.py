from typing import Literal

import pandas as pd


def get_forecast(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon_hours: int,
    market: Literal["DA", "RT"],
    method: Literal["persistence", "perfect"],
    price_node: str = "HB_HUBAVG",
) -> pd.Series:
    """
    Generate price forecast for the specified market.
    Returns a dummy time series with the correct number of elements and a time index.
    """
    # Create a time index starting from current_time, with hourly frequency
    time_index = pd.date_range(start=current_time, periods=horizon_hours, freq="h")
    # Create a dummy series
    dummy_values = [10.0] * horizon_hours
    return pd.Series(dummy_values, index=time_index, name=f"{market}_forecast")
