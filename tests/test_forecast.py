"""
Test file for forecasting functionality.
"""

from typing import Literal

import pandas as pd

from src.forecasts.build_forecast_vs_actual_plotly_figure import (
    build_forecast_vs_actual_plotly_figure,
)
from src.forecasts.forecaster import get_forecast
from src.globals import PRICE_NODE, TYPE_FORECASTS
from src.utils.load_ercot_data import load_ercot_data


def test_forecast_methods_comparison():
    """
    Test that creates 4 forecasts for the data on 2025-02-02 and plots them
    on the same figure along with historical data.
    """
    # Load the test data
    data = load_ercot_data()

    # Define the current time for forecasting (2025-02-02 at 10:00 AM)
    current_time = pd.Timestamp("2025-02-02 10:00:00")

    # Define the forecast horizon (24 hours)
    horizon_hours = 24
    market: Literal["DA", "RT"] = "DA"
    price_col = f"{PRICE_NODE}_{market}M"

    methods: list[TYPE_FORECASTS] = [
        "persistence",
        "perfect",
        "xgboost",
        "regression",
    ]
    forecasts: dict[str, pd.Series] = {}

    for method in methods:
        forecasts[method] = get_forecast(
            data=data,
            current_time=current_time,
            horizon_hours=horizon_hours,
            market=market,
            method=method,
            price_node=PRICE_NODE,
            verbose=False,
        )

    figure = build_forecast_vs_actual_plotly_figure(
        current_time=current_time,
        data=data,
        forecasts=forecasts,
        market=market,
        price_col=price_col,
    )

    figure_data = getattr(figure, "data", [])
    expected_trace_count = 2 + len(methods)
    assert len(figure_data) == expected_trace_count

    for trace in figure_data[2:]:
        assert getattr(getattr(trace, "line", None), "dash", None) == "dash"

    figure_title = getattr(getattr(figure, "layout", None), "title", None)
    title_text = getattr(figure_title, "text", "")
    assert title_text == f"{market} Price Forecast Comparison"

    figure.show()


if __name__ == "__main__":
    # Run the test
    test_forecast_methods_comparison()
