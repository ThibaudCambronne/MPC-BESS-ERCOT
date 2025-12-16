"""
Test file for forecasting functionality.
"""

from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from src.forecaster import get_forecast
from src.globals import PRICE_NODE
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

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(14, 7))

    # Historical data window: 24 hours before current_time to current_time
    hist_start = current_time - pd.Timedelta(hours=24)
    hist_end = current_time
    historical_data = data.loc[hist_start:hist_end, f"{PRICE_NODE}_{market}M"]

    # Plot historical data
    ax.plot(
        historical_data.index,
        historical_data.values,
        label="Historical (24h)",
        color="black",
        linestyle="-",
        linewidth=2,
    )

    # Generate and plot forecasts for each method
    methods = ["persistence", "perfect", "xgboost", "regression"]
    colors = {
        "persistence": "blue",
        "perfect": "green",
        "xgboost": "red",
        "regression": "purple",
    }

    for method in methods:
        try:
            # Generate forecast
            forecast = get_forecast(
                data=data,
                current_time=current_time,
                horizon_hours=horizon_hours,
                market=market,
                method=method,  # type: ignore
                price_node=PRICE_NODE,
                verbose=False,
            )

            # Plot forecast
            ax.plot(
                forecast.index,
                forecast.values,
                label=f"{method.capitalize()} Forecast",
                color=colors[method],
                linewidth=2,
            )

        except Exception as e:
            raise RuntimeError(f"Error generating {method} forecast: {e}")

    # Add vertical line at current time
    ax.axvline(current_time, color="gray", linestyle=":", label="Forecast Start")

    # Add dew point temperature on a secondary y-axis
    ax2 = ax.twinx()
    dew_point_col = "dew_point_temperature_S"
    ax.plot(
        forecast.index,
        data.loc[forecast.index, dew_point_col],
        label="Dew Point Temperature (°F)",
        color="orange",
        linestyle="--",
    )

    # Add temperature to the secondary y-axis
    ax2.plot(
        forecast.index,
        data.loc[forecast.index, "temperature_S"],
        label="Temperature (°F)",
        color="brown",
        linestyle="--",
    )
    ax2.set_ylabel("Temperature (°F)", fontsize=12)

    # Add labels and title
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Price ($/MWh)", fontsize=12)
    ax.set_title("Comparison of Forecasting Methods for 2025-02-02", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Run the test
    test_forecast_methods_comparison()
