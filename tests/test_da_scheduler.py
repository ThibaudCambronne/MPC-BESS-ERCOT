import matplotlib.pyplot as plt
import pandas as pd

from src.battery_model import BatteryParams
from src.forecaster import get_forecast
from src.globals import TIME_STEPS_PER_HOUR
from src.stage1_da_scheduler import solve_da_schedule
from src.utils import load_ercot_data


def test_da_scheduler():
    data = load_ercot_data()
    current_time = pd.Timestamp("2025-02-01 10:00:00")
    da_prices = get_forecast(
        data,
        current_time=current_time,
        horizon_hours=38,
        market="DA",
        method="persistence",
        verbose=True,
    )[-TIME_STEPS_PER_HOUR * 24 :]
    rt_prices = get_forecast(
        data,
        current_time=current_time,
        horizon_hours=38,
        market="RT",
        method="persistence",
        verbose=True,
    )[-TIME_STEPS_PER_HOUR * 24 :]

    # Battery parameters
    battery = BatteryParams()

    # Solve DA schedule
    result = solve_da_schedule(
        da_price_forecast=da_prices,
        rt_price_forecast=rt_prices,
        battery=battery,
    )

    # Verify result structure
    assert result.da_energy_bids.shape == da_prices.shape
    assert result.soc_schedule.shape == (da_prices.shape[0] + 1,)  # T+1 for end state
    assert isinstance(result.expected_revenue, float)

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    print(len(da_prices.index), "LENGTH INDEX")
    print(len(result.soc_schedule), "LENGTH SOC")

    # Plot 1: SOC Schedule
    axes[0].plot(da_prices.index, result.soc_schedule[:-1])
    axes[0].set_title("State of Charge Schedule")
    axes[0].set_ylabel("SOC")
    axes[0].grid(True)
    print(da_prices.index)
    # Plot 2: Power (DA bids, RT bids, and Dispatch)
    axes[1].plot(da_prices.index, result.da_energy_bids, label="DA energy bids")
    axes[1].plot(da_prices.index, result.rt_energy_bids, label="RT energy bids")
    axes[1].plot(da_prices.index, result.power_dispatch_schedule, label="Dispatch schedule")
    axes[1].set_title("Power Schedule")
    axes[1].set_ylabel("Power")
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Prices
    axes[2].plot(da_prices.index, da_prices, label="DA prices")
    axes[2].plot(da_prices.index, rt_prices, label="RT prices")
    axes[2].set_title("Price Forecasts")
    axes[2].set_ylabel("Price ($/MWh)")
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: Charge/Discharge
    axes[3].plot(result.diagnostic_information["charge"][:-1], label="Charge")
    axes[3].plot(result.diagnostic_information["discharge"][:-1], label="Discharge")
    axes[3].set_title("Charge/Discharge Schedule")
    axes[3].set_ylabel("Power")
    axes[3].set_xlabel("Time Step")
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig("tests/da_scheduler_results.png", dpi=150)
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Running DA Scheduler Tests...\n")
    test_da_scheduler()
    print()
    print("\n PASSED TESTS ! ")
