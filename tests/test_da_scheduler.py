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

    fig = plt.figure()
    plt.plot(result.soc_schedule)
    plt.savefig("tests/soc_test.png")
    plt.close()
    fig = plt.figure()
    plt.plot(result.da_energy_bids)
    plt.plot(result.rt_energy_bids)
    plt.plot(result.power_dispatch_schedule)
    plt.legend(["DA energy bids", "RT energy bids", "Dispatch schedule"])
    plt.savefig("tests/power_test.png")
    plt.close()

    fig = plt.figure()
    plt.plot(da_prices)
    plt.plot(rt_prices)
    plt.legend(["DA prices", "RT prices"])
    plt.savefig("tests/prices_test.png")
    plt.close()


if __name__ == "__main__":
    print("Running DA Scheduler Tests...\n")
    test_da_scheduler()
    print()
    print("\n PASSED TESTS ! ")
