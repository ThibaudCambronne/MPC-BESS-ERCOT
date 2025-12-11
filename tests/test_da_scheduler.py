import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.battery_model import BatteryParams
from src.stage1_da_scheduler import solve_da_schedule


def test_da_scheduler():
    # test prices
    times = pd.date_range('2024-01-01', periods=96, freq='15min')
    # peak prices during peak times (are these the same in texas as CA?)
    da_prices = pd.Series([20.0] * 12 * 4 + [100.0] * 6 * 4 + [20.0] * 6 * 4, index=times)
    rt_prices = da_prices - 10 * (np.random.rand(24 * 4) - 0.5 * np.ones(24 * 4))
    # rt_prices = pd.Series(np.zeros(24 * 12), index = times)
    
    # Battery parameters
    battery = BatteryParams()
    
    # Solve DA schedule
    result = solve_da_schedule(
        da_price_forecast=da_prices,
        rt_price_forecast=rt_prices,
        battery=battery,
    )
    
    # Verify result structure
    assert result.da_energy_bids.shape == (len(times),)
    assert result.soc_schedule.shape == (len(times) + 1,)  # T+1 for end state
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

    plt.plot(result.diagnostic_information['charge'])
    plt.plot(result.diagnostic_information['discharge'])
    plt.savefig("tests/charge_test.png")



if __name__ == "__main__":
    print("Running DA Scheduler Tests...\n")
    test_da_scheduler()
    print()
    print("\n PASSED TESTS ! ")
