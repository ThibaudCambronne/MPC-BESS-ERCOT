import numpy as np
import pandas as pd
from src.battery_model import BatteryParams
from src.stage1_da_scheduler import solve_da_schedule


def test_da_scheduler():
    # test prices
    hours = pd.date_range('2024-01-01', periods=24, freq='h')
    # peak prices during peak hours (are these the same in texas as CA?)
    da_prices = pd.Series([20.0] * 12 + [100.0] * 6 + [20.0] * 6, index=hours)
    
    # Battery parameters
    battery = BatteryParams()

    # Initial SoC at 50%
    initial_soc = 0.5
    
    # Solve DA schedule
    result = solve_da_schedule(
        da_price_forecast=da_prices,
        initial_soc=initial_soc,
        battery=battery,
    )
    
    # Verify result structure
    assert result.da_energy_bids.shape == (24,)
    assert result.reg_up_capacity.shape == (24,)
    assert result.reg_down_capacity.shape == (24,)
    assert result.planned_soc.shape == (25,)  # T+1 for initial state
    assert isinstance(result.expected_revenue, float)
    
    # Expected behavior: charge during low prices, discharge during high prices
    peak_hours = slice(12, 18)
    assert np.mean(result.da_energy_bids[peak_hours]) > 0, "Should discharge during peak hours"
    
    off_peak_hours = slice(0, 12)
    assert np.mean(result.da_energy_bids[off_peak_hours]) < 0, "Should charge during off-peak hours"

def test_da_scheduler_soc_constraints():
    """Test that SoC constraints are respected."""
    hours = pd.date_range('2024-01-01', periods=24, freq='h')
    
    # Extreme prices to test constraints
    da_prices = pd.Series([10.0] * 12 + [200.0] * 12, index=hours)
    rt_prices = pd.Series([15.0] * 12 + [190.0] * 12, index=hours)
    
    battery = BatteryParams(
        capacity_mwh=50.0,
        power_max_mw=20.0,
        soc_min=0.2,
        soc_max=0.8,
    )
    
    initial_soc = 0.5
    
    result = solve_da_schedule(
        da_price_forecast=da_prices,
        initial_soc=initial_soc,
        battery=battery
    )
    
    # Verify SoC stays within bounds
    assert np.all(result.planned_soc >= battery.soc_min - 1e-4), \
        f"SoC below minimum: {result.planned_soc.min()}"
    assert np.all(result.planned_soc <= battery.soc_max + 1e-4), \
        f"SoC above maximum: {result.planned_soc.max()}"
    
    print(f"SoC range: [{result.planned_soc.min():.2%}, {result.planned_soc.max():.2%}]")


if __name__ == "__main__":
    print("Running DA Scheduler Tests...\n")
    test_da_scheduler()
    print()
    test_da_scheduler_soc_constraints()
    print("\n PASSED TESTS ! ")
