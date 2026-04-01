import numpy as np
import pandas as pd

from app.utils_app.simulation_math import (
    compute_daily_tb_strategy_revenues,
    compute_tb_strategy_revenue,
)
from src.utils.battery_model import BatteryParams


def _build_quarter_hour_series(hourly_values: np.ndarray, start: str) -> pd.Series:
    values = np.repeat(hourly_values, 4)

    values[::4] = values[::4] - 0.5
    values[1::4] = values[1::4] - 0.25
    values[2::4] = values[2::4] + 0.25
    values[3::4] = values[3::4] + 0.5

    index = pd.date_range(start=start, periods=len(values), freq="15min")
    return pd.Series(values, index=index)


def test_tb2_tb4_revenue_scaling_default_battery():
    hourly_prices = np.arange(24, dtype=float)
    quarter_hour_prices = _build_quarter_hour_series(
        hourly_values=hourly_prices,
        start="2025-01-01 00:00:00",
    )
    battery = BatteryParams()

    tb2_revenue = compute_tb_strategy_revenue(
        price_series=quarter_hour_prices,
        battery=battery,
        n_pairs=2,
    )
    tb4_revenue = compute_tb_strategy_revenue(
        price_series=quarter_hour_prices,
        battery=battery,
        n_pairs=4,
    )

    # TB2 spread = (23 + 22) - (0 + 1) = 44, volume = min(25, 80/2) = 25
    assert tb2_revenue == 44.0 * 25.0
    # TB4 spread = (23 + 22 + 21 + 20) - (0 + 1 + 2 + 3) = 80, volume = min(25, 80/4) = 20
    assert tb4_revenue == 80.0 * 20.0


def test_daily_tb_revenue_outputs_for_da_and_rt():
    da_hourly = np.linspace(20.0, 43.0, 24)
    rt_hourly = np.linspace(80.0, 57.0, 24)

    da_prices = _build_quarter_hour_series(da_hourly, "2025-02-01 00:00:00")
    rt_prices = _build_quarter_hour_series(rt_hourly, "2025-02-01 00:00:00")

    revenues = compute_daily_tb_strategy_revenues(
        da_prices=da_prices,
        rt_prices=rt_prices,
        battery=BatteryParams(),
    )

    assert set(revenues.keys()) == {
        "tb2_da_revenue",
        "tb2_rt_revenue",
        "tb4_da_revenue",
        "tb4_rt_revenue",
    }
    assert np.isfinite(revenues["tb2_da_revenue"])
    assert np.isfinite(revenues["tb2_rt_revenue"])
    assert np.isfinite(revenues["tb4_da_revenue"])
    assert np.isfinite(revenues["tb4_rt_revenue"])

