import numpy as np
import pandas as pd
import pytest

from src.globals import TIME_STEPS_PER_HOUR
from src.stage2_rt_mpc import solve_rt_mpc
from src.utils.battery_model import BatteryParams
from src.utils.load_ercot_data import load_ercot_data
from tests.helpers import build_da_schedule, build_rt_forecast


@pytest.fixture(scope="module")
def battery() -> BatteryParams:
    return BatteryParams()


@pytest.fixture(scope="module")
def ercot_data() -> pd.DataFrame:
    return load_ercot_data(verbose=False)


def test_solve_rt_mpc_respects_end_of_day_soc(battery: BatteryParams):
    current_time = pd.Timestamp("2025-06-15 12:00:00")
    end_time = pd.Timestamp("2025-06-15 23:45:00")

    rt_forecast = build_rt_forecast(
        current_time=current_time, end_time=end_time + pd.Timedelta(hours=5)
    )
    da_schedule = build_da_schedule(n_steps=24 * TIME_STEPS_PER_HOUR, initial_soc=0.5)

    target_soc = 0.7
    result = solve_rt_mpc(
        current_time=current_time,
        current_soc=0.5,
        rt_price_forecast=rt_forecast,
        da_schedule=da_schedule,
        battery=battery,
        end_of_day_soc=target_soc,
    )

    index_of_end_time = rt_forecast.index.get_loc(end_time)
    assert result.soc_schedule[index_of_end_time] == pytest.approx(target_soc, abs=1e-2)


def test_solve_rt_mpc_soc_stays_within_bounds(battery: BatteryParams):
    current_time = pd.Timestamp("2025-06-15 10:00:00")
    end_time = pd.Timestamp("2025-06-15 23:45:00")

    rt_forecast = build_rt_forecast(current_time=current_time, end_time=end_time)
    da_schedule = build_da_schedule(n_steps=24 * TIME_STEPS_PER_HOUR, initial_soc=0.5)

    result = solve_rt_mpc(
        current_time=current_time,
        current_soc=0.5,
        rt_price_forecast=rt_forecast,
        da_schedule=da_schedule,
        battery=battery,
        end_of_day_soc=0.5,
    )

    assert np.isfinite(result.soc_schedule).all()
    assert (result.soc_schedule >= battery.soc_min - 1e-6).all()
    assert (result.soc_schedule <= battery.soc_max + 1e-6).all()


def test_solve_rt_mpc_raises_on_infeasible_end_of_day_soc(battery: BatteryParams):
    current_time = pd.Timestamp("2025-06-15 12:00:00")
    end_time = pd.Timestamp("2025-06-15 23:45:00")

    rt_forecast = build_rt_forecast(current_time=current_time, end_time=end_time)
    da_schedule = build_da_schedule(n_steps=24 * TIME_STEPS_PER_HOUR, initial_soc=0.5)

    with pytest.raises(ValueError, match="Optimization failed"):
        solve_rt_mpc(
            current_time=current_time,
            current_soc=0.5,
            rt_price_forecast=rt_forecast,
            da_schedule=da_schedule,
            battery=battery,
            end_of_day_soc=1.2,
        )
