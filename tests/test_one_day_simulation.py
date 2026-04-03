import pandas as pd
import pytest

from src.one_day_simulation import one_day_simulation
from src.utils.battery_model import BatteryParams
from src.utils.load_ercot_data import load_ercot_data


def test_one_day_simulation():
    data = load_ercot_data(verbose=False)
    one_day_simulation(
        data=data,
        operating_day=pd.Timestamp("2025-06-15 00:00:00"),
        battery=BatteryParams(),
        daily_simulation_horizon_hours=24,
        da_stage_kwargs={},
        use_rt_mpc=True,
        rt_stage_kwargs={},
        rt_control_horizon_type="receding",
        da_stage_forecast_method="persistence",
        rt_stage_forecast_method="persistence",
    )


def test_one_day_simulation_rt_stage_enforces_end_of_day_soc(
    ercot_data: pd.DataFrame,
    battery: BatteryParams,
):
    target_soc = 0.6
    result = one_day_simulation(
        data=ercot_data,
        operating_day=pd.Timestamp("2025-06-15"),
        battery=battery,
        use_rt_mpc=True,
        da_stage_kwargs={"initial_soc": 0.5, "end_of_day_soc": target_soc},
        rt_stage_kwargs={"end_of_day_soc": target_soc},
        da_stage_forecast_method="persistence",
        rt_stage_forecast_method="persistence",
    )

    assert result.soc_schedule[-1] == pytest.approx(target_soc, abs=5e-2)


def test_one_day_simulation_raises_on_mismatched_end_of_day_soc(
    ercot_data: pd.DataFrame,
    battery: BatteryParams,
):
    with pytest.raises(AssertionError, match="End of day SOC must be the same"):
        one_day_simulation(
            data=ercot_data,
            operating_day=pd.Timestamp("2025-06-15"),
            battery=battery,
            use_rt_mpc=True,
            da_stage_kwargs={"initial_soc": 0.5, "end_of_day_soc": 0.5},
            rt_stage_kwargs={"end_of_day_soc": 0.6},
            da_stage_forecast_method="persistence",
            rt_stage_forecast_method="persistence",
        )
