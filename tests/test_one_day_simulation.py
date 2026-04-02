import pandas as pd

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
        da_schedule_kwargs={},
        use_rt_mpc=True,
        rt_schedule_kwargs={},
        rt_control_horizon_type="receding",
        forecast_method="persistence",
    )
