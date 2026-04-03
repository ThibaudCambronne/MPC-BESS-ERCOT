import numpy as np
import pandas as pd

from src.globals import FREQUENCY, TIME_STEPS_PER_HOUR
from src.utils.data_classes import DAScheduleResult, DaySimulationResult


def build_day_result(
    operating_day: pd.Timestamp,
    base_value: float,
    initial_soc: float = 0.5,
) -> DaySimulationResult:
    n_steps = 24 * TIME_STEPS_PER_HOUR
    index = pd.date_range(start=operating_day, periods=n_steps, freq=FREQUENCY)

    da_energy = np.full(n_steps, base_value)
    da_stage_rt_energy = np.full(n_steps, base_value + 1)
    rt_energy = np.full(n_steps, base_value + 2)

    da_prices = pd.Series(np.full(n_steps, 10.0 + base_value), index=index)
    da_stage_rt_prices = pd.Series(np.full(n_steps, 20.0 + base_value), index=index)
    rt_prices = pd.Series(np.full(n_steps, 30.0 + base_value), index=index)

    return DaySimulationResult(
        date=operating_day,
        da_energy_bids=da_energy,
        da_stage_rt_energy_bids=da_stage_rt_energy,
        rt_energy_bids=rt_energy,
        da_forecast_used=da_prices,
        da_stage_rt_forecast_used=da_stage_rt_prices,
        rt_forecast_used=rt_prices,
        da_stage_soc_schedule=np.full(n_steps + 1, initial_soc),
        soc_schedule=np.full(n_steps + 1, initial_soc),
        expected_revenue=0.0,
    )


def build_da_schedule(n_steps: int, initial_soc: float = 0.5) -> DAScheduleResult:
    zeros = np.zeros(n_steps)
    return DAScheduleResult(
        da_energy_bids=zeros.copy(),
        rt_energy_bids=zeros.copy(),
        power_dispatch_schedule=zeros.copy(),
        soc_schedule=np.full(n_steps + 1, initial_soc),
        reg_up_capacity=zeros.copy(),
        reg_down_capacity=zeros.copy(),
        expected_revenue=0.0,
        diagnostic_information=None,
    )


def build_rt_forecast(
    current_time: pd.Timestamp,
    end_time: pd.Timestamp,
    value: float = 30.0,
) -> pd.Series:
    index = pd.date_range(
        start=current_time,
        end=end_time,
        freq=FREQUENCY,
        inclusive="both",
    )
    return pd.Series(np.full(len(index), value), index=index)
