import numpy as np
import pandas as pd

from src.globals import DELTA_T
from src.stage1_da_scheduler import solve_da_schedule, solve_da_schedule_cvxpy
from src.utils.battery_model import BatteryParams
from src.utils.data_classes import DAScheduleResult

from .selectors import ALGO_CVXPY, ALGO_PYOMO


def solve_schedule_for_algorithm(
    algorithm: str,
    da_price_forecast: pd.Series,
    rt_price_forecast: pd.Series,
    battery: BatteryParams,
    initial_soc: float = 0.5,
    end_of_day_soc: float = 0.5,
) -> DAScheduleResult:
    if algorithm == ALGO_CVXPY:
        return solve_da_schedule_cvxpy(
            da_price_forecast=da_price_forecast,
            rt_price_forecast=rt_price_forecast,
            battery=battery,
            initial_soc=initial_soc,
            end_of_day_soc=end_of_day_soc,
        )

    if algorithm == ALGO_PYOMO:
        return solve_da_schedule(
            da_price_forecast=da_price_forecast,
            rt_price_forecast=rt_price_forecast,
            battery=battery,
            initial_soc=initial_soc,
            end_of_day_soc=end_of_day_soc,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def compute_revenue_series(
    da_forecast: pd.Series,
    rt_forecast: pd.Series,
    da_bids: np.ndarray,
    rt_bids: np.ndarray,
    da_forecast_perfect: pd.Series,
    rt_forecast_perfect: pd.Series,
    da_bids_perfect: np.ndarray,
    rt_bids_perfect: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planned_step_revenue = (
        -(da_forecast.values * da_bids + rt_forecast.values * rt_bids) * DELTA_T
    )
    realized_step_revenue = (
        -(da_forecast_perfect.values * da_bids + rt_forecast_perfect.values * rt_bids)
        * DELTA_T
    )
    perfect_step_revenue = (
        -(
            da_forecast_perfect.values * da_bids_perfect
            + rt_forecast_perfect.values * rt_bids_perfect
        )
        * DELTA_T
    )
    return planned_step_revenue, realized_step_revenue, perfect_step_revenue
