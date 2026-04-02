import numpy as np
import pandas as pd

from src.forecasts.forecaster import get_forecast, get_forecasts_for_da
from src.globals import (
    DELTA_T,
    PRICE_NODE,
    TIME_STEPS_PER_HOUR,
    TYPE_FORECASTS,
    TYPE_RT_HORIZON,
)
from src.stage1_da_scheduler import DEFAULT_END_OF_DAY_SOC, solve_da_schedule_cvxpy
from src.stage2_rt_mpc import solve_rt_mpc
from src.utils.battery_model import BatteryParams
from src.utils.data_classes import DaySimulationResult


def one_day_simulation(
    data: pd.DataFrame,
    operating_day: pd.Timestamp,
    battery: BatteryParams,
    daily_simulation_horizon_hours: int = 24,
    da_schedule_kwargs: dict = {},
    use_rt_mpc: bool = False,
    rt_schedule_kwargs: dict = {},
    rt_control_horizon_type: TYPE_RT_HORIZON = "receding",
    rt_horizon_hours: int = 24,
    forecast_method: TYPE_FORECASTS = "perfect",
) -> DaySimulationResult:
    """
    Runs a simulation for a given day:
    - Stage 1: DA Market
        - Get forecasts at 10:00 AM on D-1
        - Solve DA optimization to get DA bids and schedule, and tentative RT schedule
    - Stage 2: RT Market
        - Every 15 minutes, get updated forecasts and solve RT MPC to get actual dispatch

    Parameters
    ----------
    data : pd.DataFrame
        ERCOT market data with price and other features.
    operating_day : pd.Timestamp
        The day for which to run the simulation (e.g., '2024-01-01').
    battery : BatteryParams
        Parameters for the battery model.
    daily_simulation_horizon_hours : int
        Number of hours to simulate (default 24).
    da_schedule_kwargs : dict
        Parameters to pass to the DA scheduling function (e.g., src.stage1_da_scheduler.solve_da_schedule_cvxpy).
    use_rt_mpc : bool
        Whether to run the RT MPC stage.
        If False, only DA scheduling is performed and RT revenue is calculated based on DA schedule and actual RT prices.
    rt_schedule_kwargs : dict
        Parameters to pass to the RT scheduling function (e.g., src.stage2_rt_mpc.solve_rt_mpc).
    rt_control_horizon_type : TYPE_HORIZON
        The type of horizon for RT control.
    rt_horizon_hours : int
        The number of hours to include in the RT MPC horizon (default 24).
    forecast_method : TYPE_FORECASTS
        The method to use for generating forecasts.

    """
    # Normalize date to midnight
    operating_day_start = pd.Timestamp(operating_day).normalize()
    schedule_time = operating_day_start - pd.Timedelta(days=1) + pd.Timedelta(hours=10)

    # === Stage 1: DA Market ===
    # forecasts and actual prices for the day
    da_forecast, rt_forecast = get_forecasts_for_da(
        data=data,
        current_time=schedule_time,
        horizon_hours=daily_simulation_horizon_hours,
        method=forecast_method,
        price_node=PRICE_NODE,
        verbose=False,
    )

    da_schedule = solve_da_schedule_cvxpy(
        da_price_forecast=da_forecast,
        rt_price_forecast=rt_forecast,
        battery=battery,
        **da_schedule_kwargs,
    )

    if not use_rt_mpc:
        rt_energy_bids = da_schedule.rt_energy_bids
        rt_soc_schedule = da_schedule.soc_schedule
        rt_prices_used = rt_forecast.to_numpy()

    else:
        # === Stage 2: Real-Time MPC (run every 15 minutes) ===
        num_intervals = (
            TIME_STEPS_PER_HOUR * daily_simulation_horizon_hours
        )  # 96 intervals in 24 hours
        rt_soc_schedule = np.zeros(num_intervals + 1)
        # Start from initial SOC from DA schedule
        rt_soc_schedule[0] = da_schedule.soc_schedule[0]

        rt_energy_bids = np.zeros(num_intervals)
        rt_prices_used = np.zeros(num_intervals)

        for t in range(num_intervals):
            current_time = operating_day_start + pd.Timedelta(minutes=15 * t)
            current_soc = rt_soc_schedule[t]

            # Get RT price forecast from current time onwards
            rt_forecast_t = get_forecast(
                data=data,
                current_time=current_time,
                horizon_hours=rt_horizon_hours,
                market="RT",
                method=forecast_method,
            )

            # Solve RT MPC
            rt_schedule_t = solve_rt_mpc(
                current_time=current_time,
                current_soc=current_soc,
                rt_price_forecast=rt_forecast_t,
                da_schedule=da_schedule,
                battery=battery,
                horizon_type=rt_control_horizon_type,
                end_of_day_soc=da_schedule_kwargs.get(
                    "end_of_day_soc", DEFAULT_END_OF_DAY_SOC
                ),
            )

            # Apply first power setpoint
            rt_energy_bids[t] = rt_schedule_t.rt_energy_bids[0]
            rt_soc_schedule[t + 1] = rt_schedule_t.soc_schedule[1]
            rt_prices_used[t] = rt_forecast_t.iloc[0]

    # === Calculate Revenues ===
    da_energy_bids = da_schedule.da_energy_bids
    da_revenue = -(da_energy_bids @ da_forecast.to_numpy() * DELTA_T)
    rt_revenue = -(rt_energy_bids @ rt_prices_used * DELTA_T)
    expected_revenue: float = da_revenue + rt_revenue  # type: ignore

    return DaySimulationResult(
        date=operating_day_start,
        da_energy_bids=da_energy_bids,
        rt_energy_bids=rt_energy_bids,
        soc_schedule=rt_soc_schedule,
        expected_revenue=expected_revenue,
    )
