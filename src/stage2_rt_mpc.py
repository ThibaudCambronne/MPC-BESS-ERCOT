from typing import Optional

import cvxpy as cp
import pandas as pd

from src.globals import FREQUENCY, TYPE_RT_HORIZON
from src.stage1_da_scheduler import get_optimization_problem

from .utils.battery_model import BatteryParams
from .utils.data_classes import DAScheduleResult, RTMPCResult


def solve_rt_mpc(
    current_time: pd.Timestamp,
    current_soc: float,
    rt_price_forecast: pd.Series,
    da_schedule: DAScheduleResult,
    battery: BatteryParams,
    horizon_type: TYPE_RT_HORIZON = "receding",
    end_of_day_soc: float = 0.5,
    cvar_alpha: float = 0.90,
    cvar_weight: float = 0,
    rt_dispatch_penalty: float = 0,
    rt_price_uncertainty: Optional[pd.Series] = None,
    rt_uncertainty_default: float = 0,  # 20
    n_scenarios: int = 20,
    scenario_seed: Optional[int] = None,
    verbose: bool = False,
) -> RTMPCResult:
    """
    Solve Stage 2 Real-Time (RT) MPC problem.
    """

    # ==================== Setup Horizon & Align Data ====================
    if horizon_type == "shrinking":
        raise NotImplementedError("Shrinking horizon not implemented yet.")
    else:
        end_time = rt_price_forecast.index[-1]

    horizon_index = pd.date_range(
        start=current_time, end=end_time, freq=FREQUENCY, inclusive="both"
    )
    # Number of time periods
    T = len(horizon_index)
    assert T > 0, "Horizon must contain at least one time step."

    # Align RT price forecast to horizon
    rt_price_forecast_aligned = rt_price_forecast.reindex(horizon_index)

    # Align DA Power commitments to horizon
    da_start = current_time.normalize()
    n_da = len(da_schedule.da_energy_bids)

    da_index = pd.date_range(start=da_start, periods=n_da, freq=FREQUENCY)
    da_series = pd.Series(da_schedule.da_energy_bids, index=da_index)

    # TODO: CHeck the fillna
    # The thing is that for receding horizon, we will eventually have time steps in
    # the following day, that aren't covered by the DA schedule.
    # We would need to get a new DA schedule for the next day.
    da_commitments = da_series.reindex(horizon_index).fillna(0.0)

    # ==================== Solve optimization ====================
    # TODO: Define the problem only once using cp.Parameters
    # for the forecasts and commitments, and then just update the parameter
    # values and resolve at each time step.
    # This will be much faster than reconstructing the problem from scratch at
    # each time step.
    problem, variables = get_optimization_problem(
        operating_day=current_time.normalize(),
        rt_price_forecast=rt_price_forecast_aligned,
        da_price_forecast=pd.Series(),
        battery=battery,
        initial_soc=current_soc,
        end_of_day_soc=end_of_day_soc,
        cvar_alpha=cvar_alpha,
        cvar_weight=cvar_weight,
        rt_dispatch_penalty=rt_dispatch_penalty,
        rt_price_uncertainty=rt_price_uncertainty,
        rt_uncertainty_default=rt_uncertainty_default,
        n_scenarios=n_scenarios,
        scenario_seed=scenario_seed,
        da_commitments=da_commitments.to_numpy(),
    )
    problem.solve(verbose=verbose)

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Optimization failed with status: {problem.status}")

    if problem.status == cp.OPTIMAL_INACCURATE:
        print("Warning: Optimization solved to optimality but is inaccurate.")

    # ==================== Extract results ====================
    return RTMPCResult(
        rt_energy_bids=variables["p_rt"].value,
        soc_schedule=variables["E"].value / battery.capacity_mwh,
        diagnostic_information={
            "problem_status": problem.status,
        },
    )
