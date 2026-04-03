from collections.abc import Callable

import numpy as np
import pandas as pd

from src.globals import FREQUENCY, TIME_STEPS_PER_HOUR, TYPE_FORECASTS, TYPE_RT_HORIZON
from src.one_day_simulation import one_day_simulation
from src.utils.battery_model import BatteryParams
from src.utils.data_classes import MultiDaySimulationResult


def _build_multi_day_index(
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    operating_days = pd.date_range(
        start=pd.Timestamp(start_day).normalize(),
        end=pd.Timestamp(end_day).normalize(),
        freq="D",
    )
    index = pd.date_range(
        start=operating_days[0],
        end=operating_days[-1] + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
        freq=FREQUENCY,
    )
    return operating_days, index


def multi_day_simulation(
    data: pd.DataFrame,
    start_day: pd.Timestamp,
    end_day: pd.Timestamp,
    battery: BatteryParams,
    da_stage_kwargs: dict | None = None,
    use_rt_mpc: bool = False,
    rt_stage_kwargs: dict | None = None,
    rt_control_horizon_type: TYPE_RT_HORIZON = "receding",
    rt_horizon_hours: int = 24,
    da_stage_forecast_method: TYPE_FORECASTS = "perfect",
    rt_stage_forecast_method: TYPE_FORECASTS = "perfect",
    progress_callback: Callable[[int, int], None] | None = None,
) -> MultiDaySimulationResult:
    """Run one_day_simulation over an inclusive date range and aggregate outputs."""
    normalized_start = pd.Timestamp(start_day).normalize()
    normalized_end = pd.Timestamp(end_day).normalize()
    if normalized_end < normalized_start:
        raise ValueError("end_day must be on or after start_day.")

    da_stage_kwargs = dict(da_stage_kwargs or {})
    rt_stage_kwargs = dict(rt_stage_kwargs or {})

    operating_days, index = _build_multi_day_index(
        start_day=normalized_start,
        end_day=normalized_end,
    )

    n_steps_per_day = 24 * TIME_STEPS_PER_HOUR
    n_total_steps = len(index)

    da_energy_bids = np.zeros(n_total_steps)
    da_stage_rt_energy_bids = np.zeros(n_total_steps)
    rt_energy_bids = np.zeros(n_total_steps)

    da_energy_bids_perfect = np.zeros(n_total_steps)
    rt_energy_bids_perfect = np.zeros(n_total_steps)

    da_forecast_used = np.full(n_total_steps, np.nan)
    da_stage_rt_forecast_used = np.full(n_total_steps, np.nan)
    rt_forecast_used = np.full(n_total_steps, np.nan)

    da_forecast_perfect = np.full(n_total_steps, np.nan)
    rt_forecast_perfect = np.full(n_total_steps, np.nan)

    total_days = len(operating_days)
    for day_counter, operating_day in enumerate(operating_days, start=1):
        day_index = pd.date_range(
            start=operating_day,
            periods=n_steps_per_day,
            freq=FREQUENCY,
        )
        positions = index.get_indexer(day_index)
        if (positions < 0).any():
            raise ValueError(
                f"Missing timestamps while mapping operating day {operating_day:%Y-%m-%d}."
            )

        try:
            day_result = one_day_simulation(
                data=data,
                operating_day=operating_day,
                battery=battery,
                daily_simulation_horizon_hours=24,
                da_stage_kwargs=da_stage_kwargs,
                use_rt_mpc=use_rt_mpc,
                rt_stage_kwargs=rt_stage_kwargs,
                rt_control_horizon_type=rt_control_horizon_type,
                rt_horizon_hours=rt_horizon_hours,
                da_stage_forecast_method=da_stage_forecast_method,
                rt_stage_forecast_method=rt_stage_forecast_method,
            )
            day_perfect = one_day_simulation(
                data=data,
                operating_day=operating_day,
                battery=battery,
                daily_simulation_horizon_hours=24,
                da_stage_kwargs=da_stage_kwargs,
                use_rt_mpc=False,
                rt_stage_kwargs={},
                rt_control_horizon_type=rt_control_horizon_type,
                rt_horizon_hours=rt_horizon_hours,
                da_stage_forecast_method="perfect",
                rt_stage_forecast_method="perfect",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Multi-day simulation failed for operating day {operating_day:%Y-%m-%d}: {exc}"
            ) from exc

        da_energy_bids[positions] = day_result.da_energy_bids
        da_stage_rt_energy_bids[positions] = day_result.da_stage_rt_energy_bids
        rt_energy_bids[positions] = day_result.rt_energy_bids

        da_energy_bids_perfect[positions] = day_perfect.da_energy_bids
        rt_energy_bids_perfect[positions] = day_perfect.rt_energy_bids

        da_forecast_used[positions] = day_result.da_forecast_used.to_numpy()
        da_stage_rt_forecast_used[positions] = (
            day_result.da_stage_rt_forecast_used.to_numpy()
        )
        rt_forecast_used[positions] = day_result.rt_forecast_used.to_numpy()

        da_forecast_perfect[positions] = day_perfect.da_forecast_used.to_numpy()
        rt_forecast_perfect[positions] = day_perfect.rt_forecast_used.to_numpy()

        if progress_callback is not None:
            progress_callback(day_counter, total_days)

    return MultiDaySimulationResult(
        index=index,
        operating_days=operating_days,
        da_energy_bids=da_energy_bids,
        da_stage_rt_energy_bids=da_stage_rt_energy_bids,
        rt_energy_bids=rt_energy_bids,
        da_forecast_used=pd.Series(da_forecast_used, index=index),
        da_stage_rt_forecast_used=pd.Series(da_stage_rt_forecast_used, index=index),
        rt_forecast_used=pd.Series(rt_forecast_used, index=index),
        da_energy_bids_perfect=da_energy_bids_perfect,
        rt_energy_bids_perfect=rt_energy_bids_perfect,
        da_forecast_perfect=pd.Series(da_forecast_perfect, index=index),
        rt_forecast_perfect=pd.Series(rt_forecast_perfect, index=index),
    )
