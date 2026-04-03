from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class DAScheduleResult:
    """Results from Stage 1 DA optimization."""

    da_energy_bids: np.ndarray  # [MW]
    rt_energy_bids: np.ndarray  # [MW]
    power_dispatch_schedule: np.ndarray  # [MW]
    soc_schedule: np.ndarray  # [0-1]
    reg_up_capacity: np.ndarray  # [MW]
    reg_down_capacity: np.ndarray  # [MW]
    expected_revenue: float  # [$]
    diagnostic_information: Optional[dict]  # stuff I need for debugging
    da_price_forecast: Optional[np.ndarray] = (
        None  # Forecast prices used (for plotting)
    )
    rt_price_forecast: Optional[np.ndarray] = (
        None  # Forecast prices used (for plotting)
    )


@dataclass
class RTMPCResult:
    """Results from Stage 2 RT MPC."""

    rt_energy_bids: np.ndarray  # [MW]
    soc_schedule: np.ndarray  # Over horizon [0-1]
    diagnostic_information: Optional[dict]


@dataclass
class DaySimulationResult:
    """Results from single-day simulation."""

    date: pd.Timestamp
    da_energy_bids: np.ndarray  # [MW]
    da_stage_rt_energy_bids: np.ndarray  # [MW]
    rt_energy_bids: np.ndarray  # [MW]
    da_forecast_used: pd.Series  # [$/MWh]
    da_stage_rt_forecast_used: pd.Series  # [$/MWh] DA forecast used for RT pricing
    rt_forecast_used: pd.Series  # [$/MWh]
    da_stage_soc_schedule: np.ndarray  # [0-1]
    soc_schedule: np.ndarray  # [0-1]
    expected_revenue: float  # [$]


@dataclass
class MultiDaySimulationResult:
    """Aggregated outputs from running one-day simulations over multiple days."""

    index: pd.DatetimeIndex
    operating_days: pd.DatetimeIndex
    da_energy_bids: np.ndarray  # [MW]
    da_stage_rt_energy_bids: np.ndarray  # [MW]
    rt_energy_bids: np.ndarray  # [MW]
    da_forecast_used: pd.Series  # [$/MWh]
    da_stage_rt_forecast_used: pd.Series  # [$/MWh]
    rt_forecast_used: pd.Series  # [$/MWh]
    da_energy_bids_perfect: np.ndarray  # [MW]
    rt_energy_bids_perfect: np.ndarray  # [MW]
    da_forecast_perfect: pd.Series  # [$/MWh]
    rt_forecast_perfect: pd.Series  # [$/MWh]


@dataclass
class SimulationResult:
    """Results from multi-day simulation."""

    daily_results: List[DaySimulationResult]
    cumulative_revenue: np.ndarray  # Shape (n_days,)
    total_revenue: float  # [$]
    da_schedules: dict  # Maps pd.Timestamp -> DAScheduleResult
