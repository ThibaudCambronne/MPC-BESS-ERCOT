from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# The dataclasses below were not modified as they are not relevant to the data loading logic.
@dataclass
class DAScheduleResult:
    """Results from Stage 1 DA optimization.
    CHANGED!"""

    da_energy_bids: np.ndarray  # Shape (24,) [MW]
    rt_energy_bids: np.ndarray  # Shape (288,) [MW]
    power_dispatch_schedule: np.ndarray  # Shape (288,) [MW]
    soc_schedule: np.ndarray  # Shape (289,) [0-1]
    reg_up_capacity: np.ndarray  # Shape (288,) [MW]
    reg_down_capacity: np.ndarray  # Shape (288,) [MW]
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

    power_setpoint: float  # [MW] (+discharge, -charge)
    predicted_soc: np.ndarray  # Over horizon
    solve_status: str  # 'optimal', 'infeasible'


@dataclass
class DaySimulationResult:
    """Results from single-day simulation."""

    date: pd.Timestamp
    total_revenue: float  # [$]
    da_revenue: float  # [$]
    rt_revenue: float  # [$]
    soc_trajectory: np.ndarray  # Shape (97,) for 15-min steps
    power_trajectory: np.ndarray  # Shape (96,) [MW]
    final_soc: float  # [0-1]

    # Detailed revenue breakdowns for plotting (NEW)
    da_step_revenues: np.ndarray  # DA revenue at each time step
    rt_step_revenues: np.ndarray  # RT revenue at each time step
    da_power_bids: np.ndarray  # DA bids for each time step
    rt_imbalance: np.ndarray  # RT imbalance (actual - DA bids) for each time step


@dataclass
class SimulationResult:
    """Results from multi-day simulation."""

    daily_results: List[DaySimulationResult]
    cumulative_revenue: np.ndarray  # Shape (n_days,)
    total_revenue: float  # [$]
    da_schedules: dict  # Maps pd.Timestamp -> DAScheduleResult
