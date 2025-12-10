from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

def load_ercot_data(filepath: str) -> pd.DataFrame:
    """
    Load ERCOT price and ancillary services data from a CSV file.
    Parses the 'key' column as datetime and sets it as the index.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and all columns from the CSV.
    """
    df = pd.read_csv(filepath)
    if 'key' not in df.columns:
        raise ValueError("Expected 'key' column in data.")
    df['key'] = pd.to_datetime(df['key'])
    df = df.set_index('key')
    return df

@dataclass
class DAScheduleResult:
    """Results from Stage 1 DA optimization."""
    da_energy_bids: np.ndarray      # Shape (24,) [MW]
    reg_up_capacity: np.ndarray     # Shape (24,) [MW]
    reg_down_capacity: np.ndarray   # Shape (24,) [MW]
    planned_soc: np.ndarray         # Shape (25,) [0-1]
    expected_revenue: float         # [$]

@dataclass
class RTMPCResult:
    """Results from Stage 2 RT MPC."""
    power_setpoint: float           # [MW] (+discharge, -charge)
    predicted_soc: np.ndarray       # Over horizon
    solve_status: str               # 'optimal', 'infeasible'

@dataclass
class DaySimulationResult:
    """Results from single-day simulation."""
    date: pd.Timestamp
    total_revenue: float            # [$]
    da_revenue: float               # [$]
    rt_revenue: float               # [$]
    soc_trajectory: np.ndarray      # Shape (97,) for 15-min steps
    power_trajectory: np.ndarray    # Shape (96,) [MW]
    final_soc: float                # [0-1]

@dataclass
class SimulationResult:
    """Results from multi-day simulation."""
    daily_results: List[DaySimulationResult]
    cumulative_revenue: np.ndarray  # Shape (n_days,)
    total_revenue: float            # [$]
