from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.globals import (
    DATA_PATH_DAM_TESTING,
    DATA_PATH_DAM_TRAINING,
    DATA_PATH_RTM,
    PRICE_NODE,
)


def load_ercot_data() -> pd.DataFrame:
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
    # ====================
    # Load DAM data
    df_dam_train = pd.read_csv(
        DATA_PATH_DAM_TRAINING, usecols=["key", f"{PRICE_NODE}_DAM"]
    )
    df_dam_test = pd.read_csv(
        DATA_PATH_DAM_TESTING, usecols=["key", f"{PRICE_NODE}_DAM"]
    )
    df_dam = pd.concat([df_dam_train, df_dam_test], ignore_index=True)

    # Specify the correct datetime format for parsing
    # Example format: '01/01/2020 1' -> '%m/%d/%Y %H'
    date_str = df_dam["key"].str.slice(0, 10)  # "MM/DD/YYYY"
    date_parsed = pd.to_datetime(date_str, format="%m/%d/%Y")
    hours = df_dam["key"].str.slice(11).astype(int) - 1
    df_dam["key"] = date_parsed + pd.to_timedelta(hours, unit="h")

    # Resample to 15-min intervals by forward filling
    df_dam = (
        df_dam.drop_duplicates()
        .set_index("key")
        .resample("15min")
        .ffill()
        .reset_index()
    )

    # ====================
    # Load RTM data
    df_rtm = pd.read_csv(DATA_PATH_RTM, usecols=["hour_timestamp", PRICE_NODE]).rename(
        columns={PRICE_NODE: f"{PRICE_NODE}_RTM", "hour_timestamp": "key"}
    )

    df_rtm["key"] = pd.to_datetime(df_rtm["key"])
    # The key are missing the minutes (the data has for instance 4 values for 00:00 instead of 00:00, 00:15, 00:30, 00:45)
    # We fix that here by adding the minutes based on the occurrence within each hour
    df_rtm["minute"] = df_rtm.groupby(df_rtm["key"]).cumcount() * 15
    df_rtm["key"] = df_rtm["key"] + pd.to_timedelta(df_rtm["minute"], unit="m")
    df_rtm = df_rtm.drop(columns=["minute"])

    # Merge DAM and RTM on datetime
    df_all = df_dam.merge(
        df_rtm,
        on="key",
        how="inner",
    ).set_index("key")

    df_all["date_str"] = df_all.index.strftime("%m/%d/%Y")
    group_sizes = df_all.groupby("date_str").size()

    hours_per_day_rtm = 24 * 4
    full_days_rtm = group_sizes[group_sizes == hours_per_day_rtm].index
    df_all = df_all[df_all["date_str"].isin(full_days_rtm)]

    return df_all


@dataclass
class DAScheduleResult:
    """Results from Stage 1 DA optimization.
    CHANGED!"""

    da_energy_bids: np.ndarray  # Shape (24,) [MW]
    rt_energy_bids: np.ndarray  # Shape (288,) [MW]
    power_dispatch_schedule: np.ndarray  # Shape (288,) [MW]
    soc_schedule: np.ndarray        # Shape (289,) [0-1]
    reg_up_capacity: np.ndarray     # Shape (288,) [MW]
    reg_down_capacity: np.ndarray   # Shape (288,) [MW]
    expected_revenue: float         # [$]
    diagnostic_information: Optional[dict] # stuff I need for debugging

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


@dataclass
class SimulationResult:
    """Results from multi-day simulation."""

    daily_results: List[DaySimulationResult]
    cumulative_revenue: np.ndarray  # Shape (n_days,)
    total_revenue: float  # [$]
