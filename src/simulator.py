import pandas as pd
from typing import Literal
from .battery_model import BatteryParams
from .utils import DaySimulationResult, SimulationResult

def simulate_day(
    data: pd.DataFrame,
    date: pd.Timestamp,
    initial_soc: float,
    battery: BatteryParams,
    forecast_method: Literal["persistence", "perfect"],
    horizon_type: Literal["shrinking", "receding"] = "receding"
) -> DaySimulationResult:
    """
    Simulate one complete day (24 hours).
    Returns DaySimulationResult.
    """
    pass

def run_simulation(
    data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    battery: BatteryParams,
    forecast_method: Literal["persistence", "perfect"],
    horizon_type: Literal["shrinking", "receding"] = "receding"
) -> SimulationResult:
    """
    Run multi-day simulation.
    Returns SimulationResult.
    """
    pass
