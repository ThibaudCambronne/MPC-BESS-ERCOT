import pandas as pd
import numpy as np
from typing import Literal
from .battery_model import BatteryParams
from .utils import DaySimulationResult, SimulationResult
from .forecaster import get_forecast
from .stage1_da_scheduler import solve_da_schedule
from .stage2_rt_mpc import solve_rt_mpc
from .globals import DELTA_T, PRICE_NODE

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
    # Normalize date to midnight
    day_start = pd.Timestamp(date).normalize()
    day_end = day_start + pd.Timedelta(days=1)

    # Extract day's actual data for revenue calculation
    day_data = data.loc[day_start:day_end - pd.Timedelta(minutes=15)]
    if len(day_data) != 96:
        raise ValueError(f"Expected 96 intervals for date {date}, got {len(day_data)}")

    # Get actual prices for the day
    actual_da_prices = day_data[f"{PRICE_NODE}_DAM"].values
    actual_rt_prices = day_data[f"{PRICE_NODE}_RTM"].values

    # === Stage 1: Day-Ahead Scheduling (run once at start of day) ===
    da_price_forecast = get_forecast(
        data=data,
        current_time=day_start,
        horizon_hours=24,
        market="DA",
        method=forecast_method
    )

    rt_price_forecast_da = get_forecast(
        data=data,
        current_time=day_start,
        horizon_hours=24,
        market="RT",
        method=forecast_method
    )

    da_schedule = solve_da_schedule(
        da_price_forecast=da_price_forecast,
        rt_price_forecast=rt_price_forecast_da,
        battery=battery,
        initial_soc=initial_soc,
        end_of_day_soc=0.5
    )

    # === Stage 2: Real-Time MPC (run every 15 minutes) ===
    num_intervals = 96
    soc_trajectory = np.zeros(num_intervals + 1)
    power_trajectory = np.zeros(num_intervals)
    soc_trajectory[0] = initial_soc

    for t in range(num_intervals):
        current_time = day_start + pd.Timedelta(minutes=15 * t)
        current_soc = soc_trajectory[t]

        # Get RT price forecast from current time onwards
        rt_price_forecast = get_forecast(
            data=data,
            current_time=current_time,
            horizon_hours=24,
            market="RT",
            method=forecast_method
        )

        # Solve RT MPC
        rt_result = solve_rt_mpc(
            current_time=current_time,
            current_soc=current_soc,
            rt_price_forecast=rt_price_forecast,
            da_commitments=da_schedule,
            battery=battery,
            horizon_type=horizon_type
        )

        # Apply power setpoint
        power_setpoint = rt_result.power_setpoint
        power_trajectory[t] = power_setpoint

        # Update SOC based on actual power dispatch
        # power_setpoint: positive = discharge, negative = charge
        if power_setpoint > 0:  # Discharging
            energy_change_mwh = -power_setpoint / battery.efficiency_discharge * DELTA_T
        else:  # Charging (power_setpoint <= 0)
            energy_change_mwh = -power_setpoint * battery.efficiency_charge * DELTA_T

        soc_next = current_soc + energy_change_mwh / battery.capacity_mwh

        # Clamp SOC to valid range
        soc_next = np.clip(soc_next, battery.soc_min, battery.soc_max)
        soc_trajectory[t + 1] = soc_next

    # === Calculate Revenues ===
    # DA revenue: sum(da_energy_bids * actual_da_prices * delta_t)
    da_revenue = -np.sum(da_schedule.da_energy_bids * actual_da_prices * DELTA_T)

    # RT revenue: sum(rt_energy_bids * actual_rt_prices * delta_t)
    rt_revenue = -np.sum(da_schedule.rt_energy_bids * actual_rt_prices * DELTA_T)

    total_revenue = da_revenue + rt_revenue

    return DaySimulationResult(
        date=day_start,
        total_revenue=total_revenue,
        da_revenue=da_revenue,
        rt_revenue=rt_revenue,
        soc_trajectory=soc_trajectory,
        power_trajectory=power_trajectory,
        final_soc=soc_trajectory[-1]
    )

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
    # Normalize dates
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    # Generate list of dates to simulate
    dates = pd.date_range(start=start, end=end, freq='D')
    n_days = len(dates)

    # Initialize storage
    daily_results = []
    cumulative_revenue = np.zeros(n_days)

    # Initial SOC for first day
    current_soc = 0.5

    # Simulate each day
    for i, date in enumerate(dates):
        print(f"Simulating day {i+1}/{n_days}: {date.date()}")

        # Simulate this day
        day_result = simulate_day(
            data=data,
            date=date,
            initial_soc=current_soc,
            battery=battery,
            forecast_method=forecast_method,
            horizon_type=horizon_type
        )

        # Store results
        daily_results.append(day_result)

        # Update cumulative revenue
        if i == 0:
            cumulative_revenue[i] = day_result.total_revenue
        else:
            cumulative_revenue[i] = cumulative_revenue[i-1] + day_result.total_revenue

        # Update SOC for next day
        current_soc = day_result.final_soc

    # Calculate total revenue
    total_revenue = sum(day.total_revenue for day in daily_results)

    return SimulationResult(
        daily_results=daily_results,
        cumulative_revenue=cumulative_revenue,
        total_revenue=total_revenue
    )
