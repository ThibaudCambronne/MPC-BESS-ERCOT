import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Literal
from .battery_model import BatteryParams
from .utils import DaySimulationResult, SimulationResult
from .forecaster import get_forecast
from .stage1_da_scheduler import solve_da_schedule
from .stage2_rt_mpc import solve_rt_mpc
from .globals import DELTA_T, PRICE_NODE

def plot_day_simulation(
    day_result: DaySimulationResult,
    actual_da_prices: np.ndarray,
    actual_rt_prices: np.ndarray,
    save_path: str = None
) -> None:
    """
    Plot results from a single day simulation.
    """
    try:
        # Create time axis (15-min intervals)
        times = np.arange(len(day_result.power_trajectory)) / 4.0
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # 1. Revenue accumulation
        step_revenues = []
        for t in range(len(day_result.power_trajectory)):
            step_rev = -(actual_rt_prices[t] * day_result.power_trajectory[t] * DELTA_T)
            step_revenues.append(step_rev)
        
        cum_revenue = np.cumsum(step_revenues)
        
        ax1.set_title(f"Day Simulation Results - {day_result.date.date()}")
        ax1.plot(times, cum_revenue, "g-", linewidth=2, label=f"Cumulative Revenue (Total: ${day_result.total_revenue:.2f})")
        ax1.set_ylabel("Cumulative Revenue [$]")
        ax1.legend()
        ax1.grid(True)
        
        # 2. Prices
        ax2.set_title("Market Prices")
        ax2.plot(times, actual_da_prices, "b-", alpha=0.7, label="DA Prices")
        ax2.plot(times, actual_rt_prices, "r-", alpha=0.7, label="RT Prices")
        ax2.set_ylabel("Price [$/MWh]")
        ax2.legend()
        ax2.grid(True)
        
        # 3. Battery Power
        ax3.set_title("Battery Power Dispatch")
        ax3.plot(times, day_result.power_trajectory, "purple", linewidth=1.5)
        ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
        ax3.set_ylabel("Power [MW]")
        ax3.text(0.02, 0.98, "Positive = Charge\nNegative = Discharge", 
                transform=ax3.transAxes, verticalalignment="top", 
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax3.grid(True)
        
        # 4. State of Charge
        ax4.set_title("Battery State of Charge")
        soc_times = np.arange(len(day_result.soc_trajectory)) / 4.0
        ax4.plot(soc_times, day_result.soc_trajectory, "orange", linewidth=2)
        ax4.set_ylabel("SoC [0-1]")
        ax4.set_xlabel("Time (Hours)")
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Day plot saved to: {os.path.abspath(save_path)}")
        
        plt.close()
        
    except Exception as e:
        print(f"Day plotting error: {e}")

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

def plot_multi_day_simulation(
    results: SimulationResult,
    save_path: str = None
) -> None:
    """
    Plot results from multi-day simulation.
    """
    try:
        n_days = len(results.daily_results)
        dates = [day.date for day in results.daily_results]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # 1. Cumulative Revenue Over Days
        ax1.set_title("Multi-Day Simulation Results")
        ax1.plot(dates, results.cumulative_revenue, "g-", linewidth=2, marker="o", 
                label=f"Total Revenue: ${results.total_revenue:.2f}")
        ax1.set_ylabel("Cumulative Revenue [$]")
        ax1.legend()
        ax1.grid(True)
        ax1.tick_params(axis="x", rotation=45)
        
        # 2. Daily Revenue Breakdown
        daily_revenues = [day.total_revenue for day in results.daily_results]
        da_revenues = [day.da_revenue for day in results.daily_results]
        rt_revenues = [day.rt_revenue for day in results.daily_results]
        
        ax2.set_title("Daily Revenue Breakdown")
        ax2.bar(dates, da_revenues, alpha=0.7, label="DA Revenue", color="blue")
        ax2.bar(dates, rt_revenues, bottom=da_revenues, alpha=0.7, label="RT Revenue", color="red")
        ax2.set_ylabel("Daily Revenue [$]")
        ax2.legend()
        ax2.grid(True, axis="y")
        ax2.tick_params(axis="x", rotation=45)
        
        # 3. Final SOC Each Day
        final_socs = [day.final_soc for day in results.daily_results]
        ax3.set_title("End-of-Day State of Charge")
        ax3.plot(dates, final_socs, "orange", linewidth=2, marker="s")
        ax3.set_ylabel("Final SOC [0-1]")
        ax3.set_xlabel("Date")
        ax3.grid(True)
        ax3.tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Multi-day plot saved to: {os.path.abspath(save_path)}")
        
        plt.close()
        
    except Exception as e:
        print(f"Multi-day plotting error: {e}")

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
