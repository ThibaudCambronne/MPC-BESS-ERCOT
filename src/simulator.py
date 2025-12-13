import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Literal, Optional
from .battery_model import BatteryParams
from .utils import DaySimulationResult, SimulationResult, DAScheduleResult
from .forecaster import get_forecast, get_forecasts_for_da
from .globals import DELTA_T, TIME_STEPS_PER_HOUR
from .stage1_da_scheduler import solve_da_schedule
from .stage2_rt_mpc import solve_rt_mpc
from .globals import DELTA_T, PRICE_NODE

def plot_day_simulation(
    day_result: DaySimulationResult,
    actual_da_prices: np.ndarray,
    actual_rt_prices: np.ndarray,
    da_schedule: Optional[DAScheduleResult] = None,
    save_path: str = None
) -> None:
    """
    Plot results from a single day simulation with real timestamps.
    Shows both DA planned dispatch and actual RT dispatch.
    """
    try:
        # Create real timestamp axis (15-min intervals)
        day_start = day_result.date
        timestamps = pd.date_range(
            start=day_start,
            periods=len(day_result.power_trajectory),
            freq='15min'
        )
        soc_timestamps = pd.date_range(
            start=day_start,
            periods=len(day_result.soc_trajectory),
            freq='15min'
        )

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

        # 1. Revenue accumulation - separate DA and RT components
        da_power_bids = da_schedule.da_energy_bids[:len(day_result.power_trajectory)] if da_schedule else np.zeros(len(day_result.power_trajectory))
        rt_imbalance = day_result.power_trajectory - da_power_bids
        
        da_step_revenues = []
        rt_step_revenues = []
        for t in range(len(day_result.power_trajectory)):
            da_step_rev = -(actual_da_prices[t] * da_power_bids[t] * DELTA_T)
            rt_step_rev = -(actual_rt_prices[t] * rt_imbalance[t] * DELTA_T)
            da_step_revenues.append(da_step_rev)
            rt_step_revenues.append(rt_step_rev)
        
        cum_da_revenue = np.cumsum(da_step_revenues)
        cum_rt_revenue = np.cumsum(rt_step_revenues)
        cum_total_revenue = cum_da_revenue + cum_rt_revenue

        ax1.set_title(f"Day Simulation Results - {day_result.date.date()}", fontsize=14, fontweight='bold')
        ax1.plot(timestamps, cum_total_revenue, "g-", linewidth=2, label=f"Total Revenue: ${day_result.total_revenue:.2f}")
        ax1.plot(timestamps, cum_da_revenue, "b--", linewidth=1.5, alpha=0.7, label=f"DA Revenue: ${day_result.da_revenue:.2f}")
        ax1.plot(timestamps, cum_rt_revenue, "r--", linewidth=1.5, alpha=0.7, label=f"RT Revenue: ${day_result.rt_revenue:.2f}")
        ax1.set_ylabel("Cumulative Revenue [$]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Prices - Show both FORECAST (what DA scheduler saw) and ACTUAL
        ax2.set_title("Market Prices: Forecast (used by DA) vs Actual")

        # Plot actual prices (solid lines)
        ax2.plot(timestamps, actual_da_prices, "b-", alpha=0.7, linewidth=1.5, label="DA Actual")
        ax2.plot(timestamps, actual_rt_prices, "r-", alpha=0.7, linewidth=1.5, label="RT Actual")

        # Plot forecast prices if available (dashed lines)
        if (da_schedule is not None and
            da_schedule.da_price_forecast is not None and
            da_schedule.rt_price_forecast is not None):
            da_forecast = da_schedule.da_price_forecast[:len(timestamps)]
            rt_forecast = da_schedule.rt_price_forecast[:len(timestamps)]
            ax2.plot(timestamps, da_forecast, "b--", alpha=0.5, linewidth=1.5,
                    label="DA Forecast (used by optimizers)")
            ax2.plot(timestamps, rt_forecast, "r--", alpha=0.5, linewidth=1.5,
                    label="RT Forecast (used by optimizers)")

        ax2.set_ylabel("Price [$/MWh]")
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Battery Power Dispatch - COMPARE DA PLAN VS RT ACTUAL
        ax3.set_title("Battery Power Dispatch: DA Plan vs RT Actual")

        # Plot DA planned dispatch
        if da_schedule is not None:
            da_power = da_schedule.power_dispatch_schedule[:len(timestamps)]
            ax3.plot(timestamps, da_power, "gray", linewidth=2, alpha=0.6,
                    linestyle='--', label='DA Plan (computed day before)')

        # Plot actual RT dispatch
        ax3.plot(timestamps, day_result.power_trajectory, "purple", linewidth=2,
                label='RT Actual (MPC dispatched)')
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

        # Shade discharge/charge regions for actual
        ax3.fill_between(timestamps, 0, day_result.power_trajectory,
                         where=(day_result.power_trajectory < 0),
                         color='red', alpha=0.2, label='Actual Discharge')
        ax3.fill_between(timestamps, 0, day_result.power_trajectory,
                         where=(day_result.power_trajectory > 0),
                         color='blue', alpha=0.2, label='Actual Charge')

        ax3.set_ylabel("Power [MW]")
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # 4. State of Charge - COMPARE DA PLAN VS RT ACTUAL
        ax4.set_title("Battery State of Charge: DA Plan vs RT Actual")

        # Plot DA planned SOC
        if da_schedule is not None:
            da_soc_timestamps = pd.date_range(
                start=day_start,
                periods=len(da_schedule.soc_schedule),
                freq='15min'
            )
            ax4.plot(da_soc_timestamps, da_schedule.soc_schedule, "gray",
                    linewidth=2, alpha=0.6, linestyle='--', label='DA Plan')

        # Plot actual RT SOC
        ax4.plot(soc_timestamps, day_result.soc_trajectory, "orange", linewidth=2,
                label='RT Actual')
        ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Target (50%)')
        ax4.set_ylabel("SoC [0-1]")
        ax4.set_xlabel("Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Format x-axis to show time nicely
        import matplotlib.dates as mdates
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax4.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

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
    da_schedule: "DAScheduleResult",
    battery: BatteryParams,
    forecast_method: Literal["persistence", "perfect"],
    horizon_type: Literal["shrinking", "receding"] = "receding"
) -> DaySimulationResult:
    """
    Simulate one complete day (24 hours) using pre-computed DA commitments.

    The DA schedule should have been computed at 10:00 AM the previous day.
    This function runs the RT MPC from 00:00 to 23:45 of the simulated day.

    Power Sign Convention (CONSISTENT ACROSS ALL MODULES):
    - Positive power = CHARGING (battery consuming power from grid)
    - Negative power = DISCHARGING (battery supplying power to grid)

    Revenue Convention:
    - When charging (p > 0): We PAY the grid, so cost > 0, revenue < 0
    - When discharging (p < 0): We RECEIVE from grid, so cost < 0, revenue > 0
    - Revenue = sum(price * |discharge_power|) - sum(price * |charge_power|)

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data
    date : pd.Timestamp
        The day to simulate (will normalize to midnight)
    initial_soc : float
        Starting SOC at 00:00 of this day
    da_schedule : DAScheduleResult
        Day-ahead commitments computed at 10 AM the previous day
    battery : BatteryParams
        Battery configuration
    forecast_method : str
        "perfect" or "persistence"
    horizon_type : str
        "receding" or "shrinking"

    Returns DaySimulationResult.
    """
    # Normalize date to midnight
    day_start = pd.Timestamp(date).normalize()
    day_end = day_start + pd.Timedelta(days=1)

    # Get actual prices for the day for revenue calculation
    actual_rt_prices = get_forecast(
        data=data,
        current_time=day_start,
        horizon_hours=24,
        market="RT",
        method="perfect",
    )

    # === Stage 2: Real-Time MPC (run every 15 minutes) ===
    num_intervals = TIME_STEPS_PER_HOUR * 24  # 96 intervals in 24 hours
    soc_trajectory = np.zeros(num_intervals + 1)
    power_trajectory = np.zeros(num_intervals)  # Actual dispatched power
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
        # Convention: positive = charging, negative = discharging
        power_setpoint = rt_result.power_setpoint
        power_trajectory[t] = power_setpoint

        # Update SOC based on actual power dispatch
        # Positive power = charging (adding energy to battery)
        # Negative power = discharging (removing energy from battery)
        if power_setpoint > 0:  # Charging
            energy_change_mwh = power_setpoint * battery.efficiency_charge * DELTA_T
        else:  # Discharging (power_setpoint < 0)
            # Energy removed from battery = |power_setpoint| / efficiency_discharge
            energy_change_mwh = power_setpoint / battery.efficiency_discharge * DELTA_T

        soc_next = current_soc + energy_change_mwh / battery.capacity_mwh

        # Clamp SOC to valid range
        if not (battery.soc_min <= soc_next <= battery.soc_max):
            raise logging.warning(f"SOC out of bounds at {current_time}: {soc_next}")

        soc_trajectory[t + 1] = soc_next

    # === Calculate Revenues ===
    # Revenue calculation based on actual market structure:
    # 1. DA Market: Revenue from power bids placed in DA market at actual DA prices
    # 2. RT Market: Revenue from difference between actual dispatch and DA bids at RT prices
    
    # Get actual DA prices for the day
    actual_da_prices = day_data[f"{PRICE_NODE}_DAM"].values
    actual_rt_prices_arr = np.array(actual_rt_prices)
    
    # Get DA power bids (what we committed to in DA market)
    da_power_bids = da_schedule.da_energy_bids[:len(power_trajectory)]
    
    # Calculate DA market revenue: DA bids × actual DA prices
    # Convention: negative power = discharge (we receive money), positive = charge (we pay)
    da_revenue = float(-np.sum(da_power_bids * actual_da_prices * DELTA_T))
    
    # Calculate RT market imbalance: actual MPC dispatch - DA bids
    rt_energy_volume= power_trajectory - da_power_bids
    
    # Calculate RT market revenue: imbalance × actual RT prices
    rt_revenue = float(-np.sum(rt_energy_volume * actual_rt_prices_arr * DELTA_T))
    
    # Total revenue = DA revenue + RT revenue
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
    n_days: int = 3,
    battery: Optional[BatteryParams] = None,
    forecast_method: Literal["persistence", "perfect"] = "perfect",
    horizon_type: Literal["shrinking", "receding"] = "receding",
    initial_soc: float = 0.5,
    end_of_day_soc: float = 0.5
) -> SimulationResult:
    """
    Run multi-day battery energy storage simulation with realistic DA market timing.

    ERCOT DAM Timing (Realistic):
    - DA market closes at 10:00 AM for the NEXT day
    - RT market runs in real-time every 15 minutes

    Example for simulating Monday-Wednesday (n_days=3, start_date=Monday):
    - Sunday 10:00 AM: Run DA scheduler for Monday
    - Monday 00:00-23:45: Run RT MPC using Sunday's DA commitments
    - Monday 10:00 AM: Run DA scheduler for Tuesday
    - Tuesday 00:00-23:45: Run RT MPC using Monday's DA commitments
    - Tuesday 10:00 AM: Run DA scheduler for Wednesday
    - Wednesday 00:00-23:45: Run RT MPC using Tuesday's DA commitments

    Parameters
    ----------
    data : pd.DataFrame
        Historical price data with DatetimeIndex
    start_date : pd.Timestamp
        First day to simulate (will run DA scheduler at 10 AM the day before)
    n_days : int
        Number of days to simulate (default: 3)
    battery : BatteryParams
        Battery configuration (default: creates new instance)
    forecast_method : Literal["persistence", "perfect"]
        - "perfect": Uses actual future prices (upper bound performance)
        - "persistence": Uses previous day's prices at same time
    horizon_type : Literal["shrinking", "receding"]
        - "receding": Fixed 24h horizon (standard MPC)
        - "shrinking": Horizon shrinks to end of day
    initial_soc : float
        Starting state of charge [0-1] at start_date 00:00 (default: 0.5)
    end_of_day_soc : float
        Target SOC at end of each day [0-1] (default: 0.5)

    Returns
    -------
    SimulationResult
        Contains daily_results, cumulative_revenue, total_revenue
    """
    # Default battery if not provided
    if battery is None:
        battery = BatteryParams()

    # Normalize start date to midnight
    start = pd.Timestamp(start_date).normalize()

    # Generate list of dates to simulate
    dates = pd.date_range(start=start, periods=n_days, freq='D')

    # Initialize storage
    daily_results = []
    cumulative_revenue = np.zeros(n_days)

    # Track SOC and DA schedules
    current_soc = initial_soc
    da_schedules = {}  # Maps date -> DAScheduleResult

    print(f"\n{'='*60}")
    print(f"SIMULATION SETUP")
    print(f"{'='*60}")
    print(f"Simulating {n_days} days: {dates[0].date()} to {dates[-1].date()}")
    print(f"Initial SOC: {initial_soc:.1%}, Target EOD SOC: {end_of_day_soc:.1%}")
    print(f"Forecast method: {forecast_method}, Horizon type: {horizon_type}")
    print(f"{'='*60}\n")

    # Pre-compute DA schedules for all days
    # For each day, run DA scheduler at 10 AM the previous day
    for i, sim_date in enumerate(dates):
        # DA market runs at 10 AM the day before
        da_time = sim_date - pd.Timedelta(days=1) + pd.Timedelta(hours=10)

        print(f"[{da_time}] Running DA scheduler for {sim_date.date()}...")

        # Get forecasts using the special DA function
        da_prices, rt_prices = get_forecasts_for_da(
            data=data,
            current_time=da_time,
            horizon_hours=24,
            method=forecast_method
        )

        # Estimate SOC at start of the target day
        # For the first day, use initial_soc
        # For subsequent days, use end_of_day_soc as estimate (actual may differ)
        if i == 0:
            est_initial_soc = initial_soc
        else:
            est_initial_soc = end_of_day_soc

        # Solve DA optimization
        da_schedule = solve_da_schedule(
            da_price_forecast=da_prices,
            rt_price_forecast=rt_prices,
            battery=battery,
            initial_soc=est_initial_soc,
            end_of_day_soc=end_of_day_soc
        )

        # Store forecast prices used (for debugging/plotting)
        da_schedule.da_price_forecast = np.array(da_prices.values)
        da_schedule.rt_price_forecast = np.array(rt_prices.values)

        da_schedules[sim_date] = da_schedule

    # Now simulate each day using the pre-computed DA schedules
    print(f"\n{'=' * 60}")
    print("RUNNING REAL-TIME SIMULATION")
    print(f"{'=' * 60}\n")

    for i, sim_date in enumerate(dates):
        print(f"[{sim_date}] Simulating day {i + 1}/{n_days}...")

        # Get DA schedule for this day (computed at 10 AM yesterday)
        da_schedule = da_schedules[sim_date]

        # Run RT MPC for the whole day
        day_result = simulate_day(
            data=data,
            date=sim_date,
            initial_soc=current_soc,
            da_schedule=da_schedule,
            battery=battery,
            forecast_method=forecast_method,
            horizon_type=horizon_type,
        )

        # Store results
        daily_results.append(day_result)

        # Update cumulative revenue
        if i == 0:
            cumulative_revenue[i] = day_result.total_revenue
        else:
            cumulative_revenue[i] = cumulative_revenue[i - 1] + day_result.total_revenue

        # Update SOC for next day
        current_soc = day_result.final_soc

        print(
            f"  Revenue: ${day_result.total_revenue:,.2f}, Final SOC: {day_result.final_soc:.1%}"
        )

    # Calculate total revenue
    total_revenue = float(sum(day.total_revenue for day in daily_results))

    return SimulationResult(
        daily_results=daily_results,
        cumulative_revenue=cumulative_revenue,
        total_revenue=total_revenue,
        da_schedules=da_schedules
    )
