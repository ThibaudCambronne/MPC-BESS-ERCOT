import logging
import os
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .battery_model import BatteryParams
from .forecaster import get_forecast, get_forecasts_for_da
from .globals import DELTA_T, TIME_STEPS_PER_HOUR
from .stage1_da_scheduler import solve_da_schedule
from .stage2_rt_mpc import solve_rt_mpc
from .utils.utils import DAScheduleResult, DaySimulationResult, SimulationResult


def plot_day_simulation(
    day_result: DaySimulationResult,
    actual_da_prices: np.ndarray,
    actual_rt_prices: np.ndarray,
    da_schedule: Optional[DAScheduleResult] = None,
    save_path: str = None,
) -> None:
    """
    Plot results from a single day simulation with real timestamps.
    Shows both DA planned dispatch and actual RT dispatch.
    """
    try:
        # Create real timestamp axis (15-min intervals)
        day_start = day_result.date
        timestamps = pd.date_range(
            start=day_start, periods=len(day_result.power_trajectory), freq="15min"
        )
        soc_timestamps = pd.date_range(
            start=day_start, periods=len(day_result.soc_trajectory), freq="15min"
        )

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

        # 1. Revenue accumulation - use pre-calculated values from simulation
        cum_da_revenue = np.cumsum(day_result.da_step_revenues)
        cum_rt_revenue = np.cumsum(day_result.rt_step_revenues)
        cum_total_revenue = cum_da_revenue + cum_rt_revenue

        ax1.set_title(
            f"Day Simulation Results - {day_result.date.date()}",
            fontsize=14,
            fontweight="bold",
        )
        ax1.plot(
            timestamps,
            cum_total_revenue,
            "g-",
            linewidth=2,
            label=f"Total Revenue: ${day_result.total_revenue:.2f}",
        )
        ax1.plot(
            timestamps,
            cum_da_revenue,
            "b--",
            linewidth=1.5,
            alpha=0.7,
            label=f"DA Revenue: ${day_result.da_revenue:.2f}",
        )
        ax1.plot(
            timestamps,
            cum_rt_revenue,
            "r--",
            linewidth=1.5,
            alpha=0.7,
            label=f"RT Revenue: ${day_result.rt_revenue:.2f}",
        )
        ax1.set_ylabel("Cumulative Revenue [$]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Prices - Show both FORECAST (what DA scheduler saw) and ACTUAL
        ax2.set_title("Market Prices: Forecast (used by DA) vs Actual")

        # Plot actual prices (solid lines)
        ax2.plot(
            timestamps,
            actual_da_prices,
            "b-",
            alpha=0.7,
            linewidth=1.5,
            label="DA Actual",
        )
        ax2.plot(
            timestamps,
            actual_rt_prices,
            "r-",
            alpha=0.7,
            linewidth=1.5,
            label="RT Actual",
        )

        # Plot forecast prices if available (dashed lines)
        if (
            da_schedule is not None
            and da_schedule.da_price_forecast is not None
            and da_schedule.rt_price_forecast is not None
        ):
            da_forecast = da_schedule.da_price_forecast[: len(timestamps)]
            rt_forecast = da_schedule.rt_price_forecast[: len(timestamps)]
            ax2.plot(
                timestamps,
                da_forecast,
                "b--",
                alpha=0.5,
                linewidth=1.5,
                label="DA Forecast (used by optimizers)",
            )
            ax2.plot(
                timestamps,
                rt_forecast,
                "r--",
                alpha=0.5,
                linewidth=1.5,
                label="RT Forecast (used by optimizers)",
            )

        ax2.set_ylabel("Price [$/MWh]")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Battery Power Dispatch - COMPARE DA PLAN VS RT ACTUAL
        ax3.set_title("Battery Power Dispatch: DA Plan vs RT Actual")

        # Plot DA planned dispatch
        if da_schedule is not None:
            da_power = da_schedule.power_dispatch_schedule[: len(timestamps)]
            ax3.plot(
                timestamps,
                da_power,
                "gray",
                linewidth=2,
                alpha=0.6,
                linestyle="--",
                label="DA Plan (computed day before)",
            )

        # Plot actual RT dispatch
        ax3.plot(
            timestamps,
            day_result.power_trajectory,
            "purple",
            linewidth=2,
            label="RT Actual (MPC dispatched)",
        )
        ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

        # Shade discharge/charge regions for actual
        ax3.fill_between(
            timestamps,
            0,
            day_result.power_trajectory,
            where=(day_result.power_trajectory < 0),
            color="red",
            alpha=0.2,
            label="Actual Discharge",
        )
        ax3.fill_between(
            timestamps,
            0,
            day_result.power_trajectory,
            where=(day_result.power_trajectory > 0),
            color="blue",
            alpha=0.2,
            label="Actual Charge",
        )

        ax3.set_ylabel("Power [MW]")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)

        # 4. State of Charge - COMPARE DA PLAN VS RT ACTUAL
        ax4.set_title("Battery State of Charge: DA Plan vs RT Actual")

        # Plot DA planned SOC
        if da_schedule is not None:
            da_soc_timestamps = pd.date_range(
                start=day_start, periods=len(da_schedule.soc_schedule), freq="15min"
            )
            ax4.plot(
                da_soc_timestamps,
                da_schedule.soc_schedule,
                "gray",
                linewidth=2,
                alpha=0.6,
                linestyle="--",
                label="DA Plan",
            )

        # Plot actual RT SOC
        ax4.plot(
            soc_timestamps,
            day_result.soc_trajectory,
            "orange",
            linewidth=2,
            label="RT Actual",
        )
        ax4.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Target (50%)")
        ax4.set_ylabel("SoC [0-1]")
        ax4.set_xlabel("Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Format x-axis to show time nicely
        import matplotlib.dates as mdates

        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax4.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

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
    horizon_type: Literal["shrinking", "receding"] = "receding",
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

    actual_da_prices = get_forecast(
        data=data,
        current_time=day_start,
        horizon_hours=24,
        market="DA",
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
            method=forecast_method,
        )

        # Solve RT MPC
        rt_result = solve_rt_mpc(
            current_time=current_time,
            current_soc=current_soc,
            rt_price_forecast=rt_price_forecast,
            da_commitments=da_schedule,
            battery=battery,
            horizon_type=horizon_type,
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

        # Clamp SOC to valid range and log if clamping occurs
        if not (battery.soc_min <= soc_next <= battery.soc_max):
            logging.warning(
                f"SOC out of bounds at {current_time}: {soc_next}, clamping to [{battery.soc_min}, {battery.soc_max}]"
            )
            soc_next = np.clip(soc_next, battery.soc_min, battery.soc_max)

        soc_trajectory[t + 1] = soc_next

    # === Calculate Revenues ===
    # Revenue calculation based on actual market structure:
    # 1. DA Market: Revenue from power bids placed in DA market at actual DA prices
    # 2. RT Market: Revenue from difference between actual dispatch and DA bids at RT prices

    # Get actual DA prices for the day
    actual_rt_prices_arr = np.array(actual_rt_prices)
    actual_da_prices_arr = np.array(actual_da_prices)

    # Get DA power bids (what we committed to in DA market)
    da_power_bids = da_schedule.da_energy_bids[: len(power_trajectory)]

    # Calculate RT market imbalance: actual MPC dispatch - DA bids
    rt_imbalance = power_trajectory - da_power_bids

    # Calculate step-by-step revenues for detailed analysis
    da_step_revenues = np.array(
        [
            -(actual_da_prices_arr[t] * da_power_bids[t] * DELTA_T)
            for t in range(len(power_trajectory))
        ]
    )
    rt_step_revenues = np.array(
        [
            -(actual_rt_prices_arr[t] * rt_imbalance[t] * DELTA_T)
            for t in range(len(power_trajectory))
        ]
    )

    # Calculate total revenues
    da_revenue = float(np.sum(da_step_revenues))
    rt_revenue = float(np.sum(rt_step_revenues))
    total_revenue = da_revenue + rt_revenue

    return DaySimulationResult(
        date=day_start,
        total_revenue=total_revenue,
        da_revenue=da_revenue,
        rt_revenue=rt_revenue,
        soc_trajectory=soc_trajectory,
        power_trajectory=power_trajectory,
        final_soc=soc_trajectory[-1],
        da_step_revenues=da_step_revenues,
        rt_step_revenues=rt_step_revenues,
        da_power_bids=da_power_bids,
        rt_imbalance=rt_imbalance,
    )


def plot_multi_day_simulation(
    results: SimulationResult,
    data: pd.DataFrame,
    battery: BatteryParams,
    save_path: str = None,
) -> None:
    """
    Plot comprehensive multi-day simulation results with 4 key visualizations.
    Includes perfect forecast comparison by running DA scheduler with perfect forecasts.
    """
    try:
        n_days = len(results.daily_results)
        dates = [day.date for day in results.daily_results]

        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # === 1. CUMULATIVE REVENUE COMPARISON (15-min resolution) ===
        ax1.set_title(
            f"Cumulative Revenue Comparison ({n_days} days)", fontweight="bold"
        )

        # Build continuous time series for actual results
        all_timestamps = []
        actual_total_revenue = []
        actual_da_revenue = []
        da_only_revenue = []

        for day_result in results.daily_results:
            # Create timestamps for this day (15-min intervals)
            day_timestamps = pd.date_range(
                start=day_result.date,
                periods=len(day_result.power_trajectory),
                freq="15min",
            )

            # Accumulate revenues within this day, then add to total
            day_cum_total = np.cumsum(
                day_result.da_step_revenues + day_result.rt_step_revenues
            )
            day_cum_da = np.cumsum(day_result.da_step_revenues)

            # Add previous days' totals if not first day
            if len(actual_total_revenue) > 0:
                prev_total = actual_total_revenue[-1]
                prev_da = actual_da_revenue[-1]
                day_cum_total += prev_total
                day_cum_da += prev_da

            all_timestamps.extend(day_timestamps)
            actual_total_revenue.extend(day_cum_total)
            actual_da_revenue.extend(day_cum_da)
            da_only_revenue.extend(day_cum_da)  # DA-only is just DA revenue

        # === COMPUTE IDEAL REVENUE WITH PERFECT FORECASTS ===
        print("Computing ideal revenue with perfect forecasts...")
        ideal_total_revenue = []
        current_soc = 0.5  # Starting SOC

        for i, day_result in enumerate(results.daily_results):
            day_start = day_result.date

            # Get perfect forecasts for this day
            perfect_da_prices = get_forecast(
                data=data,
                current_time=day_start,
                horizon_hours=24,
                market="DA",
                method="perfect",
            )
            perfect_rt_prices = get_forecast(
                data=data,
                current_time=day_start,
                horizon_hours=24,
                market="RT",
                method="perfect",
            )

            # Run DA scheduler with perfect forecasts
            perfect_da_schedule = solve_da_schedule(
                da_price_forecast=perfect_da_prices,
                rt_price_forecast=perfect_rt_prices,
                battery=battery,
                initial_soc=current_soc,
                end_of_day_soc=0.5,
            )

            # Calculate perfect revenue for this day using actual prices
            actual_da_prices_arr = np.array(perfect_da_prices)  # Already perfect
            actual_rt_prices_arr = np.array(perfect_rt_prices)  # Already perfect

            perfect_da_bids = perfect_da_schedule.da_energy_bids[
                :96
            ]  # 96 intervals per day
            perfect_power = perfect_da_schedule.power_dispatch_schedule[:96]
            perfect_rt_imbalance = perfect_power - perfect_da_bids

            # Calculate step-by-step perfect revenues
            perfect_da_step = np.array(
                [
                    -(actual_da_prices_arr[t] * perfect_da_bids[t] * DELTA_T)
                    for t in range(96)
                ]
            )
            perfect_rt_step = np.array(
                [
                    -(actual_rt_prices_arr[t] * perfect_rt_imbalance[t] * DELTA_T)
                    for t in range(96)
                ]
            )
            perfect_total_step = perfect_da_step + perfect_rt_step

            # Accumulate perfect revenues
            day_cum_perfect = np.cumsum(perfect_total_step)

            # Add previous days' totals if not first day
            if len(ideal_total_revenue) > 0:
                prev_perfect = ideal_total_revenue[-1]
                day_cum_perfect += prev_perfect

            ideal_total_revenue.extend(day_cum_perfect)
            current_soc = perfect_da_schedule.soc_schedule[
                -1
            ]  # Update SOC for next day

        # Plot all revenue curves
        ax1.plot(
            all_timestamps,
            ideal_total_revenue,
            "gold",
            linewidth=2,
            label=f"Ideal (Perfect Forecast): ${ideal_total_revenue[-1]:.2f}",
        )
        ax1.plot(
            all_timestamps,
            actual_total_revenue,
            "g-",
            linewidth=2,
            label=f"Actual (Persistence): ${actual_total_revenue[-1]:.2f}",
        )
        ax1.plot(
            all_timestamps,
            da_only_revenue,
            "b--",
            linewidth=1.5,
            label=f"DA-Only Trading: ${da_only_revenue[-1]:.2f}",
        )

        # Add day boundaries
        for day_result in results.daily_results[1:]:
            ax1.axvline(x=day_result.date, color="gray", linestyle=":", alpha=0.5)

        ax1.set_ylabel("Cumulative Revenue [$]")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # === 2. DAILY REVENUE BREAKDOWN (SIDE-BY-SIDE BARS) ===
        daily_revenues = [day.total_revenue for day in results.daily_results]
        da_revenues = [day.da_revenue for day in results.daily_results]
        rt_revenues = [day.rt_revenue for day in results.daily_results]

        ax2.set_title("Daily Revenue Breakdown", fontweight="bold")

        x_pos = np.arange(n_days)
        width = 0.35

        ax2.bar(
            x_pos - width / 2,
            da_revenues,
            width,
            label="DA Revenue",
            color="blue",
            alpha=0.7,
        )
        ax2.bar(
            x_pos + width / 2,
            rt_revenues,
            width,
            label="RT Revenue",
            color="red",
            alpha=0.7,
        )

        ax2.set_ylabel("Daily Revenue [$]")
        ax2.set_xlabel("Day")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([d.strftime("%m/%d") for d in dates], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # === 3. END-OF-DAY SOC ===
        final_socs = [day.final_soc for day in results.daily_results]
        ax3.set_title("End-of-Day State of Charge", fontweight="bold")
        ax3.plot(dates, final_socs, "orange", linewidth=2, marker="s", markersize=4)
        ax3.axhline(y=0.5, color="gray", linestyle=":", alpha=0.7, label="Target (50%)")
        ax3.set_ylabel("Final SOC [0-1]")
        ax3.set_xlabel("Date")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        if n_days > 7:
            ax3.tick_params(axis="x", rotation=45)

        # === 4. CONTINUOUS SOC TRAJECTORY ===
        ax4.set_title("Continuous SOC Trajectory", fontweight="bold")

        # Build continuous SOC trajectory
        all_soc_timestamps = []
        all_soc_values = []

        for day_result in results.daily_results:
            day_soc_timestamps = pd.date_range(
                start=day_result.date,
                periods=len(day_result.soc_trajectory),
                freq="15min",
            )
            all_soc_timestamps.extend(day_soc_timestamps)
            all_soc_values.extend(day_result.soc_trajectory)

        ax4.plot(all_soc_timestamps, all_soc_values, "orange", linewidth=1.5)
        ax4.axhline(y=0.5, color="gray", linestyle=":", alpha=0.7, label="Target (50%)")

        # Add day boundaries
        for day_result in results.daily_results[1:]:
            ax4.axvline(x=day_result.date, color="gray", linestyle=":", alpha=0.5)

        ax4.set_ylabel("SOC [0-1]")
        ax4.set_xlabel("Time")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

        # Format x-axes for time plots
        import matplotlib.dates as mdates

        for ax in [ax1, ax4]:
            if n_days <= 7:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            elif n_days <= 31:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                ax.xaxis.set_major_locator(
                    mdates.DayLocator(interval=max(1, n_days // 10))
                )
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Multi-day plot saved to: {os.path.abspath(save_path)}")

        plt.close()

    except Exception as e:
        print(f"Multi-day plotting error: {e}")
        import traceback

        traceback.print_exc()

        # 6. POWER DISPATCH OVERVIEW
        ax6.set_title("Power Dispatch Overview", fontweight="bold")

        # Calculate daily averages for power dispatch
        daily_avg_power = []
        daily_max_charge = []
        daily_max_discharge = []

        for day_result in results.daily_results:
            power = day_result.power_trajectory
            daily_avg_power.append(np.mean(power))
            daily_max_charge.append(
                np.max(power[power > 0]) if np.any(power > 0) else 0
            )
            daily_max_discharge.append(
                np.min(power[power < 0]) if np.any(power < 0) else 0
            )

        if n_days <= 31:
            x_pos = np.arange(n_days)
            ax6.bar(
                x_pos, daily_max_charge, alpha=0.7, color="blue", label="Max Charge"
            )
            ax6.bar(
                x_pos,
                daily_max_discharge,
                alpha=0.7,
                color="red",
                label="Max Discharge",
            )
            ax6.set_xticks(x_pos[:: max(1, n_days // 10)])
            ax6.set_xticklabels(
                [
                    dates[i].strftime("%m/%d")
                    for i in range(0, n_days, max(1, n_days // 10))
                ],
                rotation=45 if n_days > 14 else 0,
            )
        else:
            ax6.plot(dates, daily_max_charge, "b-", alpha=0.7, label="Max Charge")
            ax6.plot(dates, daily_max_discharge, "r-", alpha=0.7, label="Max Discharge")
            ax6.tick_params(axis="x", rotation=45)

        ax6.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
        ax6.set_ylabel("Power [MW]")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

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
    end_of_day_soc: float = 0.5,
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
    dates = pd.date_range(start=start, periods=n_days, freq="D")

    # Initialize storage
    daily_results = []
    cumulative_revenue = np.zeros(n_days)

    # Track SOC and DA schedules
    current_soc = initial_soc
    da_schedules = {}  # Maps date -> DAScheduleResult

    print(f"\n{'=' * 60}")
    print("SIMULATION SETUP")
    print(f"{'=' * 60}")
    print(f"Simulating {n_days} days: {dates[0].date()} to {dates[-1].date()}")
    print(f"Initial SOC: {initial_soc:.1%}, Target EOD SOC: {end_of_day_soc:.1%}")
    print(f"Forecast method: {forecast_method}, Horizon type: {horizon_type}")
    print(f"{'=' * 60}\n")

    # Pre-compute DA schedules for all days
    # For each day, run DA scheduler at 10 AM the previous day
    for i, sim_date in enumerate(dates):
        # DA market runs at 10 AM the day before
        da_time = sim_date - pd.Timedelta(days=1) + pd.Timedelta(hours=10)

        print(f"[{da_time}] Running DA scheduler for {sim_date.date()}...")

        # Get forecasts using the special DA function
        da_prices, rt_prices = get_forecasts_for_da(
            data=data, current_time=da_time, horizon_hours=24, method=forecast_method
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
            end_of_day_soc=end_of_day_soc,
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
        da_schedules=da_schedules,
    )
