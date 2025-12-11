import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.forecaster import get_forecast
from src.globals import DELTA_T, TIME_STEPS_PER_HOUR

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.battery_model import BatteryParams
from src.stage2_rt_mpc import solve_rt_mpc
from src.utils import DAScheduleResult, load_ercot_data


def generate_synthetic_data(start_date, steps_needed):
    """Generates synthetic price and DA schedule data."""
    data = load_ercot_data()
    current_time = pd.Timestamp("2025-02-01 10:00:00")
    da_prices = get_forecast(
        data,
        current_time=current_time,
        horizon_hours=24,
        market="DA",
        method="persistence",
        verbose=True,
    )[-TIME_STEPS_PER_HOUR * 24 :]
    rt_prices = get_forecast(
        data,
        current_time=current_time,
        horizon_hours=24,
        market="RT",
        method="persistence",
        verbose=True,
    )[-TIME_STEPS_PER_HOUR * 24 :]

    # 3. Synthetic DA Schedule (MW) - Simple Arbitrage
    da_bids = np.zeros(steps_needed)
    charge_mask = da_prices < 25
    discharge_mask = da_prices > 50
    da_bids[charge_mask] = 5.0
    da_bids[discharge_mask] = -5.0

    idx = pd.date_range(start=start_date, periods=steps_needed, freq="15min")
    rt_prices = pd.Series(data=rt_prices, index=idx)

    return rt_prices, da_bids


def run_heuristic_strategy(rt_prices, battery, steps):
    """
    A simple baseline strategy:
    - Charge if Price < 25th percentile
    - Discharge if Price > 75th percentile
    - Otherwise idle
    """
    print("Running Heuristic Baseline...")
    price_values = rt_prices.values
    p25 = np.percentile(price_values, 25)
    p75 = np.percentile(price_values, 75)

    current_soc = 0.5
    soc_history = [current_soc]
    revenue_history = []

    for t in range(steps):
        price = price_values[t]

        # Simple Logic
        if price <= p25:
            # Charge at max power
            p_cmd = battery.power_max_mw
        elif price >= p75:
            # Discharge at max power
            p_cmd = -battery.power_max_mw
        else:
            p_cmd = 0.0

        # Check Physics Limits
        if p_cmd > 0:  # Charge
            energy_to_add = p_cmd * battery.efficiency_charge * DELTA_T
            if (
                current_soc * battery.capacity_mwh + energy_to_add
                > battery.soc_max * battery.capacity_mwh
            ):
                p_cmd = 0  # Simple cut-off
        elif p_cmd < 0:  # Discharge
            energy_to_remove = abs(p_cmd) / battery.efficiency_discharge * DELTA_T
            if (
                current_soc * battery.capacity_mwh - energy_to_remove
                < battery.soc_min * battery.capacity_mwh
            ):
                p_cmd = 0  # Simple cut-off

        # Dynamics
        if p_cmd >= 0:
            e_next = (
                current_soc * battery.capacity_mwh
                + (p_cmd * battery.efficiency_charge) * DELTA_T
            )
        else:
            e_next = (
                current_soc * battery.capacity_mwh
                + (p_cmd / battery.efficiency_discharge) * DELTA_T
            )

        current_soc = e_next / battery.capacity_mwh
        current_soc = np.clip(current_soc, 0.0, 1.0)

        step_revenue = -(price * p_cmd * DELTA_T)

        soc_history.append(current_soc)
        revenue_history.append(step_revenue)

    return revenue_history, soc_history


def run_simulation_comparison():
    print("\n=============================================================")
    print("   COMPARING MPC vs BASELINE STRATEGY")
    print("=============================================================")

    # --- Setup ---
    sim_date = pd.Timestamp("2025-06-15 00:00:00")
    horizon_hours = 4
    sim_hours = 24
    steps_per_hour = 4
    sim_steps = sim_hours * steps_per_hour

    # Buffer for receding horizon
    total_data_steps = sim_steps + (horizon_hours * steps_per_hour) + 10

    battery = BatteryParams(
        capacity_mwh=20.0,
        power_max_mw=10.0,
        soc_min=0.05,
        soc_max=0.95,
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
    )

    # Data
    rt_forecast_full, da_bids_full = generate_synthetic_data(sim_date, total_data_steps)

    # 1. Run Heuristic Baseline
    rev_baseline_step, soc_baseline = run_heuristic_strategy(
        rt_forecast_full, battery, sim_steps
    )
    cum_rev_baseline = np.cumsum(rev_baseline_step)

    # 2. Run MPC
    print("Running MPC Strategy...")
    da_schedule = DAScheduleResult(
        da_energy_bids=da_bids_full,
        reg_up_capacity=np.zeros(total_data_steps),
        reg_down_capacity=np.zeros(total_data_steps),
        planned_soc=np.zeros(total_data_steps + 1),
        expected_revenue=0.0,
    )

    current_soc = 0.5
    soc_mpc = [current_soc]
    rev_mpc_step = []

    for t in range(sim_steps):
        if t % 24 == 0:
            print(f"MPC Step {t}/{sim_steps}...")

        current_time = sim_date + pd.Timedelta(minutes=15 * t)

        result = solve_rt_mpc(
            current_time=current_time,
            current_soc=current_soc,
            rt_price_forecast=rt_forecast_full,
            da_commitments=da_schedule,
            battery=battery,
            horizon_type="receding",
            horizon_hours=horizon_hours,
        )

        # Fallback if solver fails
        power_setpoint = (
            result.power_setpoint
            if result.solve_status in ["optimal", "max_iter"]
            else 0.0
        )

        # Plant Dynamics
        if power_setpoint >= 0:
            e_next = (
                current_soc * battery.capacity_mwh
                + (power_setpoint * battery.efficiency_charge) * DELTA_T
            )
        else:
            e_next = (
                current_soc * battery.capacity_mwh
                + (power_setpoint / battery.efficiency_discharge) * DELTA_T
            )

        current_soc = np.clip(e_next / battery.capacity_mwh, 0.0, 1.0)

        # Revenue
        step_price = rt_forecast_full.iloc[t]
        rev_mpc_step.append(-(step_price * power_setpoint * DELTA_T))
        soc_mpc.append(current_soc)

    cum_rev_mpc = np.cumsum(rev_mpc_step)

    # --- Plotting Benefit ---
    try:
        times = np.arange(sim_steps) / 4.0

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # 1. Cumulative Revenue
        ax1.set_title("Benefit Analysis: Cumulative Revenue")
        ax1.plot(
            times,
            cum_rev_mpc,
            "g-",
            linewidth=2,
            label=f"MPC (Total: ${cum_rev_mpc[-1]:.2f})",
        )
        ax1.plot(
            times,
            cum_rev_baseline,
            "k--",
            linewidth=1.5,
            label=f"Heuristic (Total: ${cum_rev_baseline[-1]:.2f})",
        )

        # Fill area to show benefit
        ax1.fill_between(
            times,
            cum_rev_mpc,
            cum_rev_baseline,
            where=(cum_rev_mpc > cum_rev_baseline),
            interpolate=True,
            color="green",
            alpha=0.1,
            label="Added Value",
        )
        ax1.set_ylabel("Cumulative Revenue [$]")
        ax1.legend()
        ax1.grid(True)

        # 2. Prices
        ax2.set_title("Market Prices")
        ax2.plot(times, rt_forecast_full.values[:sim_steps], "b-", alpha=0.6)
        ax2.set_ylabel("Price [$/MWh]")
        ax2.grid(True)

        # 3. State of Charge
        ax3.set_title("Battery Operation (SoC)")
        ax3.plot(times, soc_mpc[:-1], "g-", label="MPC SoC")
        ax3.plot(times, soc_baseline[:-1], "k--", alpha=0.5, label="Heuristic SoC")
        ax3.set_ylabel("SoC [0-1]")
        ax3.set_xlabel("Time (Hours)")
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()

        output_path = "mpc_benefit_analysis.png"
        plt.savefig(output_path, dpi=300)
        print(f"\nPlot saved to: {os.path.abspath(output_path)}")
        plt.close()

        print(
            f"\nFinal Benefit: ${cum_rev_mpc[-1] - cum_rev_baseline[-1]:.2f} over 24h"
        )

    except Exception as e:
        print(f"Plotting Error: {e}")


if __name__ == "__main__":
    run_simulation_comparison()
