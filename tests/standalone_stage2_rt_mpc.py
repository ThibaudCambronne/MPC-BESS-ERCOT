import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.battery_model import BatteryParams
from src.forecaster import get_forecast, get_forecasts_for_da
from src.globals import DELTA_T, TIME_STEPS_PER_HOUR
from src.stage1_da_scheduler import solve_da_schedule
from src.stage2_rt_mpc import solve_rt_mpc
from src.utils import load_ercot_data


def setup_simulation_data(sim_date, sim_horizon_hours, mpc_horizon_hours):
    """Generates data covering the Simulation Period + MPC Horizon Buffer."""
    print(f"Loading data for simulation starting {sim_date}...")
    data = load_ercot_data()

    if sim_date not in data.index:
        raise ValueError(f"Sim date {sim_date} not in data range!")

    da_run_time = sim_date - pd.Timedelta(hours=14)

    # 24h Forecast for Stage 1
    da_prices_stage1, rt_prices_stage1 = get_forecasts_for_da(
        data,
        current_time=da_run_time,
        horizon_hours=24,
        method="perfect",
        verbose=False,
    )

    # 28h Forecast for Stage 2 (Sim + Buffer)
    total_sim_hours = sim_horizon_hours + mpc_horizon_hours
    rt_prices_sim = get_forecast(
        data,
        current_time=sim_date,
        horizon_hours=total_sim_hours,
        market="RT",
        method="perfect",
        verbose=False,
    )

    return da_prices_stage1, rt_prices_stage1, rt_prices_sim


def run_simulation_comparison():
    print("\n=============================================================")
    print("   COMPARING MPC vs BASELINE STRATEGY (Corrected Revenue)")
    print("=============================================================")

    sim_date = pd.Timestamp("2025-06-15 00:00:00")
    sim_hours = 24
    mpc_horizon_hours = 4
    sim_steps = sim_hours * TIME_STEPS_PER_HOUR

    battery = BatteryParams()

    try:
        da_prices_s1, rt_prices_s1, rt_prices_sim = setup_simulation_data(
            sim_date, sim_hours, mpc_horizon_hours
        )
    except Exception as e:
        print(f"Data generation failed: {e}")
        return

    # --- Stage 1: Solve Day-Ahead Schedule ---
    print("\nRunning Stage 1 (DA Scheduler)...")
    da_result = solve_da_schedule(
        da_price_forecast=da_prices_s1,
        rt_price_forecast=rt_prices_s1,
        battery=battery,
        initial_soc=0.5,
        end_of_day_soc=0.5,
    )

    # Calculate Guaranteed DA Revenue (The "Sunk" Profit)
    # Assuming P > 0 is Discharging/Selling in da_energy_bids?
    # Usually: Positive Power = Injection (Sell). Negative = Withdrawal (Buy).
    # Let's verify standard: usually Sell is +, Buy is -.
    # BUT, in your heuristic: price * p_cmd (where p_cmd + is charge).
    # Wait, in heuristic: if p_cmd > 0 (Charge), Revenue is -(Price * P). Correct.
    # So P > 0 is CHARGE (Buy).
    # Therefore, we SELL when P < 0.
    # Revenue = -(Price * P).

    # Calculate DA Revenue
    da_revenue_total = np.sum(
        -(da_prices_s1.values * da_result.da_energy_bids * DELTA_T)
    )
    print(f"Stage 1 DA Revenue (Locked In): ${da_revenue_total:.2f}")

    # --- Baseline Strategy ---
    rt_prices_baseline = rt_prices_sim.iloc[:sim_steps]
    rev_baseline_step, soc_baseline = run_heuristic_strategy(
        rt_prices_baseline, battery, sim_steps
    )
    cum_rev_baseline = np.cumsum(rev_baseline_step)

    # --- Stage 2: MPC Simulation ---
    print("\nRunning Stage 2 (MPC Strategy)...")

    current_soc = 0.5
    soc_mpc = [current_soc]
    rev_mpc_cumulative = []  # Track total cumulative revenue
    power_mpc = []

    current_total_revenue = 0.0

    for t in range(sim_steps):
        current_time = sim_date + pd.Timedelta(minutes=15 * t)
        if t % 24 == 0:
            print(f"  Simulating Step {t}/{sim_steps}...")

        # Solve MPC
        result = solve_rt_mpc(
            current_time=current_time,
            current_soc=current_soc,
            rt_price_forecast=rt_prices_sim,
            da_commitments=da_result,
            battery=battery,
            horizon_type="receding",
            horizon_hours=mpc_horizon_hours,
        )

        power_setpoint = (
            result.power_setpoint
            if result.solve_status in ["optimal", "max_iter", "optimal_with_slack"]
            else 0.0
        )

        # Physics
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

        # --- CORRECT REVENUE CALCULATION ---
        # 1. Retrieve DA commitment for this step
        p_da_t = da_result.da_energy_bids[t]
        price_da_t = da_prices_s1.values[t]

        # 2. Retrieve RT Price
        price_rt_t = rt_prices_sim.iloc[t]

        # 3. Calculate Components
        # Revenue from DA (already locked in, but we accrue it over time for the plot)
        rev_da_t = -(price_da_t * p_da_t * DELTA_T)

        # Revenue from RT Deviation (P_RT = P_Actual - P_DA)
        p_rt_t = power_setpoint - p_da_t
        rev_rt_t = -(price_rt_t * p_rt_t * DELTA_T)

        # Total
        total_step_rev = rev_da_t + rev_rt_t

        current_total_revenue += total_step_rev
        rev_mpc_cumulative.append(current_total_revenue)

        soc_mpc.append(current_soc)
        power_mpc.append(power_setpoint)

    # --- Plotting ---
    plot_results(
        sim_steps,
        np.array(rev_mpc_cumulative),
        cum_rev_baseline,
        rt_prices_sim.iloc[:sim_steps],
        soc_mpc,
        soc_baseline,
        power_mpc,
        da_result.da_energy_bids,
    )


def run_heuristic_strategy(rt_prices, battery, steps):
    price_values = rt_prices.values
    p25, p75 = np.percentile(price_values, 25), np.percentile(price_values, 75)
    current_soc = 0.5
    soc_history = [current_soc]
    revenue_history = []

    for t in range(steps):
        price = price_values[t]
        p_cmd = 0.0
        if price <= p25:
            p_cmd = battery.power_max_mw
        elif price >= p75:
            p_cmd = -battery.power_max_mw

        # Physics
        if p_cmd > 0:
            if (
                current_soc * battery.capacity_mwh
                + p_cmd * battery.efficiency_charge * DELTA_T
                > battery.soc_max * battery.capacity_mwh
            ):
                p_cmd = 0
        elif p_cmd < 0:
            if (
                current_soc * battery.capacity_mwh
                + p_cmd / battery.efficiency_discharge * DELTA_T
                < battery.soc_min * battery.capacity_mwh
            ):
                p_cmd = 0

        if p_cmd >= 0:
            e_next = (
                current_soc * battery.capacity_mwh
                + p_cmd * battery.efficiency_charge * DELTA_T
            )
        else:
            e_next = (
                current_soc * battery.capacity_mwh
                + p_cmd / battery.efficiency_discharge * DELTA_T
            )

        current_soc = np.clip(e_next / battery.capacity_mwh, 0.0, 1.0)
        soc_history.append(current_soc)
        revenue_history.append(-(price * p_cmd * DELTA_T))

    return revenue_history, soc_history


def plot_results(
    sim_steps,
    cum_rev_mpc,
    cum_rev_baseline,
    prices,
    soc_mpc,
    soc_baseline,
    power_mpc,
    da_bids,
):
    try:
        times = np.arange(sim_steps) / 4.0
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

        ax1.set_title(f"Revenue: MPC vs Baseline (Final MPC: ${cum_rev_mpc[-1]:.2f})")
        ax1.plot(times, cum_rev_mpc, "g-", linewidth=2, label="MPC (DA+RT)")
        ax1.plot(times, cum_rev_baseline, "k--", label="Heuristic (RT Only)")
        ax1.fill_between(
            times,
            cum_rev_mpc,
            cum_rev_baseline,
            where=(cum_rev_mpc > cum_rev_baseline),
            interpolate=True,
            color="green",
            alpha=0.1,
        )
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylabel("Revenue [$]")

        ax2.set_title("Real-Time Prices")
        ax2.plot(times, prices.values, "b-", alpha=0.6)
        ax2.set_ylabel("$/MWh")
        ax2.grid(True)

        ax3.set_title("Power Dispatch")
        ax3.step(times, power_mpc, "r-", label="Realized (RT)", where="post")
        ax3.step(times, da_bids, "k--", label="DA Commitment", alpha=0.5, where="post")
        ax3.legend()
        ax3.set_ylabel("MW")
        ax3.grid(True)

        ax4.set_title("State of Charge")
        ax4.plot(times, soc_mpc[:-1], "g-", label="MPC")
        ax4.plot(times, soc_baseline[:-1], "k--", alpha=0.5, label="Heuristic")
        ax4.set_ylabel("SoC")
        ax4.set_xlabel("Time (Hours)")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        output_path = "mpc_simulation_fixed.png"
        plt.savefig(output_path, dpi=300)
        print(f"\nPlot saved to: {os.path.abspath(output_path)}")
        plt.close()

    except Exception as e:
        print(f"Plotting Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_simulation_comparison()
