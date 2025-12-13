import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.battery_model import BatteryParams
from src.forecaster import get_forecast
from src.globals import DELTA_T, TIME_STEPS_PER_HOUR, PRICE_NODE
from src.stage1_da_scheduler import solve_da_schedule
from src.stage2_rt_mpc import solve_rt_mpc
from src.utils import DAScheduleResult, load_ercot_data

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_interesting_days(data, top_n=3):
    """
    Scans the RTM data to find days with the highest price volatility.
    High volatility = High potential for MPC arbitrage.
    """
    print("Scanning data for high-volatility days...")
    
    # Filter for RTM price column
    price_col = f"{PRICE_NODE}_RTM"
    
    # Resample to daily and calculate std dev
    daily_volatility = data[price_col].resample('D').std()
    
    # Sort descending and take top N
    top_days = daily_volatility.sort_values(ascending=False).head(top_n).index
    
    # Convert to string format 'YYYY-MM-DD'
    return [d.strftime('%Y-%m-%d') for d in top_days]

def setup_simulation_data(data, sim_date_str, sim_hours=24, buffer_hours=4):
    """
    Extracts the specific slices of data needed for Stage 1 and Stage 2.
    """
    sim_date = pd.Timestamp(sim_date_str)
    
    # check availability
    if sim_date not in data.index:
        print(f"Skipping {sim_date}: Data not found.")
        return None, None, None

    # 1. Stage 1 Inputs (Day-Ahead)
    # Run at 10am D-1. We need 24h forecast for the target day.
    da_run_time = sim_date - pd.Timedelta(hours=14)
    
    da_prices_s1 = get_forecast(
        data, current_time=da_run_time, horizon_hours=14 + 24, 
        market="DA", method="perfect", verbose=False
    )[-96:] 

    rt_prices_s1 = get_forecast(
        data, current_time=da_run_time, horizon_hours=14 + 24, 
        market="RT", method="perfect", verbose=False
    )[-96:]

    # 2. Stage 2 Inputs (Real-Time Sim + Buffer)
    total_hours = sim_hours + buffer_hours
    rt_prices_sim = get_forecast(
        data, current_time=sim_date, horizon_hours=total_hours, 
        market="RT", method="perfect", verbose=False
    )
    
    # Sanity Check Length
    expected_len = int(total_hours * TIME_STEPS_PER_HOUR)
    if len(rt_prices_sim) < expected_len:
        print(f"Skipping {sim_date}: Not enough future data (End of dataset?).")
        return None, None, None
        
    return da_prices_s1, rt_prices_s1, rt_prices_sim

def run_heuristic_strategy(rt_prices, battery, steps):
    """Baseline strategy for comparison."""
    price_values = rt_prices.values
    p25, p75 = np.percentile(price_values, 25), np.percentile(price_values, 75)
    current_soc = 0.5
    soc_history = [current_soc]
    revenue_history = []

    for t in range(steps):
        price = price_values[t]
        p_cmd = 0.0
        # Simple Logic
        if price <= p25: p_cmd = battery.power_max_mw
        elif price >= p75: p_cmd = -battery.power_max_mw

        # Physics
        if p_cmd > 0:
            if current_soc * battery.capacity_mwh + p_cmd * battery.efficiency_charge * DELTA_T > battery.soc_max * battery.capacity_mwh: p_cmd = 0
        elif p_cmd < 0:
            if current_soc * battery.capacity_mwh + p_cmd / battery.efficiency_discharge * DELTA_T < battery.soc_min * battery.capacity_mwh: p_cmd = 0
        
        if p_cmd >= 0: e_next = current_soc * battery.capacity_mwh + p_cmd * battery.efficiency_charge * DELTA_T
        else: e_next = current_soc * battery.capacity_mwh + p_cmd / battery.efficiency_discharge * DELTA_T
        
        current_soc = np.clip(e_next / battery.capacity_mwh, 0.0, 1.0)
        soc_history.append(current_soc)
        revenue_history.append(-(price * p_cmd * DELTA_T))

    return revenue_history, soc_history

def run_simulation_for_date(data, date_str):
    print(f"\n>>> Running Simulation for: {date_str} <<<")
    
    # Setup
    sim_hours = 24
    mpc_horizon = 4
    sim_steps = sim_hours * TIME_STEPS_PER_HOUR
    battery = BatteryParams()

    # Get Data
    da_prices, rt_prices_s1, rt_prices_sim = setup_simulation_data(data, date_str, sim_hours, mpc_horizon)
    if da_prices is None: return

    # --- Stage 1: DA Schedule ---
    print("  Running Stage 1 (DA Scheduler)...")
    try:
        da_result = solve_da_schedule(
            da_price_forecast=da_prices,
            rt_price_forecast=rt_prices_s1,
            battery=battery,
            initial_soc=0.5, end_of_day_soc=0.5
        )
    except Exception as e:
        print(f"  Stage 1 Failed: {e}")
        return

    da_revenue_total = np.sum(-(da_prices.values * da_result.da_energy_bids * DELTA_T))
    print(f"  Stage 1 Revenue: ${da_revenue_total:.2f}")

    # --- Baseline ---
    rev_base_step, soc_base = run_heuristic_strategy(rt_prices_sim.iloc[:sim_steps], battery, sim_steps)
    cum_rev_base = np.cumsum(rev_base_step)

    # --- Stage 2: MPC ---
    print("  Running Stage 2 (MPC)...")
    curr_soc = 0.5
    soc_mpc = [curr_soc]
    rev_mpc_cum = []
    curr_total_rev = 0.0
    power_mpc = []

    for t in range(sim_steps):
        # Progress dot
        if t % 24 == 0: print(".", end="", flush=True)
        
        curr_time = pd.Timestamp(date_str) + pd.Timedelta(minutes=15 * t)
        
        res = solve_rt_mpc(
            current_time=curr_time,
            current_soc=curr_soc,
            rt_price_forecast=rt_prices_sim, # Passed full, handled inside
            da_commitments=da_result,
            battery=battery,
            horizon_type="receding",
            horizon_hours=mpc_horizon
        )
        
        p_set = res.power_setpoint if res.solve_status in ["optimal", "max_iter", "optimal_with_slack"] else 0.0
        
        # Physics
        if p_set >= 0: e_next = curr_soc * battery.capacity_mwh + p_set * battery.efficiency_charge * DELTA_T
        else: e_next = curr_soc * battery.capacity_mwh + p_set / battery.efficiency_discharge * DELTA_T
        curr_soc = np.clip(e_next / battery.capacity_mwh, 0.0, 1.0)
        
        # Revenue (Two-Settlement)
        rev_da_t = -(da_prices.values[t] * da_result.da_energy_bids[t] * DELTA_T)
        rev_rt_t = -(rt_prices_sim.iloc[t] * (p_set - da_result.da_energy_bids[t]) * DELTA_T)
        
        curr_total_rev += (rev_da_t + rev_rt_t)
        rev_mpc_cum.append(curr_total_rev)
        soc_mpc.append(curr_soc)
        power_mpc.append(p_set)

    print(f"\n  Done. Final Revenue: ${curr_total_rev:.2f} (Baseline: ${cum_rev_base[-1]:.2f})")

    # --- Plotting ---
    plot_simulation(
        date_str, sim_steps, rev_mpc_cum, cum_rev_base, 
        rt_prices_sim.iloc[:sim_steps], soc_mpc, soc_base, power_mpc, da_result.da_energy_bids
    )

def plot_simulation(date_str, steps, rev_mpc, rev_base, prices, soc_mpc, soc_base, power, da_bids):
    times = np.arange(steps) / 4.0
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    ax1.set_title(f"Simulation: {date_str} | MPC: ${rev_mpc[-1]:.0f} vs Base: ${rev_base[-1]:.0f}")
    ax1.plot(times, rev_mpc, "g-", linewidth=2, label="MPC")
    ax1.plot(times, rev_base, "k--", label="Heuristic")
    ax1.legend(); ax1.grid(True); ax1.set_ylabel("Revenue [$]")
    
    ax2.plot(times, prices.values, "b-", alpha=0.6); ax2.set_ylabel("Price RT"); ax2.grid(True)
    
    ax3.step(times, power, "r-", where='post', label="RT Dispatch")
    ax3.step(times, da_bids, "k--", where='post', label="DA Plan")
    ax3.legend(); ax3.set_ylabel("MW"); ax3.grid(True)
    
    ax4.plot(times, soc_mpc[:-1], "g-"); ax4.plot(times, soc_base[:-1], "k--")
    ax4.set_ylabel("SoC"); ax4.grid(True)
    
    # Create 'plots' folder if not exists
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/sim_{date_str}.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Plot saved to {filename}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # 1. Load Data ONCE
    print("Loading Dataset (this may take a moment)...")
    full_data = load_ercot_data()
    
    # 2. Define Dates to Test
    # Option A: Manual Dates
    # test_dates = ["2025-01-15", "2025-06-15"] 
    
    # Option B: Automatic "High Volatility" Detection
    test_dates = find_interesting_days(full_data, top_n=3)
    
    # 3. Run Loop
    print(f"\nRunning simulations for: {test_dates}")
    for date in test_dates:
        run_simulation_for_date(full_data, date)