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

def find_interesting_days(data, top_n=3):
    print("Scanning data for high-volatility days...")
    price_col = f"{PRICE_NODE}_RTM"
    daily_volatility = data[price_col].resample('D').std()
    top_days = daily_volatility.sort_values(ascending=False).head(top_n).index
    return [d.strftime('%Y-%m-%d') for d in top_days]

def setup_simulation_data(data, sim_date_str, sim_hours=24, buffer_hours=4):
    sim_date = pd.Timestamp(sim_date_str)
    if sim_date not in data.index:
        print(f"Skipping {sim_date}: Data not found.")
        return None, None, None

    da_run_time = sim_date - pd.Timedelta(hours=14)
    
    # Stage 1: Persistence Forecast
    da_prices_s1 = get_forecast(
        data, current_time=da_run_time, horizon_hours=14 + 24, 
        market="DA", method="persistence", verbose=False
    )[-96:] 

    rt_prices_s1 = get_forecast(
        data, current_time=da_run_time, horizon_hours=14 + 24, 
        market="RT", method="persistence", verbose=False
    )[-96:]

    # Stage 2: Perfect Forecast (Simulation)
    total_hours = sim_hours + buffer_hours
    rt_prices_sim = get_forecast(
        data, current_time=sim_date, horizon_hours=total_hours, 
        market="RT", method="perfect", verbose=False
    )
    
    if len(rt_prices_sim) < int(total_hours * TIME_STEPS_PER_HOUR):
        return None, None, None
        
    return da_prices_s1, rt_prices_s1, rt_prices_sim

def run_simulation_for_date(data, date_str):
    print(f"\n>>> Running Simulation for: {date_str} <<<")
    sim_hours = 24
    mpc_horizon = 4
    sim_steps = sim_hours * TIME_STEPS_PER_HOUR
    battery = BatteryParams()

    da_prices, rt_prices_s1, rt_prices_sim = setup_simulation_data(data, date_str, sim_hours, mpc_horizon)
    if da_prices is None: return

    # --- Stage 1 ---
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

    # --- Stage 2 ---
    print("  Running Stage 2 (MPC)...")
    curr_soc = 0.5
    soc_mpc = [curr_soc]
    rev_cum = []
    total_rev = 0.0
    
    # Tracking Arrays for Plotting
    p_green_list = [] # Actual Dispatch
    p_orange_list = [] # RT Deviation
    p_blue_list = []   # DA Commitment

    for t in range(sim_steps):
        if t % 24 == 0: print(".", end="", flush=True)
        
        curr_time = pd.Timestamp(date_str) + pd.Timedelta(minutes=15 * t)
        
        res = solve_rt_mpc(
            current_time=curr_time,
            current_soc=curr_soc,
            rt_price_forecast=rt_prices_sim,
            da_commitments=da_result,
            battery=battery,
            horizon_type="receding",
            horizon_hours=mpc_horizon
        )
        
        # 1. Get Requested Green Line (Dispatch)
        p_req = res.power_setpoint if res.solve_status in ["optimal", "max_iter", "optimal_with_slack"] else 0.0
        
        # 2. Strict Physics (Clamping)
        e_curr = curr_soc * battery.capacity_mwh
        e_max = battery.soc_max * battery.capacity_mwh
        e_min = battery.soc_min * battery.capacity_mwh
        
        max_ch = (e_max - e_curr) / (battery.efficiency_charge * DELTA_T)
        max_dis = (e_min - e_curr) * battery.efficiency_discharge / DELTA_T
        
        # Green Line (Actual)
        p_green = np.clip(p_req, -battery.power_max_mw, battery.power_max_mw)
        p_green = np.clip(p_green, max_dis, max_ch)
        
        # 3. Derive Components
        p_blue = da_result.da_energy_bids[t] # DA Commitment
        p_orange = p_green - p_blue          # RT Adjustment
        
        # Update State
        if p_green >= 0: e_next = e_curr + p_green * battery.efficiency_charge * DELTA_T
        else: e_next = e_curr + p_green / battery.efficiency_discharge * DELTA_T
        curr_soc = np.clip(e_next / battery.capacity_mwh, 0.0, 1.0)
        
        # Revenue
        rev_da = -(da_prices.values[t] * p_blue * DELTA_T)
        rev_rt = -(rt_prices_sim.iloc[t] * p_orange * DELTA_T)
        
        total_rev += (rev_da + rev_rt)
        
        # Store
        rev_cum.append(total_rev)
        soc_mpc.append(curr_soc)
        p_green_list.append(p_green)
        p_blue_list.append(p_blue)
        p_orange_list.append(p_orange)

    print(f"\n  Done. Final Revenue: ${total_rev:.2f}")

    plot_detailed_simulation(
        date_str, sim_steps, rev_cum, rt_prices_sim.iloc[:sim_steps], 
        soc_mpc, p_green_list, p_blue_list, p_orange_list
    )

def plot_detailed_simulation(date_str, steps, revenue, prices, soc, p_green, p_blue, p_orange):
    times = np.arange(steps) / 4.0
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # 1. Revenue
    ax1.set_title(f"Simulation: {date_str} | Total Revenue: ${revenue[-1]:,.2f}")
    ax1.plot(times, revenue, "g-", linewidth=2, label="Cumulative Revenue")
    ax1.legend(); ax1.grid(True); ax1.set_ylabel("Revenue [$]")
    
    # 2. Prices
    ax2.plot(times, prices.values, "b-", alpha=0.6, label="RT Price")
    ax2.legend(); ax2.set_ylabel("Price [$]"); ax2.grid(True)
    
    # 3. Power Components (THE KEY PLOT)
    ax3.set_title("Power Components: Green = Blue + Orange")
    ax3.step(times, p_blue, "b--", where='post', label="DA Plan (Blue)", alpha=0.6)
    ax3.step(times, p_orange, "orange", where='post', label="RT Adjustment (Orange)", alpha=0.8)
    ax3.step(times, p_green, "g-", where='post', label="Final Dispatch (Green)", linewidth=1.5)
    ax3.legend(loc="upper right"); ax3.set_ylabel("Power [MW]"); ax3.grid(True)
    
    # 4. SoC
    ax4.plot(times, soc[:-1], "k-", label="SoC")
    ax4.set_ylabel("SoC"); ax4.grid(True)
    
    os.makedirs("plots", exist_ok=True)
    filename = f"plots/sim_{date_str}_detailed.png"
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"  Plot saved to {filename}")

if __name__ == "__main__":
    full_data = load_ercot_data()
    test_dates = find_interesting_days(full_data, top_n=3)
    for date in test_dates:
        run_simulation_for_date(full_data, date)