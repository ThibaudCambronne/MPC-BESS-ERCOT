import pandas as pd
import os
from src.utils.load_ercot_data import load_ercot_data
from src.battery_model import BatteryParams
from src.simulator import run_simulation, plot_multi_day_simulation, plot_day_simulation
from src.globals import PRICE_NODE

def main():
    print("Loading ERCOT data...")
    data = load_ercot_data()
    print(f"Data loaded: {len(data)} intervals from {data.index[0]} to {data.index[-1]}")

    # Initialize battery
    battery = BatteryParams()

    # Run a short simulation (3 days as a test)
    # Note: Start from 2020-01-03 to ensure persistence forecast has previous day's data
    # start_date = pd.Timestamp("2020-01-03")
    start_date = pd.Timestamp("2024-06-01 10:00:00")
    n_days = 300

    print(f"\nRunning {n_days}-day simulation starting {start_date.date()}...")
    results = run_simulation(
        data=data,
        start_date=start_date,
        n_days=n_days,
        battery=battery,
        forecast_method="persistence",
        horizon_type="receding",
        initial_soc=0.5,
        end_of_day_soc=0.5
    )

    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Total Revenue: ${results.total_revenue:,.2f}")
    print(f"\nDaily Breakdown:")
    for i, day_result in enumerate(results.daily_results):
        print(f"  Day {i+1} ({day_result.date.date()}): ${day_result.total_revenue:,.2f}")
        print(f"    DA Revenue: ${day_result.da_revenue:,.2f}")
        print(f"    RT Revenue: ${day_result.rt_revenue:,.2f}")
        print(f"    Final SOC: {day_result.final_soc:.2%}")
    
    daily_data_list = []
    for day_result in results.daily_results:
        daily_data_list.append({
            "Date": day_result.date.date(),
            "Total Revenue": day_result.total_revenue,
            "DA Revenue": day_result.da_revenue,
            "RT Revenue": day_result.rt_revenue,
            "Final SOC": day_result.final_soc
        })

    # 2. Create DataFrame and Save to CSV
    df_daily = pd.DataFrame(daily_data_list)
    csv_filename = f"simulation_results_{start_date.strftime('%Y%m%d')}.csv"
    df_daily.to_csv(csv_filename, index=False)
    # 1. Initialize a list to hold all interval rows
    all_intervals = []

    for day_result in results.daily_results:
        # Get the date and slice the original data for this specific day
        # Assuming 'data' is indexed by datetime and covers the simulation range
        day_start = day_result.date
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
        
        # Get market prices for these 96 intervals
        day_market_data = data.loc[day_start:day_end].copy()
        
        # Check if trajectory length matches market data (usually 96 intervals)
        # If soc_trajectory has 97 points (including start of next day), use [:96]
        soc_values = day_result.soc_trajectory[:len(day_market_data)]
        
        day_market_data['SOC'] = soc_values
        
        # Calculate Power (MW) based on Change in SOC (MWh)
        # Power = (Delta SOC) / (Time Step in hours)
        # 15 mins = 0.25 hours
        day_market_data['Net_Power_MW'] = day_market_data['SOC'].diff().fillna(0) / 0.25
        
        all_intervals.append(day_market_data)

    # 2. Concatenate all days into one master DataFrame
    df_intervals_full = pd.concat(all_intervals)

    # 3. Save to CSV
    interval_csv_path = f"interval_analysis_{start_date.strftime('%Y%m%d')}.csv"
    df_intervals_full.to_csv(interval_csv_path)

    print(f"✅ Full 15-minute data saved to: {interval_csv_path}")

   # Create plots folder if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Multi-day overview plot
    end_date = start_date + pd.Timedelta(days=n_days-1)
    multi_day_plot_path = os.path.join(plots_dir, f"multi_day_simulation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png")
    plot_multi_day_simulation(results, data=data, battery=battery, save_path=multi_day_plot_path)

    # # Individual day plots
    # for i, day_result in enumerate(results.daily_results):
    #     # Get actual prices for this day
    #     day_start = day_result.date.normalize()
    #     day_end = day_start + pd.Timedelta(days=1)
    #     day_data = data.loc[day_start:day_end - pd.Timedelta(minutes=15)]

    #     if len(day_data) == 96:
    #         actual_da_prices = day_data[f"{PRICE_NODE}_DAM"].values
    #         actual_rt_prices = day_data[f"{PRICE_NODE}_RTM"].values

    #         # Get DA schedule for this day
    #         da_schedule = results.da_schedules.get(day_result.date)

    #         day_plot_path = os.path.join(plots_dir, f"day_simulation_{day_result.date.strftime('%Y%m%d')}.png")
    #         plot_day_simulation(
    #             day_result,
    #             actual_da_prices,
    #             actual_rt_prices,
    #             da_schedule=da_schedule,
    #             save_path=day_plot_path
    #         )
    
    # print(f"\nAll plots saved in: {os.path.abspath(plots_dir)}")


if __name__ == "__main__":
    main()
