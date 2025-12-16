import os

import pandas as pd

from src.battery_model import BatteryParams
from src.globals import PRICE_NODE
from src.simulator import plot_day_simulation, plot_multi_day_simulation, run_simulation
from src.utils import load_ercot_data


def main():
    print("Loading ERCOT data...")
    data = load_ercot_data()
    print(
        f"Data loaded: {len(data)} intervals from {data.index[0]} to {data.index[-1]}"
    )

    # Initialize battery
    battery = BatteryParams()

    # Run a short simulation (3 days as a test)
    # Note: Start from 2020-01-03 to ensure persistence forecast has previous day's data
    # start_date = pd.Timestamp("2020-01-03")
    start_date = pd.Timestamp("2025-02-15")
    n_days = 15

    print(f"\nRunning {n_days}-day simulation starting {start_date.date()}...")
    results = run_simulation(
        data=data,
        start_date=start_date,
        n_days=n_days,
        battery=battery,
        forecast_method="perfect",
        horizon_type="receding",
        initial_soc=0.5,
        end_of_day_soc=0.5,
    )

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    print(f"Total Revenue: ${results.total_revenue:,.2f}")
    print("\nDaily Breakdown:")
    for i, day_result in enumerate(results.daily_results):
        print(
            f"  Day {i + 1} ({day_result.date.date()}): ${day_result.total_revenue:,.2f}"
        )
        print(f"    DA Revenue: ${day_result.da_revenue:,.2f}")
        print(f"    RT Revenue: ${day_result.rt_revenue:,.2f}")
        print(f"    Final SOC: {day_result.final_soc:.2%}")

    # Create plots folder if it doesn't exist
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    # Multi-day overview plot
    end_date = start_date + pd.Timedelta(days=n_days - 1)
    multi_day_plot_path = os.path.join(
        plots_dir,
        f"multi_day_simulation_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.png",
    )
    plot_multi_day_simulation(
        results, data=data, battery=battery, save_path=multi_day_plot_path
    )

    # Individual day plots
    for i, day_result in enumerate(results.daily_results):
        # Get actual prices for this day
        day_start = day_result.date.normalize()
        day_end = day_start + pd.Timedelta(days=1)
        day_data = data.loc[day_start : day_end - pd.Timedelta(minutes=15)]

        if len(day_data) == 96:
            actual_da_prices = day_data[f"{PRICE_NODE}_DAM"].values
            actual_rt_prices = day_data[f"{PRICE_NODE}_RTM"].values

            # Get DA schedule for this day
            da_schedule = results.da_schedules.get(day_result.date)

            day_plot_path = os.path.join(
                plots_dir, f"day_simulation_{day_result.date.strftime('%Y%m%d')}.png"
            )
            plot_day_simulation(
                day_result,
                actual_da_prices,
                actual_rt_prices,
                da_schedule=da_schedule,
                save_path=day_plot_path,
            )

    print(f"\nAll plots saved in: {os.path.abspath(plots_dir)}")


if __name__ == "__main__":
    main()
