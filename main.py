import pandas as pd
from src.utils import load_ercot_data
from src.battery_model import BatteryParams
from src.simulator import run_simulation

def main():
    print("Loading ERCOT data...")
    data = load_ercot_data()
    print(f"Data loaded: {len(data)} intervals from {data.index[0]} to {data.index[-1]}")

    # Initialize battery
    battery = BatteryParams()

    # Run a short simulation (3 days as a test)
    start_date = pd.Timestamp("2020-01-02")
    end_date = pd.Timestamp("2020-01-04")

    print(f"\nRunning simulation from {start_date.date()} to {end_date.date()}...")
    results = run_simulation(
        data=data,
        start_date=start_date,
        end_date=end_date,
        battery=battery,
        forecast_method="perfect",
        horizon_type="receding"
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


if __name__ == "__main__":
    main()
