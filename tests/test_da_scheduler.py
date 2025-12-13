import matplotlib.pyplot as plt
import pandas as pd

from src.battery_model import BatteryParams
from src.forecaster import get_forecasts_for_da
from src.stage1_da_scheduler import solve_da_schedule
from src.utils import load_ercot_data


AMT_DAYS = 5
def test_da_scheduler():
    data = load_ercot_data()
    current_time = pd.Timestamp("2025-02-01 10:00:00")
    
    # 1. Prices for the scheduler (Persistence Forecast)
    da_prices_forecast, rt_prices_forecast = get_forecasts_for_da(
        data,
        current_time=current_time,
        horizon_hours=24 * AMT_DAYS,
        method="persistence",
        verbose=False,
    )

    # 2. Prices for Real Revenue Calculation (Perfect Forecast / Real Prices)
    da_prices_real, rt_prices_real = get_forecasts_for_da(
        data,
        current_time=current_time,
        horizon_hours=24 * AMT_DAYS,
        method="perfect",
        verbose=False,
    )
    
    battery = BatteryParams()
    
    # --- Define Scenarios for Comparison ---
    scenarios = {
        "Baseline (w=0, p=0)": {
            "cvar_weight": 0,
            "rt_dispatch_penalty": 0,
            "forecast_input": (da_prices_forecast, rt_prices_forecast),
            "color": "tab:blue",
            "linestyle": "-"
        },
        "Risk-Averse": {
            "cvar_weight": 0.5,
            "rt_dispatch_penalty": 50,
            "forecast_input": (da_prices_forecast, rt_prices_forecast),
            "color": "tab:orange",
            "linestyle": "--"
        },
        "Conservative": {
            "cvar_weight": 0.1,
            "rt_dispatch_penalty": 20,
            "forecast_input": (da_prices_forecast, rt_prices_forecast),
            "color": "tab:green",
            "linestyle": ":"
        },
        "Perfect Prediction (w=0, p=0)": { 
            "cvar_weight": 0,
            "rt_dispatch_penalty": 0,
            "forecast_input": (da_prices_real, rt_prices_real), # Use Real Prices as input
            "color": "black",
            "linestyle": "-."
        },
    }
    
    results_comparison = {}
    
    print("Solving DA Schedule for different scenarios...")
    
    for name, params in scenarios.items():
        print(f"  -> Running scenario: {name}")
        
        # Determine the price input for the solver
        da_input, rt_input = params["forecast_input"]

        # Solve DA schedule with specific parameters
        result = solve_da_schedule(
            da_price_forecast=da_input,
            rt_price_forecast=rt_input,
            battery=battery,
            cvar_weight=params["cvar_weight"],
            rt_dispatch_penalty=params["rt_dispatch_penalty"],
        )

        # The 'real' revenue calculation MUST always use the REAL market prices, 
        # regardless of what prices were used to generate the schedule.
        revenue_real = result.da_energy_bids * da_prices_real + result.rt_energy_bids * rt_prices_real
        
        # Calculate cumulative revenue
        cumulative_revenue_real = revenue_real.cumsum()
        
        # Store the result
        results_comparison[name] = {
            "cumulative_revenue_real": cumulative_revenue_real,
            "total_revenue": cumulative_revenue_real.iloc[-1],
            "color": params["color"],
            "linestyle": params["linestyle"]
        }

    # --- Plotting Cumulative Real Revenue Comparison ---
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sort the results so the 'Perfect Prediction' is plotted last (on top)
    sorted_names = sorted(results_comparison.keys(), key=lambda k: 1 if k == "Perfect Prediction (w=0, p=0)" else 0)

    for name in sorted_names:
        data = results_comparison[name]
        ax.plot(
            data["cumulative_revenue_real"].index, 
            data["cumulative_revenue_real"], 
            label=f"{name} (Total: ${data['total_revenue']:.2f})",
            color=data["color"],
            linestyle=data["linestyle"],
            linewidth=2 if "Perfect" in name else 1.5,
            alpha=1 if "Perfect" in name else 0.8
        )

    ax.set_title("Cumulative Real Revenue: Optimization Scenarios vs. Perfect Prediction")
    ax.set_ylabel("Cumulative Revenue (\$ / MWh)")
    ax.set_xlabel("Time")
    ax.legend(title="Optimization Strategy")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("tests/da_scheduler_comparison_revenue_with_perfect.png", dpi=150)
    plt.show() 
    plt.close()
    
    print("\nComparison of Total Real Revenue:")
    for name, data in results_comparison.items():
        print(f"  - {name}: ${data['total_revenue']:.2f}")


if __name__ == "__main__":
    print("Running DA Scheduler Parameter Comparison with Perfect Prediction...\n")
    test_da_scheduler()
    print()
    print("\n ANALYSIS COMPLETE! ")