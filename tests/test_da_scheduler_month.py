import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.battery_model import BatteryParams
from src.forecaster import get_forecasts_for_da
from src.stage1_da_scheduler import solve_da_schedule
from src.utils.load_ercot_data import load_ercot_data

# --- Configuration ---
AMT_DAYS = 2
TOTAL_DAYS = 300
START_DATE = "2024-06-01 10:00:00"


def test_monthly_da_scheduler_comparison():
    """
    Runs the DA scheduler over a 28-day period in 2-day increments
    and compares cumulative revenue across different strategies.
    """

    # 1. Load Data
    data = load_ercot_data()

    print(f"Data Loaded. Range: {data.index.min()} to {data.index.max()}")

    scenarios_config = {
        "Baseline xgboost (w=0, p=0, Unc=0)": {
            "cvar_weight": 0,
            "rt_uncertainty_default": 0,
            "rt_dispatch_penalty": 0,
            "color": "#2259f1",
            "linestyle": "-",
            "type": "xgboost",
        },
        "Risk-Averse xgboost (w=0.5, Unc=20)": {
            "cvar_weight": 0.5,
            "rt_uncertainty_default": 20,
            "rt_dispatch_penalty": 0,
            "color": "#89AAC9",
            "linestyle": ":",
            "type": "xgboost",
        },
        # "Conservative xgboost (w=0.1, Unc=20)": {
        #     "cvar_weight": 0.1,
        #     "rt_uncertainty_default": 20,
        #     "rt_dispatch_penalty": 0,
        #     "color": "#0a2980",
        #     "linestyle": ":",
        #     "type": "xgboost",
        # },
        "Risk-Averse Persistance (w=0.1, Unc=20)": {
            "cvar_weight": 0.1,
            "rt_uncertainty_default": 20,
            "rt_dispatch_penalty": 0,
            "color": "#eddd00",
            "linestyle": ":",
            "type": "persistence",
        },
        "Perfect Uncertainty Persistance (w=0.1, Unc=20)": {
            "cvar_weight": 0.1,
            "rt_uncertainty_default": 20,
            "use_perfect_uncertainty": True,
            "rt_dispatch_penalty": 0,
            "color": "#fb7d00",
            "linestyle": ":",
            "type": "persistence",
        },
        "Perfect Uncertainty xgboost (w=0.1)": {
            "cvar_weight": 0.1,
            "rt_uncertainty_default": 0,
            "rt_dispatch_penalty": 0,
            "use_perfect_uncertainty": True,
            "color": "#47c6e6",
            "linestyle": ":",
            "type": "xgboost",
        },
        "Baseline Persistence (w=0, p=0, Unc=0)": {
            "cvar_weight": 0,
            "rt_uncertainty_default": 0,
            "rt_dispatch_penalty": 0,
            "color": "tab:red",
            "linestyle": "-",
            "type": "persistence",
        },
        "Baseline Regression (w=0, p=0, Unc=0)": {
            "cvar_weight": 0,
            "rt_uncertainty_default": 0,
            "rt_dispatch_penalty": 0,
            "color": "tab:green",
            "linestyle": "-",
            "type": "regression",
        },
        "Perfect Uncertainty regression (w=0.1)": {
            "cvar_weight": 0.1,
            "rt_uncertainty_default": 0,
            "rt_dispatch_penalty": 0,
            "use_perfect_uncertainty": True,
            "color": "#62e647",
            "linestyle": ":",
            "type": "xgboost",
        },
        "Baseline Persistance, just DA market (w=0.1)": {
            "cvar_weight": 0,
            "rt_uncertainty_default": 0,
            "rt_dispatch_penalty": 1e5,
            "use_perfect_uncertainty": True,
            "color": "#e64797",
            "linestyle": ":",
            "type": "xgboost",
        },
        # "Perfect Forecast": {
        #     "cvar_weight": 0,
        #     "rt_uncertainty_default": 0,
        #     "rt_dispatch_penalty": 0,
        #     "color": "tab:brown",
        #     "linestyle": "-",
        #     "type": "perfect",
        # },
    }

    # Initialize storage
    scenario_results_store = {}
    for name in scenarios_config:
        scenario_results_store[name] = []

    # --- MAIN TIME LOOP ---
    start_timestamp = pd.Timestamp(START_DATE)
    num_iterations = TOTAL_DAYS // AMT_DAYS

    battery = BatteryParams()

    for i in tqdm(range(num_iterations), desc="Processing Time Blocks"):
        current_time = start_timestamp + pd.Timedelta(days=i * AMT_DAYS)

        # 1. Generate Forecasts
        da_fc_xgb, rt_fc_xgb = get_forecasts_for_da(
            data,
            current_time=current_time,
            horizon_hours=24 * AMT_DAYS,
            method="xgboost",
            verbose=False,
        )
        da_fc_reg, rt_fc_reg = get_forecasts_for_da(
            data,
            current_time=current_time,
            horizon_hours=24 * AMT_DAYS,
            method="regression",
            verbose=False,
        )
        da_fc_pers, rt_fc_pers = get_forecasts_for_da(
            data,
            current_time=current_time,
            horizon_hours=24 * AMT_DAYS,
            method="persistence",
            verbose=False,
        )
        da_real, rt_real = get_forecasts_for_da(
            data,
            current_time=current_time,
            horizon_hours=24 * AMT_DAYS,
            method="perfect",
            verbose=False,
        )

        # 3. Run Scenarios
        for name, params in scenarios_config.items():
            if params["type"] == "xgboost":
                da_input, rt_input = da_fc_xgb, rt_fc_xgb
            elif params["type"] == "persistence":
                da_input, rt_input = da_fc_pers, rt_fc_pers
            elif params["type"] == "perfect":
                da_input, rt_input = da_real, rt_real
            elif params["type"] == "regression":
                da_input, rt_input = da_fc_reg, rt_fc_reg
            else:
                raise ValueError(f"Unknown forecast type in scenario {name}")

            unc_input = None
            if params.get("use_perfect_uncertainty"):
                unc_input = (rt_real - rt_input).abs()

            for i in range(2):
                try:
                    result = solve_da_schedule(
                        da_price_forecast=da_input,
                        rt_price_forecast=rt_input,
                        battery=battery,
                        cvar_weight=params["cvar_weight"],
                        rt_uncertainty_default=params["rt_uncertainty_default"],
                        rt_dispatch_penalty=params["rt_dispatch_penalty"],
                        n_scenarios=20,
                        rt_price_uncertainty=unc_input,
                    )
                except Exception as e:
                    print(f"Error in scenario {name}: {e}")
                    continue

            # Revenue Calc
            revenue_chunk = (
                result.da_energy_bids * da_real + result.rt_energy_bids * rt_real
            )
            scenario_results_store[name].append(revenue_chunk)

    # --- AGGREGATION ---
    results_comparison = {}

    for name, list_of_series in scenario_results_store.items():
        full_revenue_series = pd.concat(list_of_series)
        cumulative_revenue = -full_revenue_series.cumsum()

        results_comparison[name] = {
            "cumulative_revenue_real": cumulative_revenue,
            "total_revenue": cumulative_revenue.iloc[-1],
            "color": scenarios_config[name]["color"],
            "linestyle": scenarios_config[name]["linestyle"],
        }

    # --- ASSERTIONS (Validation) ---
    assert len(results_comparison) == len(scenarios_config), (
        "Not all scenarios generated results"
    )
    for name, res in results_comparison.items():
        assert not res["cumulative_revenue_real"].empty, (
            f"{name} produced empty results"
        )
        assert not np.isnan(res["total_revenue"]), f"{name} total revenue is NaN"

    # --- PLOTTING ---
    # Create the 'tests' directory if it doesn't exist so plotting doesn't fail
    os.makedirs("tests", exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sorted_names = sorted(
        results_comparison.keys(),
        key=lambda k: results_comparison[k]["total_revenue"],
        reverse=True,
    )

    for name in sorted_names:
        res = results_comparison[name]
        ax.plot(
            res["cumulative_revenue_real"].index,
            res["cumulative_revenue_real"],
            label=f"{name} | Total: ${res['total_revenue']:,.2f}",
            color=res["color"],
            linestyle=res["linestyle"],
            linewidth=1.5,
            alpha=0.9,
        )

    ax.set_title(
        f"Cumulative Real Revenue over {TOTAL_DAYS} Days\n(2-Day Rolling Optimization)"
    )
    ax.set_ylabel("Cumulative Revenue ($ / MWh)")
    ax.set_xlabel("Time")
    ax.legend(title="Optimization Strategy", loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_filename = "tests/monthly_da_scheduler_comparison.png"
    plt.savefig(output_filename, dpi=150)

    # We generally don't use plt.show() in automated tests, but we print the path
    print(f"\nPlot saved to {output_filename}")


if __name__ == "__main__":
    test_monthly_da_scheduler_comparison()
