import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

from app.utils_app.cumulative_plot import add_schedule_comparison_traces
from app.utils_app.simulation_math import (
    compute_daily_tb_strategy_revenues,
    compute_revenue_series,
)
from src.colors import DA_STAGE_RT_FORECAST_COLOR, RT_COLOR
from src.forecasts.build_forecast_vs_actual_plotly_figure import (
    build_forecast_vs_actual_plotly_figure,
)
from src.globals import FREQUENCY, PRICE_NODE, TIME_STEPS_PER_HOUR
from src.multi_day_simulation import multi_day_simulation
from src.utils.battery_model import BatteryParams
from src.utils.data_classes import MultiDaySimulationResult
from src.utils.load_ercot_data import load_ercot_data

RT_ALGO_NO_CONTROL = "No RT Control"
RT_ALGO_MPC = "MPC"


def parse_month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not re.match(r"^\d{4}-\d{2}$", month):
        raise ValueError(f"Invalid month '{month}'. Expected format YYYY-MM.")
    try:
        month_period = pd.Period(month, freq="M")
    except Exception as exc:
        raise ValueError(f"Invalid month '{month}'. Expected format YYYY-MM.") from exc
    return month_period.start_time.normalize(), month_period.end_time.normalize()


def build_output_path(output_dir: str, month: str, filename: str | None = None) -> Path:
    output_dir_path = Path(output_dir)
    output_name = filename or f"monthly_simulation_{month}.png"
    return output_dir_path / output_name


def build_monthly_figure(
    data: pd.DataFrame,
    month_start: pd.Timestamp,
    month_result: MultiDaySimulationResult,
    da_forecast_method: str,
    rt_forecast_method: str,
    battery: BatteryParams,
    run_parameters_text: str,
):
    (
        da_stage_planned_revenue,
        da_stage_actual_revenue,
        rt_stage_planned_revenue,
        rt_stage_actual_revenue,
        perfect_revenue,
    ) = compute_revenue_series(
        da_forecast=month_result.da_forecast_used,
        da_stage_rt_forecast_used=month_result.da_stage_rt_forecast_used,
        rt_forecast=month_result.rt_forecast_used,
        da_bids=month_result.da_energy_bids,
        da_stage_rt_energy_bids=month_result.da_stage_rt_energy_bids,
        rt_bids=month_result.rt_energy_bids,
        da_forecast_perfect=month_result.da_forecast_perfect,
        rt_forecast_perfect=month_result.rt_forecast_perfect,
        da_bids_perfect=month_result.da_energy_bids_perfect,
        rt_bids_perfect=month_result.rt_energy_bids_perfect,
    )

    da_stage_planned_cumulative = np.cumsum(da_stage_planned_revenue)
    da_stage_actual_cumulative = np.cumsum(da_stage_actual_revenue)
    rt_stage_planned_cumulative = np.cumsum(rt_stage_planned_revenue)
    rt_stage_actual_cumulative = np.cumsum(rt_stage_actual_revenue)
    perfect_cumulative = np.cumsum(perfect_revenue)

    tb_day_end_timestamps: list[pd.Timestamp] = []
    tb2_da_daily_revenue: list[float] = []
    tb2_rt_daily_revenue: list[float] = []
    tb4_da_daily_revenue: list[float] = []
    tb4_rt_daily_revenue: list[float] = []

    for operating_day_start in month_result.operating_days:
        day_index = pd.date_range(
            start=operating_day_start,
            periods=24 * TIME_STEPS_PER_HOUR,
            freq=FREQUENCY,
        )
        day_tb_revenues = compute_daily_tb_strategy_revenues(
            da_prices=month_result.da_forecast_perfect.reindex(day_index),
            rt_prices=month_result.rt_forecast_perfect.reindex(day_index),
            battery=battery,
        )
        tb_day_end_timestamps.append(day_index[-1])
        tb2_da_daily_revenue.append(day_tb_revenues["tb2_da_revenue"])
        tb2_rt_daily_revenue.append(day_tb_revenues["tb2_rt_revenue"])
        tb4_da_daily_revenue.append(day_tb_revenues["tb4_da_revenue"])
        tb4_rt_daily_revenue.append(day_tb_revenues["tb4_rt_revenue"])

    tb2_da_cumulative = np.cumsum(tb2_da_daily_revenue)
    tb2_rt_cumulative = np.cumsum(tb2_rt_daily_revenue)
    tb4_da_cumulative = np.cumsum(tb4_da_daily_revenue)
    tb4_rt_cumulative = np.cumsum(tb4_rt_daily_revenue)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.06,
        subplot_titles=(
            "Cumulative Revenue Comparison",
            "DA/RT Forecast vs Real Prices",
        ),
    )

    add_schedule_comparison_traces(
        fig=fig,
        x_values=month_result.index,
        da_only_planned=da_stage_planned_cumulative,
        da_only_actual=da_stage_actual_cumulative,
        final_planned=rt_stage_planned_cumulative,
        final_actual=rt_stage_actual_cumulative,
        perfect_cumulative=perfect_cumulative,
        row=1,
        col=1,
        legendgroup="subplot1",
        tb_marker_series=[
            {
                "x": tb_day_end_timestamps,
                "y": tb2_da_cumulative,
                "name": "TB2 DA Revenue",
            },
            {
                "x": tb_day_end_timestamps,
                "y": tb2_rt_cumulative,
                "name": "TB2 RT Revenue",
            },
            {
                "x": tb_day_end_timestamps,
                "y": tb4_da_cumulative,
                "name": "TB4 DA Revenue",
            },
            {
                "x": tb_day_end_timestamps,
                "y": tb4_rt_cumulative,
                "name": "TB4 RT Revenue",
            },
        ],
    )

    month_data = data.reindex(month_result.index)
    da_price_col = f"{PRICE_NODE}_DAM"
    rt_price_col = f"{PRICE_NODE}_RTM"

    price_fig = build_forecast_vs_actual_plotly_figure(
        current_time=month_start,
        data=month_data,
        forecasts={
            f"DA Stage {da_forecast_method}": month_result.da_forecast_used.resample(
                "h"
            ).mean(),
        },
        market="DA/RT",
        price_col=da_price_col,
        rt_forecasts={
            f"DA Stage {da_forecast_method}": month_result.da_stage_rt_forecast_used.resample(
                "h"
            ).mean(),
            f"RT Stage {rt_forecast_method}": month_result.rt_forecast_used.resample(
                "h"
            ).mean(),
        },
        rt_forecast_colors={
            f"DA Stage {da_forecast_method}": DA_STAGE_RT_FORECAST_COLOR,
            f"RT Stage {rt_forecast_method}": RT_COLOR,
        },
        rt_price_col=rt_price_col,
        highlight_market_order_mismatch=True,
        historical_days=0,
        visible_history_hours=0,
    )

    for trace in price_fig.data:
        trace.legendgroup = "subplot2"  # type: ignore
        fig.add_trace(trace, row=2, col=1)

    for shape in price_fig.layout.shapes or []:
        fig.add_shape(shape, row=2, col=1)

    if price_fig.layout.xaxis.range:
        fig.update_xaxes(range=list(price_fig.layout.xaxis.range), row=2, col=1)

    fig.update_layout(
        height=920,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            xanchor="left",
            yanchor="top",
        ),
        legend_tracegroupgap=150,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.1,
        y=0.1,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        align="left",
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="rgba(0, 0, 0, 0.15)",
        borderwidth=1,
        font=dict(size=11),
        text=f"Run Parameters:<br>{run_parameters_text}",
    )
    fig.update_yaxes(title_text="Revenue [$]", row=1, col=1)
    fig.update_yaxes(title_text="Price [$/MWh]", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run monthly multi-day simulation and save the page-3-equivalent plot."
    )
    parser.add_argument(
        "--month",
        required=True,
        help="Target month in YYYY-MM format (example: 2025-06).",
    )
    parser.add_argument(
        "--da_stage_forecast_method",
        "--da-stage-forecast-method",
        default="persistence",
        choices=["persistence", "perfect", "xgboost", "regression"],
        help="Forecast method used in the DA stage.",
    )
    parser.add_argument(
        "--rt_stage_forecast_method",
        "--rt-stage-forecast-method",
        default="persistence",
        choices=["persistence", "perfect", "xgboost", "regression"],
        help="Forecast method used in the RT stage.",
    )
    parser.add_argument(
        "--da_algorithm_method",
        "--da-algorithm-method",
        default="deterministic",
        choices=["deterministic"],
        help="DA optimization algorithm. Currently only deterministic is supported.",
    )
    parser.add_argument(
        "--rt_stage_algorithm",
        "--rt-stage-algorithm",
        "--rt_stage-algorithm",
        default=RT_ALGO_MPC,
        choices=[RT_ALGO_MPC, RT_ALGO_NO_CONTROL],
        help="RT stage algorithm.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory where the PNG figure is saved.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional custom output filename (PNG).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    month_start, month_end = parse_month_bounds(args.month)

    da_forecast_method = args.da_stage_forecast_method
    rt_forecast_method = args.rt_stage_forecast_method
    da_algorithm_method = args.da_algorithm_method
    rt_algorithm_method = args.rt_stage_algorithm

    da_initial_soc = 0.5
    da_end_of_day_soc = 0.5
    rt_end_of_day_soc = 0.5
    rt_control_horizon_type = "receding"
    rt_horizon_hours = 24

    if rt_algorithm_method == RT_ALGO_MPC:
        use_rt_mpc = True
    elif rt_algorithm_method == RT_ALGO_NO_CONTROL:
        use_rt_mpc = False
    else:
        raise ValueError(f"Unknown RT algorithm selected: {rt_algorithm_method}")

    print("Loading ERCOT data...")
    data = load_ercot_data(price_node=PRICE_NODE, verbose=False)
    battery = BatteryParams()

    print(
        f"Running monthly simulation for {args.month} "
        f"(DA algo: {da_algorithm_method}, DA forecast: {da_forecast_method}, "
        f"RT algo: {rt_algorithm_method}, RT forecast: {rt_forecast_method})..."
    )

    run_parameters_text = "<br>".join(
        [
            f"Month: {args.month}",
            f"Price node: {PRICE_NODE}",
            f"DA algorithm: {da_algorithm_method}",
            f"DA forecast: {da_forecast_method}",
            f"RT algorithm: {rt_algorithm_method}",
            f"RT forecast: {rt_forecast_method}",
            f"DA initial SOC: {da_initial_soc:.2f}",
            f"DA end-of-day SOC: {da_end_of_day_soc:.2f}",
            f"RT end-of-day SOC: {rt_end_of_day_soc:.2f}",
            f"RT horizon type: {rt_control_horizon_type}",
            f"RT horizon hours: {rt_horizon_hours}",
        ]
    )

    def on_progress(completed_days: int, total_days: int) -> None:
        print(
            f"\rProgress: {completed_days}/{total_days} days",
            end="",
            flush=True,
        )

    month_result = multi_day_simulation(
        data=data,
        start_day=month_start,
        end_day=month_end,
        battery=battery,
        da_stage_kwargs={
            "initial_soc": da_initial_soc,
            "end_of_day_soc": da_end_of_day_soc,
        },
        use_rt_mpc=use_rt_mpc,
        rt_stage_kwargs={"end_of_day_soc": rt_end_of_day_soc},
        rt_control_horizon_type=rt_control_horizon_type,
        rt_horizon_hours=rt_horizon_hours,
        da_stage_forecast_method=da_forecast_method,
        rt_stage_forecast_method=rt_forecast_method,
        progress_callback=on_progress,
    )
    print("\nSimulation complete.")

    print("Building figure...")
    fig = build_monthly_figure(
        data=data,
        month_start=month_start,
        month_result=month_result,
        da_forecast_method=da_forecast_method,
        rt_forecast_method=rt_forecast_method,
        battery=battery,
        run_parameters_text=run_parameters_text,
    )

    output_path = build_output_path(
        output_dir=args.output_dir,
        month=args.month,
        filename=args.output_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_image(output_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to save PNG plot. Install kaleido (for example: pip install kaleido) "
            "and retry."
        ) from exc

    print(f"Saved monthly plot to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
