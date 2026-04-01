import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

from app.utils_app.battery_controls import render_battery_params_expander
from app.utils_app.cumulative_plot import add_cumulative_revenue_traces
from app.utils_app.data import get_cached_ercot_data
from app.utils_app.metrics import render_revenue_kpis
from app.utils_app.selectors import (
    render_algorithm_selector,
    render_forecast_method_selector,
    render_month_selector,
)
from app.utils_app.simulation_math import (
    compute_daily_tb_strategy_revenues,
    compute_revenue_series,
    solve_schedule_for_algorithm,
)
from src.forecasts.build_forecast_vs_actual_plotly_figure import (
    build_forecast_vs_actual_plotly_figure,
)
from src.forecasts.forecaster import get_forecasts_for_da
from src.globals import FREQUENCY, PRICE_NODE, TIME_STEPS_PER_HOUR

st.title("Monthly Simulation")
st.caption("Stage-1 day-ahead scheduling iterated over the selected month")

data = get_cached_ercot_data()
datetime_index = pd.DatetimeIndex(data.index)

col_month, col_method, col_algorithm = st.columns(3)
with col_month:
    month_start, month_end = render_month_selector(datetime_index)
with col_method:
    selected_method = render_forecast_method_selector()
with col_algorithm:
    selected_algorithm = render_algorithm_selector()

battery = render_battery_params_expander()

operating_days = pd.date_range(start=month_start, end=month_end, freq="D")
month_index = pd.date_range(
    start=month_start,
    end=month_end + pd.Timedelta(days=1) - pd.Timedelta(minutes=15),
    freq=FREQUENCY,
)

month_n_steps = len(month_index)
planned_step_revenue = np.zeros(month_n_steps)
realized_step_revenue = np.zeros(month_n_steps)
perfect_step_revenue = np.zeros(month_n_steps)

da_price_col = f"{PRICE_NODE}_DAM"
rt_price_col = f"{PRICE_NODE}_RTM"

tb_day_end_timestamps: list[pd.Timestamp] = []
tb2_da_daily_revenue: list[float] = []
tb2_rt_daily_revenue: list[float] = []
tb4_da_daily_revenue: list[float] = []
tb4_rt_daily_revenue: list[float] = []

month_da_forecast = np.full(month_n_steps, np.nan)
month_rt_forecast = np.full(month_n_steps, np.nan)

failed_days: list[str] = []

progress_bar = st.progress(0, text="Running monthly simulation...")

for day_idx, operating_day_start in enumerate(operating_days):
    schedule_time = operating_day_start - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
    n_steps = 24 * TIME_STEPS_PER_HOUR
    day_index = pd.date_range(
        start=operating_day_start, periods=n_steps, freq=FREQUENCY
    )

    pos = month_index.get_indexer(day_index)
    if (pos < 0).any():
        failed_days.append(f"{operating_day_start:%Y-%m-%d} (missing timestamps)")
        progress_bar.progress(
            int((day_idx + 1) / len(operating_days) * 100),
            text=f"Running monthly simulation... {day_idx + 1}/{len(operating_days)} days",
        )
        continue

    try:
        da_forecast, rt_forecast = get_forecasts_for_da(
            data=data,
            current_time=schedule_time,
            horizon_hours=24,
            method=selected_method,
            price_node=PRICE_NODE,
            verbose=False,
        )

        da_forecast_perfect, rt_forecast_perfect = get_forecasts_for_da(
            data=data,
            current_time=schedule_time,
            horizon_hours=24,
            method="perfect",
            price_node=PRICE_NODE,
            verbose=False,
        )

        schedule = solve_schedule_for_algorithm(
            algorithm=selected_algorithm,
            da_price_forecast=da_forecast,
            rt_price_forecast=rt_forecast,
            battery=battery,
            initial_soc=0.5,
            end_of_day_soc=0.5,
        )
        schedule_perfect = solve_schedule_for_algorithm(
            algorithm=selected_algorithm,
            da_price_forecast=da_forecast_perfect,
            rt_price_forecast=rt_forecast_perfect,
            battery=battery,
            initial_soc=0.5,
            end_of_day_soc=0.5,
        )

        day_planned, day_realized, day_perfect = compute_revenue_series(
            da_forecast=da_forecast,
            rt_forecast=rt_forecast,
            da_bids=schedule.da_energy_bids,
            rt_bids=schedule.rt_energy_bids,
            da_forecast_perfect=da_forecast_perfect,
            rt_forecast_perfect=rt_forecast_perfect,
            da_bids_perfect=schedule_perfect.da_energy_bids,
            rt_bids_perfect=schedule_perfect.rt_energy_bids,
        )

        planned_step_revenue[pos] = day_planned
        realized_step_revenue[pos] = day_realized
        perfect_step_revenue[pos] = day_perfect

        month_da_forecast[pos] = da_forecast.values
        month_rt_forecast[pos] = rt_forecast.values

        day_tb_revenues = compute_daily_tb_strategy_revenues(
            da_prices=da_forecast_perfect,
            rt_prices=rt_forecast_perfect,
            battery=battery,
        )
        tb_day_end_timestamps.append(day_index[-1])
        tb2_da_daily_revenue.append(day_tb_revenues["tb2_da_revenue"])
        tb2_rt_daily_revenue.append(day_tb_revenues["tb2_rt_revenue"])
        tb4_da_daily_revenue.append(day_tb_revenues["tb4_da_revenue"])
        tb4_rt_daily_revenue.append(day_tb_revenues["tb4_rt_revenue"])

    except Exception as exc:
        failed_days.append(f"{operating_day_start:%Y-%m-%d} ({exc})")

    progress_bar.progress(
        int((day_idx + 1) / len(operating_days) * 100),
        text=f"Running monthly simulation... {day_idx + 1}/{len(operating_days)} days",
    )

progress_bar.empty()

planned_cumulative = np.cumsum(planned_step_revenue)
realized_cumulative = np.cumsum(realized_step_revenue)
perfect_cumulative = np.cumsum(perfect_step_revenue)

tb2_da_cumulative = np.cumsum(tb2_da_daily_revenue)
tb2_rt_cumulative = np.cumsum(tb2_rt_daily_revenue)
tb4_da_cumulative = np.cumsum(tb4_da_daily_revenue)
tb4_rt_cumulative = np.cumsum(tb4_rt_daily_revenue)

render_revenue_kpis(
    planned_total=planned_cumulative[-1],
    realized_total=realized_cumulative[-1],
    perfect_total=perfect_cumulative[-1],
    tb2_da_total=tb2_da_cumulative[-1],
    tb4_da_total=tb4_da_cumulative[-1],
)

if failed_days:
    preview = "\n".join(failed_days[:5])
    suffix = "" if len(failed_days) <= 5 else f"\n... and {len(failed_days) - 5} more"
    st.warning(
        f"Failed days: {len(failed_days)}/{len(operating_days)}\n\n{preview}{suffix}"
    )

month_data = data.reindex(month_index)

price_fig = build_forecast_vs_actual_plotly_figure(
    current_time=month_start,
    data=month_data,
    forecasts={selected_method: pd.Series(month_da_forecast, index=month_index)},
    market="DA/RT",
    price_col=da_price_col,
    rt_forecasts={selected_method: pd.Series(month_rt_forecast, index=month_index)},
    rt_price_col=rt_price_col,
    highlight_market_order_mismatch=True,
    historical_days=0,
    visible_history_hours=0,
)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.38, 0.62],
    vertical_spacing=0.06,
    subplot_titles=(
        "Cumulative Revenue",
        "DA/RT Forecast vs Real Prices",
    ),
)

tb_marker_series = [
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
]

add_cumulative_revenue_traces(
    fig=fig,
    x_values=month_index,
    planned_cumulative=planned_cumulative,
    realized_cumulative=realized_cumulative,
    perfect_cumulative=perfect_cumulative,
    row=1,
    col=1,
    legendgroup="subplot1",
    tb_marker_series=tb_marker_series,
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
    template="plotly_white",
    hovermode="x unified",
    showlegend=True,
    legend=dict(
        xanchor="left",
        yanchor="top",
    ),
    legend_tracegroupgap=220,
)
fig.update_yaxes(title_text="Revenue [$]", row=1, col=1)
fig.update_yaxes(title_text="Price [$/MWh]", row=2, col=1)
fig.update_xaxes(title_text="Time", row=2, col=1)

st.plotly_chart(fig, width="stretch")

st.write(
    f"Month: **{month_start:%Y-%m}**, days: **{len(operating_days)}**, "
    f"successful runs: **{len(operating_days) - len(failed_days)}**, "
    f"model: **{selected_method}**, algorithm: **{selected_algorithm}**"
)
