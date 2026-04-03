import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

from app.utils_app.battery_controls import render_battery_params_expander
from app.utils_app.cumulative_plot import add_schedule_comparison_traces
from app.utils_app.data import get_cached_ercot_data
from app.utils_app.metrics import (
    render_forecast_accuracy_markdown,
    render_revenue_kpis,
)
from app.utils_app.selectors import (
    RT_ALGO_MPC,
    RT_ALGO_NO_CONTROL,
    render_da_algorithm_selector,
    render_month_selector,
    render_price_node_selector,
    render_rt_algorithm_selector,
)
from app.utils_app.simulation_math import (
    compute_daily_tb_strategy_revenues,
    compute_forecast_accuracy_metrics,
    compute_revenue_series,
)
from src.colors import DA_STAGE_RT_FORECAST_COLOR, RT_COLOR
from src.forecasts.build_forecast_vs_actual_plotly_figure import (
    build_forecast_vs_actual_plotly_figure,
)
from src.globals import FREQUENCY, TIME_STEPS_PER_HOUR
from src.multi_day_simulation import multi_day_simulation

st.title("Monthly Simulation")
st.caption("Stage-1 day-ahead scheduling iterated over the selected month")

data = get_cached_ercot_data()
datetime_index = pd.DatetimeIndex(data.index)


col_price_node, col_month = st.columns(2)
with col_price_node:
    _selected_price_node = render_price_node_selector(
        disabled=True,
    )
with col_month:
    month_start, month_end = render_month_selector(datetime_index)

selected_da_algorithm, da_forecast_method, da_initial_soc, da_end_of_day_soc = (
    render_da_algorithm_selector()
)

selected_rt_algorithm, rt_forecast_method, rt_end_of_day_soc = (
    render_rt_algorithm_selector()
)

battery = render_battery_params_expander()

if selected_rt_algorithm == RT_ALGO_NO_CONTROL:
    use_rt_mpc = False
elif selected_rt_algorithm == RT_ALGO_MPC:
    use_rt_mpc = True
else:
    st.error(f"Unknown RT algorithm selected: {selected_rt_algorithm}")
    st.stop()


progress_bar = st.progress(0, text="Running monthly simulation...")


def on_progress(completed_days: int, total_days: int) -> None:
    progress_bar.progress(
        int(completed_days / total_days * 100),
        text=f"Running monthly simulation... {completed_days}/{total_days} days",
    )


try:
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
        rt_control_horizon_type="receding",
        rt_horizon_hours=24,
        da_stage_forecast_method=da_forecast_method,
        rt_stage_forecast_method=rt_forecast_method,
        progress_callback=on_progress,
    )
except Exception as exc:
    progress_bar.empty()
    st.error(f"Monthly simulation failed: {exc}")
    st.stop()

progress_bar.empty()

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

# ==================== TB metric calculations ====================

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

render_revenue_kpis(
    da_stage_actual=da_stage_actual_cumulative[-1],
    rt_stage_actual=rt_stage_actual_cumulative[-1],
    perfect_total=perfect_cumulative[-1],
    tb2_da_total=tb2_da_cumulative[-1],
    tb4_da_total=tb4_da_cumulative[-1],
)


# ==================== Create subplot ====================

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.38, 0.62],
    vertical_spacing=0.06,
    subplot_titles=(
        "Cumulative Revenue Comparison",
        "DA/RT Forecast vs Real Prices",
    ),
)

# ==================== 1st Figure: Cumulative Revenue Comparison ====================
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

# ==================== General layout updates ====================
month_data = data.reindex(month_result.index)

da_price_col = f"{_selected_price_node}_DAM"
rt_price_col = f"{_selected_price_node}_RTM"

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

# ==================== General layout updates ====================

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

# ==================== Additional Information ====================
n_days = len(month_result.operating_days)
st.write(
    f"Using node **{_selected_price_node}**, "
    f"Month: **{month_start:%Y-%m}**, days: **{n_days}**, "
    f"successful runs: **{n_days}**, "
    f"DA algorithm: **{selected_da_algorithm}**, RT algorithm: **{selected_rt_algorithm}**"
)


forecast_accuracy_rows = [
    (
        "DA",
        "DA Stage",
        da_forecast_method,
        compute_forecast_accuracy_metrics(
            forecast=month_result.da_forecast_used,
            actual=month_result.da_forecast_perfect,
        ),
    ),
    (
        "RT",
        "DA Stage",
        da_forecast_method,
        compute_forecast_accuracy_metrics(
            forecast=month_result.da_stage_rt_forecast_used,
            actual=month_result.rt_forecast_perfect,
        ),
    ),
    (
        "RT",
        "RT Stage",
        rt_forecast_method,
        compute_forecast_accuracy_metrics(
            forecast=month_result.rt_forecast_used,
            actual=month_result.rt_forecast_perfect,
        ),
    ),
]
render_forecast_accuracy_markdown(forecast_accuracy_rows)
