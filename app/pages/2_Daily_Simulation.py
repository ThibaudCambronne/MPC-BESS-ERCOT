import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots

from app.utils_app.battery_controls import render_battery_params_expander
from app.utils_app.cumulative_plot import (
    add_bid_soc_panel_traces,
    add_schedule_comparison_traces,
)
from app.utils_app.data import get_cached_ercot_data
from app.utils_app.metrics import render_revenue_kpis
from app.utils_app.selectors import (
    RT_ALGO_MPC,
    RT_ALGO_NO_CONTROL,
    render_da_algorithm_selector,
    render_operating_day_selector,
    render_rt_algorithm_selector,
)
from app.utils_app.simulation_math import (
    compute_daily_tb_strategy_revenues,
    compute_revenue_series,
)
from src.colors import (
    DA_COLOR,
    DA_STAGE_DA_FORECAST_COLOR,
    DA_STAGE_RT_FORECAST_COLOR,
    RT_COLOR,
    SOC_COLOR,
)
from src.forecasts.build_forecast_vs_actual_plotly_figure import (
    build_forecast_vs_actual_plotly_figure,
)
from src.globals import (
    FREQUENCY,
    PRICE_NODE,
    TIME_STEPS_PER_HOUR,
)
from src.one_day_simulation import one_day_simulation

st.title("Daily Simulation")
st.caption("Stage-1 day-ahead scheduling: planned vs realized revenue")

data = get_cached_ercot_data()
datetime_index = pd.DatetimeIndex(data.index)

operating_day_start = render_operating_day_selector(datetime_index)


selected_da_algorithm, da_forecast_method, da_initial_soc, da_end_of_day_soc = (
    render_da_algorithm_selector()
)


selected_rt_algorithm, rt_forecast_method, rt_end_of_day_soc = (
    render_rt_algorithm_selector()
)

battery = render_battery_params_expander()

horizon_hours = 24

n_steps = horizon_hours * TIME_STEPS_PER_HOUR
operating_index = pd.date_range(
    start=operating_day_start,
    periods=n_steps,
    freq=FREQUENCY,
)

try:
    if selected_rt_algorithm == RT_ALGO_NO_CONTROL:
        use_rt_mpc = False
    elif selected_rt_algorithm == RT_ALGO_MPC:
        use_rt_mpc = True
    else:
        raise ValueError(f"Unknown RT algorithm selected: {selected_rt_algorithm}")

    schedule_final = one_day_simulation(
        data=data,
        operating_day=operating_day_start,
        battery=battery,
        daily_simulation_horizon_hours=horizon_hours,
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
    )

    schedule_perfect = one_day_simulation(
        data=data,
        operating_day=operating_day_start,
        battery=battery,
        daily_simulation_horizon_hours=horizon_hours,
        da_stage_kwargs={
            "initial_soc": da_initial_soc,
            "end_of_day_soc": da_end_of_day_soc,
        },
        use_rt_mpc=False,
        rt_stage_kwargs={},
        rt_control_horizon_type="receding",
        rt_horizon_hours=24,
        da_stage_forecast_method="perfect",
        rt_stage_forecast_method="perfect",
    )
except Exception as exc:
    print(exc)
    st.error(f"Scheduling run failed: {exc}")
    st.stop()

(
    da_stage_planned_revenue,
    da_stage_actual_revenue,
    rt_stage_planned_revenue,
    rt_stage_actual_revenue,
    perfect_revenue,
) = compute_revenue_series(
    da_forecast=schedule_final.da_forecast_used,
    da_stage_rt_forecast_used=schedule_final.da_stage_rt_forecast_used,
    rt_forecast=schedule_final.rt_forecast_used,
    da_bids=schedule_final.da_energy_bids,
    da_stage_rt_energy_bids=schedule_final.da_stage_rt_energy_bids,
    rt_bids=schedule_final.rt_energy_bids,
    da_forecast_perfect=schedule_perfect.da_forecast_used,
    rt_forecast_perfect=schedule_perfect.rt_forecast_used,
    da_bids_perfect=schedule_perfect.da_energy_bids,
    rt_bids_perfect=schedule_perfect.rt_energy_bids,
)

da_stage_planned_cumulative = np.cumsum(da_stage_planned_revenue)
da_stage_actual_cumulative = np.cumsum(da_stage_actual_revenue)
rt_stage_planned_cumulative = np.cumsum(rt_stage_planned_revenue)
rt_stage_actual_cumulative = np.cumsum(rt_stage_actual_revenue)
perfect_cumulative = np.cumsum(perfect_revenue)

tb_revenues = compute_daily_tb_strategy_revenues(
    da_prices=schedule_perfect.da_forecast_used,
    rt_prices=schedule_perfect.rt_forecast_used,
    battery=battery,
)

render_revenue_kpis(
    da_stage_actual=da_stage_actual_cumulative[-1],
    rt_stage_actual=rt_stage_actual_cumulative[-1],
    perfect_total=perfect_cumulative[-1],
    tb2_da_total=tb_revenues["tb2_da_revenue"],
    tb4_da_total=tb_revenues["tb4_da_revenue"],
)

# ==================== Graph Creation ====================

fig = make_subplots(
    rows=4,
    shared_xaxes=True,
    row_heights=[0.22, 0.18, 0.18, 0.42],
    vertical_spacing=0.05,
    subplot_titles=(
        "Cumulative Revenue Comparison",
        "DA stage Schedule: Energy Bids and SOC",
        "Final Schedule: Energy Bids and SOC",
        "DA/RT Forecast vs Real Prices",
    ),
    specs=[
        [{"secondary_y": False}],
        [{"secondary_y": True}],
        [{"secondary_y": True}],
        [{"secondary_y": False}],
    ],
)


# ==================== 1st figure: Cumulative revenue comparison ====================

day_end_x = [operating_index[-1]]
add_schedule_comparison_traces(
    fig=fig,
    x_values=operating_index,
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
            "x": day_end_x,
            "y": [tb_revenues["tb2_da_revenue"]],
            "name": "TB2 DA Revenue",
        },
        {
            "x": day_end_x,
            "y": [tb_revenues["tb2_rt_revenue"]],
            "name": "TB2 RT Revenue",
        },
        {
            "x": day_end_x,
            "y": [tb_revenues["tb4_da_revenue"]],
            "name": "TB4 DA Revenue",
        },
        {
            "x": day_end_x,
            "y": [tb_revenues["tb4_rt_revenue"]],
            "name": "TB4 RT Revenue",
        },
    ],
)

# ==================== 2nd figure: DA-only schedule bids + SOC ====================

add_bid_soc_panel_traces(
    fig=fig,
    operating_index=operating_index,
    operating_day_start=operating_day_start,
    bid_traces=[
        {
            "y": schedule_final.da_energy_bids,
            "name": "DA Bid (MW)",
            "color": DA_STAGE_DA_FORECAST_COLOR,
        },
        {
            "y": schedule_final.da_stage_rt_energy_bids,
            "name": "RT Bid (MW)",
            "color": DA_STAGE_RT_FORECAST_COLOR,
        },
    ],
    soc_values=schedule_final.da_stage_soc_schedule,
    soc_name="Battery SOC",
    soc_color=SOC_COLOR,
    row=2,
    col=1,
    legendgroup="subplot2",
)

# ==================== 3rd figure: final schedule bids + SOC ====================

add_bid_soc_panel_traces(
    fig=fig,
    operating_index=operating_index,
    operating_day_start=operating_day_start,
    bid_traces=[
        {
            "y": schedule_final.da_energy_bids,
            "name": "Final DA Bid (MW)",
            "color": DA_COLOR,
        },
        {
            "y": schedule_final.rt_energy_bids,
            "name": "Final RT Bid (MW)",
            "color": RT_COLOR,
        },
    ],
    soc_values=schedule_final.soc_schedule,
    soc_name="Final SOC",
    soc_color=SOC_COLOR,
    row=3,
    col=1,
    legendgroup="subplot3",
)

# ==================== 4th figure: Prices ====================

price_fig = build_forecast_vs_actual_plotly_figure(
    current_time=operating_day_start,
    data=data,
    forecasts={da_forecast_method: schedule_final.da_forecast_used},
    market="DA/RT",
    price_col=f"{PRICE_NODE}_DAM",
    rt_forecasts={
        f"DA Stage {da_forecast_method}": schedule_final.da_stage_rt_forecast_used,
        f"RT Stage {rt_forecast_method}": schedule_final.rt_forecast_used,
    },
    rt_forecast_colors={
        f"DA Stage {da_forecast_method}": DA_STAGE_RT_FORECAST_COLOR,
        f"RT Stage {rt_forecast_method}": RT_COLOR,
    },
    rt_price_col=f"{PRICE_NODE}_RTM",
    highlight_market_order_mismatch=True,
    historical_days=2,
    visible_history_hours=8,
)
for trace in price_fig.data:
    trace.legendgroup = "subplot4"  # type: ignore
    fig.add_trace(trace, row=4, col=1)

for shape in price_fig.layout.shapes or []:
    fig.add_shape(shape, row=4, col=1)

if price_fig.layout.xaxis.range:
    fig.update_xaxes(range=list(price_fig.layout.xaxis.range), row=4, col=1)

# ==================== General layout updates ====================
fig.update_layout(
    height=1240,
    barmode="relative",
    template="plotly_white",
    hovermode="x unified",
    showlegend=True,
    legend=dict(
        xanchor="left",
        yanchor="top",
    ),
    legend_tracegroupgap=140,
)
fig.update_yaxes(title_text="Revenue [$]", row=1, col=1)
fig.update_yaxes(title_text="Power [MW]", row=2, col=1, secondary_y=False)
fig.update_yaxes(
    {"title_text": "SOC [%]", "range": [0, 100]}, row=2, col=1, secondary_y=True
)
fig.update_yaxes(title_text="Power [MW]", row=3, col=1, secondary_y=False)
fig.update_yaxes(
    {"title_text": "SOC [%]", "range": [0, 100]}, row=3, col=1, secondary_y=True
)
fig.update_yaxes(title_text="Price [$/MWh]", row=4, col=1, secondary_y=False)
fig.update_xaxes(title_text="Time", row=4, col=1)

st.plotly_chart(fig, width="stretch")

schedule_time = operating_day_start - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
st.write(
    f"Schedule run time: **{schedule_time:%Y-%m-%d %H:%M}**, operating day: **{operating_day_start:%Y-%m-%d}**, DA algorithm: **{selected_da_algorithm}**, RT algorithm: **{selected_rt_algorithm}**, DA model: **{da_forecast_method}**, RT model: **{rt_forecast_method}**"
)
