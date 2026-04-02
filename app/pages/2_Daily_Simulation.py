import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.utils_app.battery_controls import render_battery_params_expander
from app.utils_app.cumulative_plot import add_cumulative_revenue_traces
from app.utils_app.data import get_cached_ercot_data
from app.utils_app.metrics import render_revenue_kpis
from app.utils_app.selectors import (
    ALGO_DA_AND_RT_MPC,
    ALGO_DA_ONLY,
    render_algorithm_selector,
    render_forecast_method_selector,
    render_operating_day_selector,
)
from app.utils_app.simulation_math import (
    compute_daily_tb_strategy_revenues,
    compute_revenue_series,
)
from src.colors import (
    DA_COLOR,
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

col_date, col_method, col_algorithm = st.columns(3)
with col_date:
    operating_day_start = render_operating_day_selector(datetime_index)

with col_method:
    selected_forecast_method = render_forecast_method_selector()

with col_algorithm:
    selected_algorithm = render_algorithm_selector()

battery = render_battery_params_expander()

horizon_hours = 24

n_steps = horizon_hours * TIME_STEPS_PER_HOUR
operating_index = pd.date_range(
    start=operating_day_start,
    periods=n_steps,
    freq=FREQUENCY,
)

try:
    if selected_algorithm == ALGO_DA_ONLY:
        use_rt_mpc = False
    elif selected_algorithm == ALGO_DA_AND_RT_MPC:
        use_rt_mpc = True
    else:
        raise ValueError(f"Unknown algorithm selected: {selected_algorithm}")
    schedule = one_day_simulation(
        data=data,
        operating_day=operating_day_start,
        battery=battery,
        daily_simulation_horizon_hours=horizon_hours,
        da_schedule_kwargs={
            "initial_soc": 0.5,
            "end_of_day_soc": 0.5,
        },
        use_rt_mpc=use_rt_mpc,
        rt_schedule_kwargs={},
        rt_control_horizon_type="receding",
        rt_horizon_hours=24,
        forecast_method=selected_forecast_method,
    )
    schedule_perfect = one_day_simulation(
        data=data,
        operating_day=operating_day_start,
        battery=battery,
        daily_simulation_horizon_hours=horizon_hours,
        da_schedule_kwargs={
            "initial_soc": 0.5,
            "end_of_day_soc": 0.5,
        },
        use_rt_mpc=False,
        rt_schedule_kwargs={},
        rt_control_horizon_type="receding",
        rt_horizon_hours=24,
        forecast_method="perfect",
    )
except Exception as exc:
    print(exc)
    st.error(f"{selected_algorithm} Day-ahead scheduling failed: {exc}")
    st.stop()

planned_step_revenue, realized_step_revenue, perfect_step_revenue = (
    compute_revenue_series(
        da_forecast=schedule.da_forecast_used,
        rt_forecast=schedule.rt_forecast_used,
        da_bids=schedule.da_energy_bids,
        da_plan_for_rt_energy_bids=schedule.da_plan_for_rt_energy_bids,
        rt_bids=schedule.rt_energy_bids,
        da_forecast_perfect=schedule_perfect.da_forecast_used,
        rt_forecast_perfect=schedule_perfect.rt_forecast_used,
        da_bids_perfect=schedule_perfect.da_energy_bids,
        rt_bids_perfect=schedule_perfect.rt_energy_bids,
    )
)

planned_cumulative = np.cumsum(planned_step_revenue)
realized_cumulative = np.cumsum(realized_step_revenue)
perfect_cumulative = np.cumsum(perfect_step_revenue)

tb_revenues = compute_daily_tb_strategy_revenues(
    da_prices=schedule_perfect.da_forecast_used,
    rt_prices=schedule_perfect.rt_forecast_used,
    battery=battery,
)

render_revenue_kpis(
    planned_total=planned_cumulative[-1],
    realized_total=realized_cumulative[-1],
    perfect_total=perfect_cumulative[-1],
    tb2_da_total=tb_revenues["tb2_da_revenue"],
    tb4_da_total=tb_revenues["tb4_da_revenue"],
)

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.30, 0.22, 0.48],
    vertical_spacing=0.05,
    subplot_titles=(
        "Cumulative Revenue",
        "Planned Energy Bids (Signed MW)",
        "DA/RT Forecast vs Real Prices",
    ),
    specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
)


# 1st figure: Cumulative revenue comparison

day_end_x = [operating_index[-1]]
add_cumulative_revenue_traces(
    fig=fig,
    x_values=operating_index,
    planned_cumulative=planned_cumulative,
    realized_cumulative=realized_cumulative,
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

# 2nd figure: DA/RT bids + SOC

fig.add_trace(
    go.Bar(
        x=operating_index,
        y=np.round(schedule.da_energy_bids, 1),
        name="DA Bid (MW)",
        marker_color=DA_COLOR,
        legendgroup="subplot2",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=operating_index,
        y=np.round(schedule.rt_energy_bids, 1),
        name="RT Bid (MW)",
        marker_color=RT_COLOR,
        legendgroup="subplot2",
    ),
    row=2,
    col=1,
)

# Add SOC trajectory on secondary y-axis
# soc_schedule has length n_steps+1, so we need to align it with operating_index
soc_index = pd.date_range(
    start=operating_day_start,
    periods=len(schedule.soc_schedule),
    freq=FREQUENCY,
)
fig.add_trace(
    go.Scatter(
        x=soc_index,
        y=np.round(schedule.soc_schedule * 100, 0),
        mode="lines",
        name="Battery SOC",
        line={"color": SOC_COLOR, "width": 2, "dash": "dot"},
        legendgroup="subplot2",
    ),
    row=2,
    col=1,
    secondary_y=True,
)

# 3rd figure: Prices
price_fig = build_forecast_vs_actual_plotly_figure(
    current_time=operating_day_start,
    data=data,
    forecasts={selected_forecast_method: schedule.da_forecast_used},
    market="DA/RT",
    price_col=f"{PRICE_NODE}_DAM",
    rt_forecasts={selected_forecast_method: schedule.rt_forecast_used},
    rt_price_col=f"{PRICE_NODE}_RTM",
    highlight_market_order_mismatch=True,
    historical_days=2,
    visible_history_hours=8,
)
for trace in price_fig.data:
    trace.legendgroup = "subplot3"  # type: ignore
    fig.add_trace(trace, row=3, col=1)

for shape in price_fig.layout.shapes or []:
    fig.add_shape(shape, row=3, col=1)

if price_fig.layout.xaxis.range:
    fig.update_xaxes(range=list(price_fig.layout.xaxis.range), row=3, col=1)

fig.update_layout(
    height=980,
    barmode="relative",
    template="plotly_white",
    hovermode="x unified",
    showlegend=True,
    legend=dict(
        xanchor="left",
        yanchor="top",
    ),
    legend_tracegroupgap=160,
)
fig.update_yaxes(title_text="Revenue [$]", row=1, col=1)
fig.update_yaxes(title_text="Power [MW]", row=2, col=1, secondary_y=False)
fig.update_yaxes(
    {"title_text": "SOC [%]", "range": [0, 100]}, row=2, col=1, secondary_y=True
)
fig.update_yaxes(title_text="Price [$/MWh]", row=3, col=1, secondary_y=False)
fig.update_xaxes(title_text="Time", row=3, col=1)

st.plotly_chart(fig, width="stretch")

schedule_time = operating_day_start - pd.Timedelta(days=1) + pd.Timedelta(hours=10)
st.write(
    f"Schedule run time: **{schedule_time:%Y-%m-%d %H:%M}**, operating day: **{operating_day_start:%Y-%m-%d}**, model: **{selected_forecast_method}**"
)
