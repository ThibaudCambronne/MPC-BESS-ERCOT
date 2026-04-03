from typing import cast

import pandas as pd
import streamlit as st

from src.globals import TYPE_FORECASTS

ALGO_DA_ONLY = "DA Schedule Only"
ALGO_DA_AND_RT_MPC = "DA Schedule + RT MPC Adjustments"

DA_ALGO_DETERMINISTIC = "Deterministic"
RT_ALGO_NO_CONTROL = "No RT Control"
RT_ALGO_MPC = "MPC"


def render_operating_day_selector(
    datetime_index: pd.DatetimeIndex,
    label: str = "Choose operating day",
) -> pd.Timestamp:
    available_dates = pd.Index(datetime_index.date).unique()
    min_date = pd.Timestamp(available_dates.min()).date()
    max_date = pd.Timestamp(available_dates.max()).date()

    selected_date = st.date_input(
        label,
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )
    return pd.Timestamp(selected_date)


def render_month_selector(
    datetime_index: pd.DatetimeIndex,
    label: str = "Choose month",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    months = pd.PeriodIndex(datetime_index.to_period("M").unique()).sort_values()
    selected_month = st.selectbox(label, options=months, index=len(months) - 1)

    month_period = pd.Period(selected_month, freq="M")
    month_start = month_period.start_time
    month_end = month_period.end_time.normalize()
    return month_start, month_end


def render_forecast_method_selector(
    label: str = "Forecast model",
    default_index: int = 0,
) -> TYPE_FORECASTS:
    methods: list[TYPE_FORECASTS] = [
        "persistence",
        "perfect",
        "xgboost",
        "regression",
    ]
    selected = st.selectbox(label, options=methods, index=default_index)
    return cast(TYPE_FORECASTS, selected)


def render_algorithm_selector(
    label: str = "Scheduling algorithm",
    default_index: int = 0,
) -> str:
    return str(
        st.selectbox(
            label,
            options=[ALGO_DA_ONLY, ALGO_DA_AND_RT_MPC],
            index=default_index,
        )
    )


def render_da_algorithm_selector():
    da_algo_col, da_params_col = st.columns(2, vertical_alignment="bottom")
    with da_algo_col:
        selected_da_algorithm = st.selectbox(
            "DA Stage Algorithm",
            options=[DA_ALGO_DETERMINISTIC],
            index=0,
        )
    with da_params_col:
        da_forecast_method, da_initial_soc, da_end_of_day_soc = (
            render_da_control_expander()
        )
    return selected_da_algorithm, da_forecast_method, da_initial_soc, da_end_of_day_soc


def render_rt_algorithm_selector():
    rt_algo_col, rt_params_col = st.columns(2, vertical_alignment="bottom")
    with rt_algo_col:
        selected_rt_algorithm = st.selectbox(
            "RT Stage Algorithm",
            options=[RT_ALGO_NO_CONTROL, RT_ALGO_MPC],
            index=0,
        )
    with rt_params_col:
        rt_forecast_method, rt_end_of_day_soc = render_rt_control_expander()
    return selected_rt_algorithm, rt_forecast_method, rt_end_of_day_soc


def render_da_control_expander() -> tuple[TYPE_FORECASTS, float, float]:
    with st.expander("DA Stage Parameters", expanded=False, icon=":material/menu:"):
        da_forecast_method = render_forecast_method_selector(
            label="DA forecast method",
        )
        col1, col2 = st.columns(2)
        with col1:
            initial_soc = st.number_input(
                "Initial SOC",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="da_initial_soc",
            )
        with col2:
            end_of_day_soc = st.number_input(
                "End of Day SOC",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="da_end_of_day_soc",
            )
    return da_forecast_method, float(initial_soc), float(end_of_day_soc)


def render_rt_control_expander() -> tuple[TYPE_FORECASTS, float]:
    with st.expander("RT Stage Parameters", expanded=False, icon=":material/menu:"):
        rt_forecast_method = render_forecast_method_selector(
            label="RT forecast method",
        )
        end_of_day_soc = st.number_input(
            "End of Day SOC",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="rt_end_of_day_soc",
        )
    return rt_forecast_method, float(end_of_day_soc)
