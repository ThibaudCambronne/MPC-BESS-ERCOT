import streamlit as st

from src.utils.battery_model import BatteryParams


def render_battery_params_expander(
    default: BatteryParams | None = None,
    expanded: bool = False,
) -> BatteryParams:
    default_battery_params = default or BatteryParams()

    with st.expander("⚙️ Battery Parameters", expanded=expanded):
        col1, col2, col3 = st.columns(3, gap="medium")
        with col1:
            capacity_mwh = st.number_input(
                "Capacity [MWh]",
                value=float(default_battery_params.capacity_mwh),
                min_value=1.0,
                step=10.0,
            )
            power_max_mw = st.number_input(
                "Max Power [MW]",
                value=float(default_battery_params.power_max_mw),
                min_value=0.1,
                step=1.0,
            )
        with col2:
            soc_min = st.slider(
                "Min SOC",
                min_value=0.0,
                max_value=0.4,
                value=float(default_battery_params.soc_min),
                step=0.05,
            )
            soc_max = st.slider(
                "Max SOC",
                min_value=0.6,
                max_value=1.0,
                value=float(default_battery_params.soc_max),
                step=0.05,
            )
        with col3:
            efficiency_charge = st.slider(
                "Charge Efficiency",
                min_value=0.8,
                max_value=1.0,
                value=float(default_battery_params.efficiency_charge),
                step=0.01,
            )
            efficiency_discharge = st.slider(
                "Discharge Efficiency",
                min_value=0.8,
                max_value=1.0,
                value=float(default_battery_params.efficiency_discharge),
                step=0.01,
            )
            throughput_limit = st.number_input(
                "Throughput Limit [MWh]",
                value=float(default_battery_params.throughput_limit),
                min_value=1.0,
                step=10.0,
            )

    if soc_min >= soc_max:
        st.warning("Min SOC must be lower than Max SOC. Using default SOC bounds.")
        soc_min = default_battery_params.soc_min
        soc_max = default_battery_params.soc_max

    return BatteryParams(
        capacity_mwh=capacity_mwh,
        power_max_mw=power_max_mw,
        soc_min=soc_min,
        soc_max=soc_max,
        efficiency_charge=efficiency_charge,
        efficiency_discharge=efficiency_discharge,
        throughput_limit=throughput_limit,
    )
