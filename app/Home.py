import streamlit as st

st.set_page_config(
    page_title="MPC-BESS ERCOT",
    page_icon="battery",
    layout="wide",
)

st.title("MPC-BESS ERCOT Dashboard")
st.write("Use the page selector in the sidebar to explore the different pages.")

st.markdown(
    """
### Available Pages
- **Price Forecasts**: Select a day and market (DA or RT), then compare different forecasts with actual prices.
- **Daily Simulation**: Run day-ahead and real-time scheduling stages for a selected operating day and compare planned/realized/perfect revenues.
- **Monthly Simulation**: Iterate daily simulations over a selected month and compare planned/realized/perfect revenues.
"""
)
