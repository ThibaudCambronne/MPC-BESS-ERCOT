import pandas as pd
import streamlit as st

from src.globals import PRICE_NODE
from src.utils.load_ercot_data import load_ercot_data


@st.cache_data(show_spinner="Loading ERCOT data...")
def get_cached_ercot_data(price_node: str = PRICE_NODE) -> pd.DataFrame:
    return load_ercot_data(price_node=price_node, verbose=False)
