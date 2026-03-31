import pandas as pd
import streamlit as st

from src.utils.load_ercot_data import load_ercot_data


@st.cache_data(show_spinner="Loading ERCOT data...")
def get_cached_ercot_data() -> pd.DataFrame:
    return load_ercot_data(verbose=False)
