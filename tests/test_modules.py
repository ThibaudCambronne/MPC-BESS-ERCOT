# Basic test skeleton for ERCOT BESS MPC modules
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def test_imports():
    from src.forecaster import get_forecast

    current_time = pd.Timestamp.now()
    print(
        get_forecast(
            data=pd.DataFrame(),
            current_time=current_time,
            horizon_hours=5,
            market="DA",
            method="persistence",
        )
    )
