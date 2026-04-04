from pathlib import Path
from typing import Literal

DATA_FOLDER = Path(__file__).parent.parent / "data"
PATH_DAM_TRAINING_DATA = DATA_FOLDER / "All_2020_2024_with_AS.csv"
PATH_DAM_TRAINING_DATA_SAMPLE = DATA_FOLDER / "All_2020_2024_with_AS_sample_3m.csv"
PATH_DAM_TESTING_DATA = DATA_FOLDER / "All_2025_with_AS.csv"
PATH_DAM_TESTING_DATA_SAMPLE = DATA_FOLDER / "All_2025_with_AS_sample_3m.csv"
PATH_RTM_DATA = DATA_FOLDER / "RTM_all_2020_2025_enriched.csv"
PATH_RTM_DATA_SAMPLE = DATA_FOLDER / "RTM_all_2020_2025_enriched_sample_3m.csv"

DELTA_T = 0.25  # Time step, in hours (e.g., 0.25 h = 15 minutes)
TIME_STEPS_PER_HOUR = int(1 / DELTA_T)
FREQUENCY = f"{60 // TIME_STEPS_PER_HOUR}min"


MPC_BATTERY_EPSILON = 1e-6  # Battery capacity in MWh


TYPE_FORECASTS = Literal["persistence", "perfect", "xgboost", "regression"]
TYPE_RT_HORIZON = Literal["shrinking", "receding"]

WEATHER_FEATURES = [
    "dew_point_temperature_S",
    "temperature_S",
]
PRICE_NODE = "HB_SOUTH"  # Price node to use for forecasts and optimization
POSSIBLE_PRICE_NODES = [
    "HB_SOUTH",
    "HB_BUSAVG",
    "HB_HOUSTON",
    "HB_HUBAVG",
    "HB_NORTH",
    "HB_PAN",
    "HB_WEST",
    "LZ_AEN",
    "LZ_CPS",
    "LZ_HOUSTON",
    "LZ_LCRA",
    "LZ_NORTH",
    "LZ_RAYBN",
    "LZ_SOUTH",
    "LZ_WEST",
]
