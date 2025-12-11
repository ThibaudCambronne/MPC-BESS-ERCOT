from pathlib import Path

DATA_PATH_DAM_TRAINING = (
    Path(__file__).parent.parent / "data" / "All_2020_2024_with_AS.csv"
)
DATA_PATH_DAM_TESTING = Path(__file__).parent.parent / "data" / "All_2025_with_AS.csv"
DATA_PATH_RTM = Path(__file__).parent.parent / "data" / "RTM_all_2020_2025_enriched.csv"

DELTA_T = 0.25  # Time step in hours (15 minutes)
TIME_STEPS_PER_HOUR = int(1 / DELTA_T)
