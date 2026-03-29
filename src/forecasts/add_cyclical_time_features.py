import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def add_cyclical_time_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Encodes cyclical time features (hour and day of week) using sine and cosine transformations."""
    time_in_minutes = df.index.day * 24 * 60 + df.index.hour * 60 + df.index.minute
    # time of the hour
    df["minute_sin"] = np.sin(2 * np.pi * time_in_minutes / 60)
    df["minute_cos"] = np.cos(2 * np.pi * time_in_minutes / 60)

    # Time of the day
    df["hour_sin"] = np.sin(2 * np.pi * time_in_minutes / (24 * 60))
    df["hour_cos"] = np.cos(2 * np.pi * time_in_minutes / (24 * 60))

    # Day of the week
    df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    time_features_cols = [
        "minute_sin",
        "minute_cos",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]

    return df, time_features_cols


def plot_time_features(time_features_cols: list[str], training_data: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    for time_feature in time_features_cols:
        ax.plot(
            training_data.index,
            training_data[time_feature],
            label=time_feature,
        )
    ax.set_title("Time Features")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    fig.show()
