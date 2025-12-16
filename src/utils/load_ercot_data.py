import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.globals import (
    DATA_PATH_DAM_TESTING,
    DATA_PATH_DAM_TRAINING,
    DATA_PATH_RTM,
    PRICE_NODE,
    WEATHER_FEATURES,
)


def load_ercot_data(verbose: bool = True) -> pd.DataFrame:
    """
    Load ERCOT price and ancillary services data from a CSV file,
    including all WEATHER_FEATURES for the regression forecast method.

    Parameters
    ----------
    verbose : bool, optional
        If True, displays information about missing data and interpolation, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and all columns from the CSV.
    """
    # Define the core columns needed for both price and regression features
    DAM_COLUMNS = [
        "key",
        f"{PRICE_NODE}_DAM",
    ] + WEATHER_FEATURES
    RTM_COLUMNS = [
        "hour_timestamp",
        PRICE_NODE,
    ]  # No need to get the weather data again from RTM file

    # ====================
    # Load DAM data
    # NOTE: Including WEATHER_FEATURES columns
    try:
        df_dam_train = pd.read_csv(DATA_PATH_DAM_TRAINING, usecols=DAM_COLUMNS)
        df_dam_test = pd.read_csv(DATA_PATH_DAM_TESTING, usecols=DAM_COLUMNS)
    except ValueError as e:
        # Handle case where the new column might be missing from the files
        print(f"Error loading DAM data with required columns: {e}")
        print(
            f"Ensure that all WEATHER_FEATURES are present in {DATA_PATH_DAM_TRAINING} and {DATA_PATH_DAM_TESTING}."
        )
        raise

    df_dam = pd.concat([df_dam_train, df_dam_test], ignore_index=True)

    # Specify the correct datetime format for parsing
    # Example format: '01/01/2020 1' -> '%m/%d/%Y %H'
    date_str = df_dam["key"].str.slice(0, 10)  # "MM/DD/YYYY"
    date_parsed = pd.to_datetime(date_str, format="%m/%d/%Y")
    hours = df_dam["key"].str.slice(11).astype(int) - 1
    df_dam["key"] = date_parsed + pd.to_timedelta(hours, unit="h")

    # Resample to 15-min intervals by forward filling
    df_dam = (
        df_dam.drop_duplicates()
        .set_index("key")
        .resample("15min")
        .ffill(limit=4)
        .reset_index()
    )

    # ====================
    # Load RTM data
    df_rtm = pd.read_csv(DATA_PATH_RTM, usecols=RTM_COLUMNS)

    df_rtm = df_rtm.rename(
        columns={PRICE_NODE: f"{PRICE_NODE}_RTM", "hour_timestamp": "key"}
    )

    df_rtm["key"] = pd.to_datetime(df_rtm["key"])

    # The key are missing the minutes (the data has for instance 4 values for 00:00 instead of 00:00, 00:15, 00:30, 00:45)
    # We fix that here by adding the minutes based on the occurrence within each hour
    df_rtm["minute"] = df_rtm.groupby(df_rtm["key"]).cumcount() * 15
    df_rtm["key"] = df_rtm["key"] + pd.to_timedelta(df_rtm["minute"], unit="m")
    df_rtm = df_rtm.drop(columns=["minute"])

    # ====================
    # Merge DAM and RTM on datetime. DAM now contains all WEATHER_FEATURES.
    df_all = df_dam.merge(
        df_rtm,
        on="key",
        how="inner",
    ).set_index("key")

    # Ensure only full days are kept
    df_all["date_str"] = df_all.index.strftime("%m/%d/%Y")  # type: ignore
    group_sizes = df_all.groupby("date_str").size()

    hours_per_day_rtm = 24 * 4
    full_days_rtm = group_sizes[group_sizes == hours_per_day_rtm].index
    full_days_dam = group_sizes[group_sizes == hours_per_day_rtm].index
    df_all = df_all[
        df_all["date_str"].isin(full_days_rtm) & df_all["date_str"].isin(full_days_dam)
    ].drop(columns=["date_str"])

    columns_to_interpolate = WEATHER_FEATURES

    df_missing = pd.DataFrame()
    if verbose:
        df_missing = df_all.loc[:, columns_to_interpolate].isna().astype(int)

    df_all[columns_to_interpolate] = df_all[columns_to_interpolate].interpolate(
        method="time"
    )

    if verbose:
        plot_start = pd.Timestamp("2025-01-15 00:00:00")
        plot_end = pd.Timestamp("2025-01-20 23:45:00")
        plot_data = df_all.loc[plot_start:plot_end].copy()
        fig, axs = plt.subplots(1, len(columns_to_interpolate), figsize=(10, 4))
        if len(columns_to_interpolate) == 1:
            axs = [axs]
        for i, col in enumerate(columns_to_interpolate):
            max_val = plot_data[col].max()
            min_val = plot_data[col].min()

            axs[i].bar(
                plot_data.index,
                np.where(df_missing.loc[plot_data.index, col] == 1, max_val, 0),
                width=0.01,
                color="red",
                alpha=0.3,
                label="Missing Data",
            )
            axs[i].bar(
                plot_data.index,
                np.where(df_missing.loc[plot_data.index, col] == 1, min_val, 0),
                width=0.01,
                color="red",
                alpha=0.3,
            )

            axs[i].plot(plot_data.index, plot_data[col], label=col)
            axs[i].set_title("Interpolated Features")
            axs[i].set_xlabel("Time")
        fig.legend()
        fig.show()

    return df_all
