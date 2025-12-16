import pandas as pd

from src.globals import (
    DATA_PATH_DAM_TESTING,
    DATA_PATH_DAM_TRAINING,
    DATA_PATH_RTM,
    PRICE_NODE,
)


def load_ercot_data() -> pd.DataFrame:
    """
    Load ERCOT price and ancillary services data from a CSV file,
    including 'dew_point_temperature_S' for the regression forecast method.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and all columns from the CSV.
    """
    # Define the core columns needed for both price and regression features
    DAM_COLUMNS = ["key", f"{PRICE_NODE}_DAM", "dew_point_temperature_S"]
    RTM_COLUMNS = [
        "hour_timestamp",
        PRICE_NODE,
    ]  # No need to get the weather data again from RTM file

    # ====================
    # Load DAM data
    # NOTE: Including "dew_point_temperature_S" column
    try:
        df_dam_train = pd.read_csv(DATA_PATH_DAM_TRAINING, usecols=DAM_COLUMNS)
        df_dam_test = pd.read_csv(DATA_PATH_DAM_TESTING, usecols=DAM_COLUMNS)
    except ValueError as e:
        # Handle case where the new column might be missing from the files
        print(f"Error loading DAM data with required columns: {e}")
        print(
            f"Ensure that 'dew_point_temperature_S' is present in {DATA_PATH_DAM_TRAINING} and {DATA_PATH_DAM_TESTING}."
        )
        raise

    df_dam = pd.concat([df_dam_train, df_dam_test], ignore_index=True)

    # Specify the correct datetime format for parsing
    # Example format: '01/01/2020 1' -> '%m/%d/%Y %H'
    date_str = df_dam["key"].str.slice(0, 10)  # "MM/DD/YYYY"
    date_parsed = pd.to_datetime(date_str, format="%m/%d/%Y")
    hours = df_dam["key"].str.slice(11).astype(int) - 1
    df_dam["key"] = date_parsed + pd.to_timedelta(hours, unit="h")

    # Resample to 15-min intervals by forward filling (this carries the dew point temp forward too)
    df_dam = (
        df_dam.drop_duplicates()
        .set_index("key")
        .resample("15min")
        .ffill()
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

    # Merge DAM and RTM on datetime. DAM now contains 'dew_point_temperature_S'.
    df_all = df_dam.merge(
        df_rtm,
        on="key",
        how="inner",
    ).set_index("key")

    df_all["date_str"] = df_all.index.strftime("%m/%d/%Y")
    group_sizes = df_all.groupby("date_str").size()

    hours_per_day_rtm = 24 * 4
    full_days_rtm = group_sizes[group_sizes == hours_per_day_rtm].index
    df_all = df_all[df_all["date_str"].isin(full_days_rtm)].drop(columns=["date_str"])

    # Final check to ensure the required column is present
    if "dew_point_temperature_S" not in df_all.columns:
        raise RuntimeError(
            "The column 'dew_point_temperature_S' is missing after data loading and merging. Please check your source files."
        )

    return df_all
