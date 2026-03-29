import numpy as np
import pandas as pd

from src.forecasts.add_cyclical_time_features import (
    add_cyclical_time_features,
    plot_time_features,
)
from src.globals import WEATHER_FEATURES


def prepare_training_data(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    price_col: str,
    training_days: int,
    number_of_lags: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Prepare training data for forecasting models.

    This function encapsulates the data preparation logic used by both the XGBoost
    and regression forecasting methods. It creates features including:
    - Cyclical encoding for hour and day of week
    - Weather features
    - Lagged price features

    Returns:
        X_train: Training features
        y_train: Training target
        feature_cols: List of feature column names
    """
    required_cols = WEATHER_FEATURES + [price_col]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(
            f"Missing required columns in data: {', '.join([c for c in required_cols if c not in data.columns])}"
        )

    # Define training period
    training_start = current_time - pd.Timedelta(days=training_days)
    training_end = current_time

    historical_data = data.loc[:current_time].copy()

    # Create the lagged price features
    lag_cols = []
    lagged_data = []
    for lag in range(1, number_of_lags + 1):
        lag_col_name = f"lagged_price_{lag}"
        lag_cols.append(lag_col_name)
        lagged_data.append(historical_data[price_col].shift(lag).rename(lag_col_name))

    # Concatenate all lagged columns at once
    historical_data = pd.concat([historical_data] + lagged_data, axis=1)

    training_data = historical_data.loc[
        (historical_data.index >= training_start)
        & (historical_data.index < training_end)
    ].copy()

    # Apply cyclical encoding
    training_data, time_features_cols = add_cyclical_time_features(training_data)

    if verbose:
        plot_time_features(time_features_cols, training_data)

    # Define feature columns
    all_feature_cols = time_features_cols + WEATHER_FEATURES + lag_cols

    # Prepare training features and target
    X_train = training_data[all_feature_cols].values
    y_train = training_data[price_col].values

    # Remove rows with NaN values
    valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    if len(X_train) == 0:
        raise ValueError(
            "No valid training data after feature engineering and NaN removal."
        )

    return X_train, y_train, all_feature_cols
