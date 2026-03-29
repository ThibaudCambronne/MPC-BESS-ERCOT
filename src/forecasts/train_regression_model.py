import pandas as pd
from sklearn.linear_model import LinearRegression

from src.forecasts.prepare_training_data import prepare_training_data


def train_regression_model(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    price_col: str,
    training_days: int,
    number_of_lags: int,
):
    """
    Train a linear regression model for price forecasting.
    """

    # Prepare training data
    X_train, y_train, feature_cols = prepare_training_data(
        data=data,
        current_time=current_time,
        price_col=price_col,
        training_days=training_days,
        number_of_lags=number_of_lags,
    )

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
