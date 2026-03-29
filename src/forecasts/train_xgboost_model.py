import pandas as pd
import xgboost as xgb

from src.forecasts.prepare_training_data import prepare_training_data


def train_xgboost_model(
    data: pd.DataFrame,
    current_time: pd.Timestamp,
    price_col: str,
    training_days: int,
    number_of_lags: int,
):
    """
    Train an XGBoost model for price forecasting.
    """

    # Prepare training data
    X_train, y_train, feature_cols = prepare_training_data(
        data=data,
        current_time=current_time,
        price_col=price_col,
        training_days=training_days,
        number_of_lags=number_of_lags,
    )

    # Initialize XGBoost Regressor
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        tree_method="hist",  # Faster training
    )

    # Fit the model
    model.fit(X_train, y_train)
    return model
