import pandas as pd
from prophet import Prophet


def get_prophet_future_data(prophet_model: Prophet) -> pd.DataFrame:
    future = prophet_model.make_future_dataframe(periods=30)
    prophet_future_data = prophet_model.predict(future)
    return prophet_future_data
