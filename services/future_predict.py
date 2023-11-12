import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor


def get_prophet_forecast(prophet_model: Prophet) -> pd.DataFrame:
    future = prophet_model.make_future_dataframe(periods=30)
    prophet_forecast = prophet_model.predict(future)
    return prophet_forecast


def get_current_df_with_future_prophet(
    prophet_forecast: pd.DataFrame, cost_df: pd.DataFrame, period: int
) -> pd.DataFrame:
    """
    現状のコストデータに加えて、{period}分の未来コストデータを加える.
    """
    pred_df = cost_df.tail(period)
    pred_df["date"] = pred_df["date"] + pd.DateOffset(weeks=period)

    pred_df = pd.concat([cost_df, pred_df], axis=0)

    current_df = pred_df.merge(
        prophet_forecast,
        left_index=True,
        right_index=True,
        how="left",
    )
    return current_df.sort_values(by="date", ascending=True).tail(period)


def get_optimized_df_with_prophet(
    prophet_forecast: pd.DataFrame,
    cost_df: pd.DataFrame,
    period: int,
    optimize_param: dict[str, float],
) -> pd.DataFrame:
    optimized_df = get_current_df_with_future_prophet(prophet_forecast, cost_df, period)

    for key in optimize_param.keys():
        if key in optimized_df.columns:
            optimized_df.iloc[
                -period:, optimized_df.columns.get_loc(key)
            ] *= optimize_param[key]
    return optimized_df.sort_values(by="date", ascending=True).tail(period)


def predict(
    df_for_pred: pd.DataFrame,
    features: list[str],
    rf_model: RandomForestRegressor,
) -> pd.DataFrame:
    pred = rf_model.predict(df_for_pred[features])
    pred_df = pd.DataFrame({"date": df_for_pred["date"], "pred": pred})
    return pred_df


def plot_prediction(
    current_pred_df: pd.DataFrame,
    optimized_pred_df: pd.DataFrame,
) -> plt.figure:
    sale_before_optimize = current_pred_df["pred"].sum()
    sale_after_optimize = optimized_pred_df["pred"].sum()

    fig, ax = plt.subplots(figsize=(14, 7))
    plt.ticklabel_format(style="plain", axis="y")
    ax.plot(
        current_pred_df["date"],
        current_pred_df["pred"],
        color="green",
        label="current",
    )
    ax.plot(
        optimized_pred_df["date"],
        optimized_pred_df["pred"],
        color="blue",
        label="optimized",
    )
    plt.title(f"before : {sale_before_optimize} VS after : {sale_after_optimize}")
    plt.legend()
    return fig
