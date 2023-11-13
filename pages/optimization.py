import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor

from services import draw_response_curve, future_predict


def optimize(
    cost_columns: list[str],
    shap_df: pd.DataFrame,
    df_with_prophet: pd.DataFrame,
    prophet_model: Prophet,
    rf_model: RandomForestRegressor,
    mmm_df: pd.DataFrame,
):
    PROPHET_COLUMNS = ["trend", "yearly"]
    FEATURE_COLUMNS = cost_columns + PROPHET_COLUMNS

    for col in cost_columns:
        st.pyplot(draw_response_curve.response_curve(shap_df, df_with_prophet, col))

    show_mean_spend(df_with_prophet, cost_columns)

    forecast = future_predict.get_prophet_forecast(prophet_model)

    if "optimize_params" not in st.session_state:
        st.session_state["optimize_params"] = {
            "tvcm": 1.25,
            "web": 1.25,
            "newspaper": 0.8,
        }

    st.session_state["optimize_params"]["tvcm"] = st.number_input(
        "tvcmのコストを入力してください", value=st.session_state["optimize_params"]["tvcm"]
    )
    st.session_state["optimize_params"]["web"] = st.number_input(
        "webのコストを入力してください", value=st.session_state["optimize_params"]["web"]
    )
    st.session_state["optimize_params"]["newspaper"] = st.number_input(
        "newspaperのコストを入力してください",
        value=st.session_state["optimize_params"]["newspaper"],
    )

    current_df_for_pred = future_predict.get_current_df_with_future_prophet(
        forecast[PROPHET_COLUMNS], mmm_df, 30
    )
    optimized_df_for_pred = future_predict.get_optimized_df_with_prophet(
        forecast[PROPHET_COLUMNS], mmm_df, 30, st.session_state["optimize_params"]
    )

    st.pyplot(
        future_prediction(
            FEATURE_COLUMNS, rf_model, current_df_for_pred, optimized_df_for_pred
        )
    )


def show_mean_spend(cost_df: pd.DataFrame, features: list[str]):
    st.markdown("#### 平均コスト/週")
    for feature in features:
        mean_spend = cost_df[feature].mean().astype("int")
        st.write(f"{feature}:{mean_spend}")


def future_prediction(
    cost_columns: list[str],
    rf_model: RandomForestRegressor,
    current_df,
    optimized_df,
) -> plt.figure:
    current_pred_df = future_predict.predict(current_df, cost_columns, rf_model)
    optimized_pred_df = future_predict.predict(optimized_df, cost_columns, rf_model)

    return future_predict.plot_prediction(current_pred_df, optimized_pred_df)
