import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from services import (
    draw_response_curve,
    future_predict,
    prophet,
    random_forest_regressor,
    shap_feature_importance,
)
from services.utils import read_dataset

SALES_COLUMNS = ["tvcm", "newspaper", "web"]
PROPHET_COLUMNS = ["trend", "yearly"]
FEATURE_COLUMNS = SALES_COLUMNS + PROPHET_COLUMNS


OPTIMIZE_PARAMS = {
    "tvcm": 1.25,
    "web": 1.25,
    "newspaper": 0.8,
}


def display():
    st.header("Upload Dataset")

    upload_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if upload_file:
        try:
            mmm_df = read_dataset.read_data(upload_file, "Date")
            st.subheader("売上とコストの可視化", divider="rainbow")
            plot_cost_and_revenue(mmm_df, "date", ["sales"], mmm_df.columns[2:])
        except ValueError:
            st.error("データの日付は「Dateまたはdate」、売上は「Salesまたはsales」にしてください。")

        st.subheader("コストに対する売上貢献度を見る", divider="rainbow")
        pred, prophet_model = prophet.fit_predict_prophet_model(mmm_df)
        df_with_prophet = prophet.extract_prophet_data(
            pred, mmm_df, target_prophet_cols=["trend", "yearly"]
        )
        shap_df, rf_model = random_forest_predict(df_with_prophet, FEATURE_COLUMNS)

        feature_importance = shap_feature_importance.extract_spend_effect_share(
            shap_df, SALES_COLUMNS, df_with_prophet
        )
        st.pyplot(shap_feature_importance.plot_roi(feature_importance))
        st.pyplot(shap_feature_importance.plot_spend_effect_share(feature_importance))

        for feature in SALES_COLUMNS:
            st.pyplot(
                draw_response_curve.response_curve(shap_df, df_with_prophet, feature)
            )

        calc_mean_spend(df_with_prophet, SALES_COLUMNS)

        st.pyplot(future_prediction(prophet_model, mmm_df, FEATURE_COLUMNS, rf_model))


# TODO:下の関数は、servicesディレクトリに移動させたい
def plot_cost_and_revenue(
    df: pd.DataFrame,
    date_column: str,
    revenue_columns: list[str],
    cost_columns: list[str],
):
    st.line_chart(data=df, x=date_column, y=revenue_columns)
    st.line_chart(data=df, x=date_column, y=cost_columns)


def calc_mean_spend(cost_df: pd.DataFrame, features: list[str]):
    st.write("平均コスト")
    for feature in features:
        mean_spend = cost_df[feature].mean().astype("int")
        st.write(f"{feature}:{mean_spend}/週")


def random_forest_predict(
    df_with_prophet: pd.DataFrame, feature_columns: list[str], target_column="sales"
) -> (pd.DataFrame, RandomForestRegressor):
    rf_model, pred = random_forest_regressor.regressor(
        df_with_prophet, target_column, list(df_with_prophet.columns)[2:]
    )

    rmse, r2_score = random_forest_regressor.calc_rmse_r2(
        df_with_prophet[target_column], pred
    )

    shap_values = shap_feature_importance.extract_shap_value(
        rf_model, df_with_prophet[feature_columns]
    )

    st.write(f"予測精度(r2_score):{r2_score}")
    return pd.DataFrame(shap_values, columns=feature_columns), rf_model


def future_prediction(prophet_model, mmm_df, features, rf_model) -> plt.figure:
    forecast = future_predict.get_prophet_forecast(prophet_model)

    current_df = future_predict.get_current_df_with_future_prophet(
        forecast[PROPHET_COLUMNS], mmm_df, 30
    )
    optimized_df = future_predict.get_optimized_df_with_prophet(
        forecast[PROPHET_COLUMNS], mmm_df, 30, OPTIMIZE_PARAMS
    )

    current_pred_df = future_predict.predict(current_df, features, rf_model)
    optimized_pred_df = future_predict.predict(optimized_df, features, rf_model)

    return future_predict.plot_prediction(current_pred_df, optimized_pred_df)
