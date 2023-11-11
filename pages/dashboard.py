import pandas as pd
import streamlit as st

from services import prophet, random_forest_regressor
from services.utils import read_dataset


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

        pred, prophet_model = prophet.fit_predict_prophet_model(mmm_df)
        df_with_prophet = prophet.extract_prophet_data(
            pred, mmm_df, target_prophet_cols=["trend", "yearly"]
        )
        rmse, r2_score = plot_random_forest_predict(df_with_prophet)
        # print(rmse)
        # print(r2_score)


def plot_cost_and_revenue(
    df: pd.DataFrame,
    date_column: str,
    revenue_columns: list[str],
    cost_columns: list[str],
):
    st.line_chart(data=df, x=date_column, y=revenue_columns)
    st.line_chart(data=df, x=date_column, y=cost_columns)


def plot_random_forest_predict(
    df_with_prophet: pd.DataFrame, target_column="sales"
) -> (float, float):
    rf_model, pred = random_forest_regressor.regressor(
        df_with_prophet, target_column, list(df_with_prophet.columns)[2:]
    )

    rmse, r2_score = random_forest_regressor.calc_rmse_r2(
        df_with_prophet[target_column], pred
    )
    return rmse, r2_score
