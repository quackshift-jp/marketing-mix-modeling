import pandas as pd
import streamlit as st

from services import prophet
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
        # st.table(df_with_prophet)


def plot_cost_and_revenue(
    df: pd.DataFrame,
    date_column: str,
    revenue_columns: list[str],
    cost_columns: list[str],
):
    st.line_chart(data=df, x=date_column, y=revenue_columns)
    st.line_chart(data=df, x=date_column, y=cost_columns)
