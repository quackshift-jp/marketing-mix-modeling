import pandas as pd
import streamlit as st

from services.utils import read_dataset


def display():
    st.header("Upload Dataset")

    upload_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if upload_file:
        try:
            df = read_dataset.read_data(upload_file, "Date")

            plot_cost_and_revenue(df, "date", ["sales"], df.columns[2:-1])
        except ValueError:
            st.error("データの日付は「Dateまたはdate」、売上は「Salesまたはsales」にしてください。")


def plot_cost_and_revenue(
    df: pd.DataFrame,
    date_column: str,
    revenue_columns: list[str],
    cost_columns: list[str],
):
    st.line_chart(data=df, x=date_column, y=revenue_columns)
    st.line_chart(data=df, x=date_column, y=cost_columns)
