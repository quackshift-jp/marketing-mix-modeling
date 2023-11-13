import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

from pages import optimization
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


def display(upload_file):
    try:
        mmm_df = read_dataset.read_data(upload_file, "Date")
    except ValueError:
        st.error("データの日付は「Dateまたはdate」、売上は「Salesまたはsales」にしてください。")

    revenue_columns = st.selectbox("売上カラムを選択してください", mmm_df.columns)
    cost_columns = st.multiselect("説明変数を選択してください（複数選択可）", mmm_df.columns)

    st.subheader("売上とコストの可視化", divider="rainbow")
    st.line_chart(data=mmm_df, x="date", y=revenue_columns)
    st.line_chart(data=mmm_df, x="date", y=cost_columns)

    execute_ramdom_forest = st.button("ランダムフォレストで予測する", key=1)
    if execute_ramdom_forest:
        show_progress_bar()
        pred, prophet_model = prophet.fit_predict_prophet_model(mmm_df)
        df_with_prophet = prophet.extract_prophet_data(
            pred, mmm_df, target_prophet_cols=["trend", "yearly"]
        )
        shap_df, rf_model = random_forest_predict(df_with_prophet, FEATURE_COLUMNS)

        st.subheader("各チャネルの売上貢献度を見る", divider="rainbow")

        feature_importance = shap_feature_importance.extract_spend_effect_share(
            shap_df, SALES_COLUMNS, df_with_prophet
        )
        st.pyplot(shap_feature_importance.plot_roi(feature_importance))
        st.pyplot(shap_feature_importance.plot_spend_effect_share(feature_importance))

        optimization.optimize(
            cost_columns,
            shap_df,
            df_with_prophet,
            prophet_model,
            rf_model,
            mmm_df,
        )


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

    st.markdown("#### 予測精度(r2_score)")
    st.write(r2_score)
    return pd.DataFrame(shap_values, columns=feature_columns), rf_model


def show_progress_bar() -> None:
    my_bar = st.progress(0)
    st.write("In Progress...")
    for parcent in range(100):
        time.sleep(0.01)
        my_bar.progress(parcent + 1)
    st.write("Complete🎉")
