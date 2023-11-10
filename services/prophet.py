import pandas as pd
from prophet import Prophet


def rename_columns_for_prophet(
    df: pd.DataFrame, date_column: str, sale_column: str
) -> pd.DataFrame:
    """
    prophetの仕様で、dsとyのカラム名を指定する必要がある
    """
    df_copy = df.copy()
    return df_copy.rename(columns={date_column: "ds", sale_column: "y"})


def fit_predict_prophet_model(df: pd.DataFrame) -> (pd.DataFrame, Prophet):
    """Prophetモデルを作成し、fitとpredictを実行する
    prophet_model.add_regressorをした場合は、そのカラムをfitとpredictに追加する
    Prophetの詳細はGitHubを参照[https://facebook.github.io/prophet/docs/quick_start.html]
    """
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )

    df = rename_columns_for_prophet(df, "date", "sales")

    prophet_model.fit(df[["ds", "y"]])
    pred = prophet_model.predict(df[["ds", "y"]])
    return pred, prophet_model


def extract_prophet_data(
    pred: pd.DataFrame, df: pd.DataFrame, target_prophet_cols: list[str]
) -> pd.DataFrame:
    """
    日別売上・コストが入ったデータフレームに、prophetデータを加える
    args:
        pred:
            Prophetで予測したデータ
        df:
            日別売上・コストが入ったデータフレーム
    """
    df_with_prophet = df.copy()
    for col in [col for col in pred.columns if col in target_prophet_cols]:
        df_with_prophet[col] = pred[col]
    return df_with_prophet
