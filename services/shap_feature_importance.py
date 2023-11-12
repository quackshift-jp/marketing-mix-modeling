import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor


def extract_shap_value(
    rf_model: RandomForestRegressor, features: list[str]
) -> np.ndarray:
    """
    モデルのSHAP値を算出する
    SHAP値とは
      「その特徴量が無い時、目的変数がどのように変化するか」を貢献度とする
      協力ゲーム理論を応用した特徴量貢献度算出ロジックで、シンプルな線形変換することによって算出を行う
      （ざっくりとしたSHAP値の算出ロジック説明）
      1. すべての可能な特徴の組み合わせを作成
      2. モデルの平均的な予測値を算出
      3. 各組み合わせにおいて、ある特徴量を含まないモデルの予測値と平均予測値との差を計算
      3. 各組み合わせにおいて、ある特徴量を含まないモデルの予測値と平均予測値との差を計算
      4. 各組み合わせにおいて、ある特徴量がモデルの予測値を平均値からどれだけ変化したかを計算し、特徴量とする
    args:
      rf_model: 訓練済みのRandomForestモデル
      features: 訓練に使用した特徴量カラムリスト
    """
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X=features)
    return shap_values


def extract_spend_effect_share(
    shap_df: pd.DataFrame, features: list, cost_df: pd.DataFrame
) -> pd.DataFrame:
    """チャネルの貢献度を可視化
    各チャネルの消費割合と、売上貢献割合を算出する

    args:
      shap_df : SHAP値の入ったデータフレーム
      channels: 貢献度を可視化するチャネルをリスト指定
      cost_df: コストデータの結果の入ったMMM用データ
    return:
      spend_effect_share : 消費割合、売上貢献割合の入ったデータフレーム
    """
    responses = pd.DataFrame(
        shap_df[features].abs().sum(axis=0), columns=["effect_share"]
    )
    response_percentages = responses / responses.sum()

    spends = pd.DataFrame(cost_df[features].sum(axis=0), columns=["spend_share"])
    spends_percentages = spends / spends.sum()

    spend_effect_share = pd.merge(
        response_percentages, spends_percentages, left_index=True, right_index=True
    )
    spend_effect_share = spend_effect_share.reset_index().rename(
        columns={"index": "media"}
    )

    return spend_effect_share


def plot_roi(spend_effect_share: pd.DataFrame) -> plt.figure:
    df_roi = calc_shap_roi(spend_effect_share)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_roi["media"], df_roi["roi"])
    ax.set_xlabel("Media")
    ax.set_ylabel("ROI")
    for i, value in enumerate(df_roi["roi"]):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
    return fig


def calc_shap_roi(spend_effect_share: pd.DataFrame) -> pd.DataFrame:
    roi = spend_effect_share["effect_share"] / spend_effect_share["spend_share"]
    df_roi = pd.DataFrame({"media": spend_effect_share["media"], "roi": roi})
    return df_roi


def plot_spend_effect_share(spend_effect_share: pd.DataFrame) -> plt.figure:
    x = np.array(spend_effect_share["media"].unique())
    x_position = np.arange(len(x))

    y_control = np.array(spend_effect_share["spend_share"])
    y_stress = np.array(spend_effect_share["effect_share"])

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x_position, y_control, width=0.4, label="spend share", color="lightblue")
    ax.bar(x_position + 0.4, y_stress, width=0.4, label="effect share", color="blue")
    ax.legend()
    ax.set_xticks(x_position + 0.2)
    ax.set_xticklabels(x)
    return fig
