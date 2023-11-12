import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def response_curve(
    df_shap_values: pd.DataFrame,
    cost_df: pd.DataFrame,
    feature: str,
) -> plt.figure:
    """
    各チャネルのレスポンスカーブをregplotで回帰モデルに可視化する
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ticklabel_format(style="plain", axis="y")
    plt.ticklabel_format(style="plain", axis="x")
    sns.regplot(
        x=cost_df[feature],
        y=df_shap_values[feature],
        label=feature,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "C2", "linewidth": 3},
        lowess=True,
        ax=ax,
    ).set(title=f"{feature}: Spend vs Shapley")
    ax.axhline(0, linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel(f"{feature} spend")
    ax.set_ylabel(f"SHAP Value for {feature}")

    return fig
