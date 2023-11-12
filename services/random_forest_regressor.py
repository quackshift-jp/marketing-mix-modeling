import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def make_random_forest_model(
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    ccp_alpha: float,
    bootstrap: bool = True,
    random_state: int = 123,
) -> RandomForestRegressor:
    """
    ランダムフォレストのモデルを生成する
    args:
      n_estimators :The number of trees in the forest
      max_depth: The maximum depth of the tree
      min_samples_split: The minimum number of samples required to split an internal node
      min_samples_leaf: The minimum number of samples required to be at a leaf node
      bootstrap: Whether bootstrap samples are used when building trees
      random_state: Controls the randomness
      ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning
    """
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
        "random_state": random_state,
        "ccp_alpha": ccp_alpha,
    }
    rf = RandomForestRegressor(**params)
    return rf


def fit_predict_rf_model(
    rf_model: RandomForestRegressor,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> (RandomForestRegressor, np.ndarray):
    rf_model.fit(x_train, y_train)
    pred = rf_model.predict(x_test)
    # Shapの計算に、RandomForestRegressorを用いるため、モデルをreturnする
    return (rf_model, pred)


def calc_rmse_r2(y_true, y_pred) -> (float, float):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2


def regressor(
    df: pd.DataFrame, target_column: str, feature_columns: list[str]
) -> (RandomForestRegressor, np.ndarray):
    tscv = TimeSeriesSplit(n_splits=3, test_size=10)
    for train_index, test_index in tscv.split(df):
        x_train = df.iloc[train_index][feature_columns]
        y_train = df[target_column].values[train_index]

    rf = make_random_forest_model(20, 9, 4, 8, 0.15)
    return fit_predict_rf_model(rf, x_train, y_train, df[feature_columns])
