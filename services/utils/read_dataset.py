import pandas as pd


def read_data(file_name: str, date_column_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name, parse_dates=[date_column_name])
    df.columns = [col.lower() for col in df.columns]
    return df
