

import pandas as pd


def drop_empty_columns(df, columns: list[str]):
    """Drop columns that are empty."""
    return df.dropna(subset=columns)
