import pandas as pd

def or_columns(expanded_table: pd.DataFrame, name: str, components: list[dict]) -> pd.DataFrame:
    """Boolean OR across component columns (NaN -> False)."""
    cols = components
    result = expanded_table.copy()
    result[name] = expanded_table[cols].fillna(False).astype(bool).any(axis=1).astype(bool)
    return result

def and_columns(expanded_table: pd.DataFrame, name: str, components: list[dict]) -> pd.DataFrame:
    """Boolean AND across component columns (NaN -> False)."""
    cols = components
    result = expanded_table.copy()
    result[name] = expanded_table[cols].fillna(False).astype(bool).all(axis=1).astype(bool)
    return result

def add_column(expanded_table: pd.DataFrame, name: str, components: list[dict]) -> pd.DataFrame:
    """Add component columns together."""
    cols = components
    result = expanded_table.copy()
    result[name] = expanded_table[cols]
    return result
