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


def coalesce_first(
    expanded_table: pd.DataFrame, name: str, components: list[str]
) -> pd.DataFrame:
    """First non-null across ``components`` (left-to-right), for numeric or other dtypes."""
    result = expanded_table.copy()
    acc = expanded_table[components[0]]
    for c in components[1:]:
        acc = acc.combine_first(expanded_table[c])
    result[name] = acc
    return result
