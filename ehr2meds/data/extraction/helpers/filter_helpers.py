import pandas as pd


def drop_empty_columns(df, columns: list[str]):
    """Drop columns that are empty."""
    return df.dropna(subset=columns)


def bool_match(df, target_col: str, op: str, val):
    try:
        val_num = float(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"val must be numeric; got {val!r}") from e

    s = pd.to_numeric(df[target_col], errors="coerce")
    if op == "==":
        mask = s == val_num
    elif op == "!=":
        mask = s != val_num
    elif op == ">=":
        mask = s >= val_num
    elif op == "<=":
        mask = s <= val_num
    elif op == ">":
        mask = s > val_num
    elif op == "<":
        mask = s < val_num
    else:
        raise ValueError(
            f"Unsupported operator {op!r}; expected one of ==, !=, >=, <=, >, <"
        )

    return df.loc[mask].copy()