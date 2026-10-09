import pandas as pd
from ehr2meds.preMEDS.constants import (
    MANDATORY_COLUMNS,
    ROW_INDEX,
    SUBJECT_ID,
)
from typing import Dict


def add_row_idx(df: pd.DataFrame, start: int = 0) -> pd.DataFrame:
    """Add a stable, contiguous source-row index to a preMEDS chunk."""
    df[ROW_INDEX] = range(start, start + len(df))
    return df


def check_columns(df: pd.DataFrame, columns_map: dict):
    """Check if all columns in columns_map are present in df."""
    missing_columns = set(columns_map.keys()) - set(df.columns)
    if missing_columns:
        available_columns = pd.DataFrame({"Available Columns": sorted(df.columns)})
        requested_columns = pd.DataFrame({"Requested Columns": sorted(columns_map.keys())})
        error_msg = f"\nMissing columns: {sorted(missing_columns)}\n\n"
        error_msg += "Columns comparison:\n"
        error_msg += f"{pd.concat([available_columns, requested_columns], axis=1).to_string()}"
        raise ValueError(error_msg)


def map_pids_to_ints(df: pd.DataFrame, subject_id_mapping: Dict[str, int]) -> pd.DataFrame:
    """Map string patient IDs to integers; keep only IDs that are in the mapping."""
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(object).astype(str)

    df[SUBJECT_ID] = df[SUBJECT_ID].map(subject_id_mapping)
    if df[SUBJECT_ID].isna().any():
        missing_ids = df[SUBJECT_ID][df[SUBJECT_ID].isna()].unique()
        print(f"Found {len(missing_ids)} subject IDs in the data that are not in the mapping. These IDs will be dropped")
    df = df.dropna(subset=[SUBJECT_ID], how="any")
    df[SUBJECT_ID] = df[SUBJECT_ID].astype(int)
    return df


def clean_data(df: pd.DataFrame, tracker=None) -> pd.DataFrame:
    """Remove rows with missing mandatory values and duplicate records."""
    # Clean data
    if all(col in df.columns for col in MANDATORY_COLUMNS):
        before = len(df)
        df = df.dropna(subset=MANDATORY_COLUMNS, how="any")

        if tracker is not None:
            tracker.record_change(
                "Mandatory-column filtering",
                len(df) - before,
            )
    # row_idx is always unique, so don't consider that column
    columns_to_check = [col for col in df.columns if col != ROW_INDEX]

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=columns_to_check)

    if tracker is not None:
        tracker.record_change(
            "Deduplication",
            len(df) - before,
        )

    return df

def apply_value_map(df: pd.DataFrame, value_map_cfg: dict) -> pd.DataFrame:
    """Replace specific column values; other values are left unchanged"""
    for col, mapping in value_map_cfg.items():
        if col not in df.columns:
            continue

        # Numeric columns cannot hold string replacements. Object dtype permits
        # those replacements without changing the existing values.
        maps_to_string = any(isinstance(value, str) for value in mapping.values())
        if maps_to_string and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object)

        # need to handle integer-like values, so they map cleanly
        # e.g., 5.0 and "5.0" should map to the same value as 5
        numeric = pd.to_numeric(df[col], errors="coerce")
        integer_like = numeric.notna() & (numeric % 1 == 0)
        if any(isinstance(key, int) for key in mapping):
            canonical_integers = numeric.loc[integer_like].astype("Int64")
            if maps_to_string:
                canonical_integers = canonical_integers.astype("string")
            df.loc[integer_like, col] = canonical_integers

        for key, value in mapping.items():
            if isinstance(key, int):
                df.loc[numeric.eq(key), col] = value

        df.replace({col: mapping}, inplace=True)
        # Some columns contain both numerics and strings (e.g., 5.0 and ALCC01).
        # Arrow cannot serialize that mixture consistently.
        # If a mapping introduces string codes,
        # represent every non-null value as a string;
        # unfamiliar values retain their literal value instead of being guessed.
        if maps_to_string:
            df[col] = df[col].astype("string")
    return df


def normalize_integer_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert integer-like values to nullable integers.

    Values such as ``0``, ``0.0``, and ``"0.0"`` become the integer ``0``.
    Nonnumeric and non-integral values become null.
    """
    for col in columns:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        integers = numeric.where(numeric % 1 == 0)
        df[col] = integers.astype("Int64")
    return df


def normalize_code_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and uppercase string values without changing other object values."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        is_string = df[col].map(lambda value: isinstance(value, str))
        df.loc[is_string, col] = df.loc[is_string, col].str.strip().str.upper()
    return df


def validate_subject_id(df: pd.DataFrame) -> None:
    """Checks that the subject_id column exists and is an integer"""
    if SUBJECT_ID not in df.columns:
        raise ValueError(f"Missing required column: {SUBJECT_ID}")
    if not pd.api.types.is_integer_dtype(df[SUBJECT_ID]):
        raise ValueError(
            f"{SUBJECT_ID} column must be of integer type\n\
                Hint: Use the subject_id_mapping configuration to map string IDs to integers."
        )


def remove_timezones(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timezone-aware datetime columns to timezone-naive UTC."""
    for col in df.select_dtypes(include=["datetimetz"]).columns:
        df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)

    return df
