import polars as pl
from ehr2meds.preMEDS.constants import (
    MANDATORY_COLUMNS,
    ROW_INDEX,
    SUBJECT_ID,
)
from typing import Dict


def add_row_idx(df: pl.DataFrame, start: int = 0) -> pl.DataFrame:
    """Add a stable, contiguous source-row index to a preMEDS chunk."""
    return df.with_columns(pl.int_range(start, start + df.height, dtype=pl.Int64).alias(ROW_INDEX))


def map_pids_to_ints(df: pl.DataFrame, subject_id_mapping: Dict[str, int]) -> pl.DataFrame:
    """Map string patient IDs to integers; keep only IDs that are in the mapping."""
    ids = df[SUBJECT_ID].cast(pl.String)
    mapped = ids.replace_strict(subject_id_mapping, default=None, return_dtype=pl.Int64)
    missing = ids.filter(mapped.is_null()).n_unique()
    if missing:
        print(f"Found {missing} subject IDs in the data that are not in the mapping. These IDs will be dropped")
    return df.with_columns(mapped.alias(SUBJECT_ID)).drop_nulls(SUBJECT_ID)


class SeenRows:
    """Fingerprints of the rows kept so far from one table, so duplicates in later chunks are dropped.

    Two 64-bit hashes per row (16 bytes) instead of the rows themselves; with 128 bits an accidental
    match between different rows is practically impossible.
    """

    KEYS = ("__seen_hash_0", "__seen_hash_1")

    def __init__(self):
        self.fingerprints = None

    def drop_seen(self, df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        row = pl.struct(columns)
        df = df.with_columns(row.hash(seed=0).alias(self.KEYS[0]), row.hash(seed=1).alias(self.KEYS[1]))
        if self.fingerprints is not None:
            df = df.join(self.fingerprints, on=self.KEYS, how="anti", maintain_order="left")
        new = df.select(self.KEYS)
        self.fingerprints = new if self.fingerprints is None else pl.concat([self.fingerprints, new])
        return df.drop(self.KEYS)


def clean_data(df: pl.DataFrame, tracker, seen: SeenRows | None = None) -> pl.DataFrame:
    """Remove rows with missing mandatory values and duplicate records.

    ``seen`` carries rows kept from earlier chunks of the same table, so duplicates are removed across chunks.
    """
    if all(col in df.columns for col in MANDATORY_COLUMNS):
        before = df.height
        df = df.drop_nulls(MANDATORY_COLUMNS)
        tracker.record_change("Mandatory-column filtering", df.height - before)

    # row_idx is always unique, so don't consider that column
    columns_to_check = [col for col in df.columns if col != ROW_INDEX]

    before = df.height
    df = df.unique(subset=columns_to_check, keep="first", maintain_order=True)
    if seen is not None:
        df = seen.drop_seen(df, columns_to_check)
    tracker.record_change("Deduplication", df.height - before)

    return df


def as_number(df: pl.DataFrame, col: str) -> pl.Expr:
    """Parse a column as Float64; unparseable values become null (like pd.to_numeric(errors="coerce"))."""
    dtype = df.schema[col]
    if dtype == pl.String:
        return pl.col(col).str.strip_chars().cast(pl.Float64, strict=False)
    if dtype.is_numeric():
        return pl.col(col).cast(pl.Float64)
    return pl.lit(None, pl.Float64)


def apply_value_map(df: pl.DataFrame, value_map_cfg: dict) -> pl.DataFrame:
    """Replace specific column values; other values are left unchanged"""
    for col, mapping in value_map_cfg.items():
        if col not in df.columns:
            continue

        numeric = as_number(df, col)
        integer_like = numeric % 1 == 0
        has_int_keys = any(isinstance(key, int) for key in mapping)

        # A column that gets string codes (or already holds strings) is a string column;
        # e.g. a mix of 5.0 and ALCC01 must be stored as one type.
        as_string = df.schema[col] == pl.String or any(isinstance(value, str) for value in mapping.values())
        if as_string:
            out_dtype = pl.String
            base = pl.col(col).cast(pl.String)
            if has_int_keys:
                # 5, 5.0 and "5.0" all map like the integer key 5, and unmapped ones read "5"
                base = pl.when(integer_like).then(numeric.cast(pl.Int64).cast(pl.String)).otherwise(base)
        else:
            out_dtype = df.schema[col]
            base = pl.col(col)

        out = base
        for key, value in mapping.items():
            if isinstance(key, int):
                hit = numeric == key
            elif as_string:
                hit = base == str(key)
            else:
                continue
            out = pl.when(hit).then(pl.lit(value).cast(out_dtype)).otherwise(out)
        df = df.with_columns(out.alias(col))
    return df


def normalize_integer_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Convert integer-like values to nullable integers.

    Values such as ``0``, ``0.0``, and ``"0.0"`` become the integer ``0``.
    Nonnumeric and non-integral values become null.
    """
    for col in columns:
        if col not in df.columns:
            continue
        if df.schema[col].is_integer():
            df = df.with_columns(pl.col(col).cast(pl.Int64))
            continue
        numeric = as_number(df, col)
        df = df.with_columns(pl.when(numeric % 1 == 0).then(numeric.cast(pl.Int64, strict=False)).alias(col))
    return df


def normalize_code_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Strip and uppercase string values."""
    return df.with_columns(pl.col(pl.String).str.strip_chars().str.to_uppercase())


def validate_subject_id(df: pl.DataFrame) -> None:
    """Checks that the subject_id column exists and is an integer"""
    if SUBJECT_ID not in df.columns:
        raise ValueError(f"Missing required column: {SUBJECT_ID}")
    if not df.schema[SUBJECT_ID].is_integer():
        raise ValueError(
            f"{SUBJECT_ID} column must be of integer type\n\
                Hint: Use the subject_id_mapping configuration to map string IDs to integers."
        )


def remove_timezones(df: pl.DataFrame) -> pl.DataFrame:
    """Convert timezone-aware datetime columns to timezone-naive UTC."""
    tz_columns = [col for col, dtype in df.schema.items() if isinstance(dtype, pl.Datetime) and dtype.time_zone]
    return df.with_columns(pl.col(tz_columns).dt.convert_time_zone("UTC").dt.replace_time_zone(None))
