import polars as pl
from ehr2meds.preMEDS.timestamps import parse_date
from ehr2meds.preMEDS.utils import as_number


def apply_mapping(
    df: pl.DataFrame,
    map_table: pl.DataFrame,
    join_col: str,
    source_col: str,
    target_columns: dict[str, str | None],
    how: str = "inner",
    drop_source: bool = False,
) -> pl.DataFrame:
    """Join one or more columns from a mapping table onto a dataframe.

    ``target_columns`` maps one or more source column names to their output names.
    A null output name preserves the source column name.
    """
    if not target_columns:
        raise ValueError("target_columns must contain at least one column")

    output_columns = [output if output is not None else source for source, output in target_columns.items()]
    if len(output_columns) != len(set(output_columns)):
        raise ValueError("Mapped columns must have unique output names")

    map_table = map_table.select(join_col, *target_columns)
    if df.schema[source_col] != map_table.schema[join_col]:
        mapping_as_numeric = map_table.select(as_number(map_table, join_col))[join_col]
        if mapping_as_numeric.null_count() == map_table[join_col].null_count():
            # Match equivalent keys such as 5, 5.0, and "5.0".
            df = df.with_columns(as_number(df, source_col))
            map_table = map_table.with_columns(mapping_as_numeric)
        else:
            df = df.with_columns(pl.col(source_col).cast(pl.String))
            map_table = map_table.with_columns(pl.col(join_col).cast(pl.String))

    # Null keys never match, so rows without a key get null targets (left) or are dropped (inner).
    df = df.join(map_table, left_on=source_col, right_on=join_col, how=how, maintain_order="left")

    if drop_source:
        df = df.drop(source_col)

    rename_columns = {source: output for source, output in target_columns.items() if output is not None}
    return df.rename(rename_columns)


def normalize_sor_id(values: pl.Expr) -> pl.Expr:
    """Normalize SOR identifiers read from CSV and spreadsheet exports."""
    values = values.cast(pl.String).str.strip_chars()
    values = values.str.replace(r'^="(.*)"$', "$1")
    values = values.str.replace(r"\.0$", "")
    return pl.when(values != "").then(values)


def add_sor_attributes(df: pl.DataFrame, sor_table: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Map each SOR unit and date to its region and primary specialty."""
    keys = [config["source_column"], config["source_date_column"]]

    contact_keys = (
        df.select(keys)
        .unique()
        .with_columns(
            normalize_sor_id(pl.col(keys[0])).alias("normalized_sor_id"),
            # Same parser as the timestamp columns, so a contact's SOR date matches its timestamp date.
            parse_date(df, keys[1], config.get("date_format")).alias("contact_date"),
        )
    )
    candidates = contact_keys.join(
        sor_table.select(
            pl.col(config["join_on"]).alias("normalized_sor_id"), "valid_from", "valid_to", "region", "primary_specialty"
        ),
        on="normalized_sor_id",
    )
    date_valid_matches = (
        candidates.filter(
            pl.col("contact_date").is_not_null(),
            pl.col("valid_from").is_null() | (pl.col("contact_date") >= pl.col("valid_from")),
            pl.col("valid_to").is_null() | (pl.col("contact_date") <= pl.col("valid_to")),
        )
        .sort("valid_from", descending=True, nulls_last=True)
        .unique(subset=keys, keep="first")
        .select(*keys, "region", pl.col("primary_specialty").alias("specialty"))
    )
    return df.drop("region", "specialty", strict=False).join(
        date_valid_matches, on=keys, how="left", validate="m:1", maintain_order="left"
    )


MAPPING_STRATEGIES = {
    "sor": (add_sor_attributes, ("valid_from", "valid_to", "region", "primary_specialty")),
}
