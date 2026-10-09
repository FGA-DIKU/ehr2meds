import pandas as pd
from ehr2meds.preMEDS.constants import ROW_INDEX

def apply_mapping(
    df: pd.DataFrame,
    map_table: pd.DataFrame,
    join_col: str,
    source_col: str,
    target_columns: dict[str, str | None],
    how: str = "inner",
    drop_source: bool = False,
) -> pd.DataFrame:
    """Join one or more columns from a mapping table onto a dataframe.

    ``target_columns`` maps one or more source column names to their output names.
    A null output name preserves the source column name.
    """
    if not target_columns:
        raise ValueError("target_columns must contain at least one column")

    output_columns = [output if output is not None else source for source, output in target_columns.items()]
    if len(output_columns) != len(set(output_columns)):
        raise ValueError("Mapped columns must have unique output names")

    if df[source_col].dtype != map_table[join_col].dtype:
        source_as_numeric = pd.to_numeric(df[source_col], errors="coerce")
        mapping_as_numeric = pd.to_numeric(map_table[join_col], errors="coerce")
        mapping_keys_are_numeric = mapping_as_numeric.notna().equals(map_table[join_col].notna())
        if mapping_keys_are_numeric:
            # Match equivalent keys such as 5, 5.0, and "5.0".
            df[source_col] = source_as_numeric
            map_table[join_col] = mapping_as_numeric
        else:
            df[source_col] = df[source_col].astype(str)
            map_table[join_col] = map_table[join_col].astype(str)

    df = pd.merge(
        df,
        map_table[[join_col, *target_columns]],
        left_on=source_col,
        right_on=join_col,
        how=how,
    )

    if join_col != source_col:
        df = df.drop(columns=[join_col])
    if drop_source:
        df = df.drop(columns=[source_col])

    rename_columns = {source: output for source, output in target_columns.items() if output is not None}
    return df.rename(columns=rename_columns) if rename_columns else df


def normalize_sor_id(values: pd.Series) -> pd.Series:
    """Normalize SOR identifiers read from CSV and spreadsheet exports."""
    values = values.astype("string").str.strip()
    values = values.str.replace(r'^="(.*)"$', r"\1", regex=True)
    return values.str.replace(r"\.0$", "", regex=True).replace("", pd.NA)


def add_sor_attributes(df: pd.DataFrame, sor_table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Map each SOR unit and date to its region and primary specialty."""
    source_id_column = config["source_column"]
    source_date_column = config["source_date_column"]
    mapping_id_column = config["join_on"]
    sor_table = sor_table.rename(columns={mapping_id_column: "mapping_sor_id"})

    contact_keys = df[[source_id_column, source_date_column]].drop_duplicates().copy()
    contact_keys["normalized_sor_id"] = normalize_sor_id(contact_keys[source_id_column])
    contact_keys["contact_date"] = pd.to_datetime(contact_keys[source_date_column],format=config.get("date_format"),errors="coerce").dt.normalize()
    candidates = contact_keys.merge(
        sor_table,
        left_on="normalized_sor_id",
        right_on="mapping_sor_id",
        how="left",
    )
    is_date_valid = (
        candidates["mapping_sor_id"].notna()
        & candidates["contact_date"].notna()
        & (candidates["valid_from"].isna() | candidates["contact_date"].ge(candidates["valid_from"]))
        & (candidates["valid_to"].isna() | candidates["contact_date"].le(candidates["valid_to"]))
    )
    date_valid_matches = (
        candidates.loc[is_date_valid]
        .sort_values("valid_from", ascending=False, na_position="last")
        .drop_duplicates([source_id_column, source_date_column], keep="first")
    )
    attributes = contact_keys[[source_id_column, source_date_column]].merge(
        date_valid_matches[[source_id_column, source_date_column, "region", "primary_specialty"]],
        on=[source_id_column, source_date_column],
        how="left",
        validate="one_to_one",
    )
    attributes = attributes.rename(columns={"primary_specialty": "specialty"})
    result = df.drop(columns=["region", "specialty"], errors="ignore").merge(
        attributes,
        on=[source_id_column, source_date_column],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    return result.sort_values(ROW_INDEX, kind="stable") if ROW_INDEX in result else result


MAPPING_STRATEGIES = {
    "sor": (add_sor_attributes, ("valid_from", "valid_to", "region", "primary_specialty")),
}
