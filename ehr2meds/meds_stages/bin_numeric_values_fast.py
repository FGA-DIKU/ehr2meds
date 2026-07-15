"""Efficient drop-in replacement for the MEDS-Transforms `bin_numeric_values` stage.

Produces byte-for-byte identical output to the built-in stage (same bin indices, same
`{code}//value_[{left},{right})` / `{bin}` formatting, same null and out-of-range
handling, same `drop_numeric_value` behavior), but computes the bin index with a
per-code `join_asof` instead of the built-in's per-row `list.explode().search_sorted().over("__idx")`.

Why it is faster: the built-in explodes each row's endpoint list (rows x n_edges) and
runs a window keyed on a unique per-row index, so peak memory scales with rows x edges
and blows up on large shards. Here the endpoints are turned into a small per-code long
table once, and the big frame only ever carries (code, numeric_value); the match is an
O(n log n), memory-linear as-of join. Same 100 bins, seconds instead of hours.

Register it in your package's pyproject.toml under the MEDS_transforms.stages group, e.g.
    [project.entry-points."MEDS_transforms.stages"]
    bin_numeric_values_fast = "ehr2meds_custom_stages.bin_numeric_values_fast:stage"
then `pip install -e .` your custom-stages package so the entry point is discovered.
"""

import re
from collections.abc import Callable
from pathlib import Path

import polars as pl
from meds import CodeMetadataSchema, DataSchema
from omegaconf import DictConfig, OmegaConf

from MEDS_transforms.stages import Stage
from MEDS_transforms.utils import PKG_PFX, resolve_pkg_path

CODE = DataSchema.code_name
NV = DataSchema.numeric_value_name


def _bin_frame(
    df: pl.LazyFrame,
    code_metadata: pl.DataFrame,
    bin_with_columns: list[str],
    code_with_bin_name: str,
    join_cols: list[str],
    do_drop_numeric_value: bool,
) -> pl.LazyFrame:
    # 1) Per-code endpoint list from the (small) metadata only; coalesce the bin columns.
    ep = pl.coalesce(
        [
            pl.when(pl.col(c).is_not_null()).then(pl.concat_list(pl.col(c).struct.unnest()))
            for c in bin_with_columns
        ]
    )
    meta = code_metadata.lazy().select(*join_cols, ep.alias("__ep")).filter(pl.col("__ep").is_not_null())

    # 2) Long edge table for the as-of match, carrying the formatted [left, right) bounds.
    edges = (
        meta.with_row_index("__cr")
        .explode("__ep")
        .rename({"__ep": "__edge"})
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("__cr").alias("__idx"),
            pl.len().over("__cr").alias("__n"),
            pl.col("__edge").shift(-1).over("__cr").alias("__next"),
        )
        .with_columns(
            pl.col("__edge").cast(pl.String).alias("__left"),
            pl.when(pl.col("__idx") == pl.col("__n"))
            .then(pl.lit("inf"))
            .otherwise(pl.col("__next").cast(pl.String))
            .alias("__right"),
        )
        .select(*join_cols, "__edge", "__idx", "__left", "__right")
    )
    # First edge per code, used for the below-all-bins bin (index 0). Unique per code -> no fan-out.
    min_edge = meta.select(*join_cols, pl.col("__ep").list.first().cast(pl.String).alias("__e1"))

    # 3) As-of match: largest edge <= value, within code. Big frame carries only (row, code, value).
    d = df.with_row_index("__row")
    matched = (
        d.select("__row", *join_cols, NV)
        .filter(pl.col(NV).is_not_null())
        .sort(NV)
        .join_asof(edges.sort("__edge"), left_on=NV, right_on="__edge", by=join_cols, strategy="backward")
        .select(
            "__row",
            pl.col("__idx").alias("__m_idx"),
            pl.col("__left").alias("__m_left"),
            pl.col("__right").alias("__m_right"),
        )
    )
    d = d.join(min_edge, on=join_cols, how="left").join(matched, on="__row", how="left")

    do_bin = pl.col(NV).is_not_null() & pl.col("__e1").is_not_null()
    idx = pl.when(do_bin).then(pl.col("__m_idx").fill_null(0)).otherwise(None)
    left = pl.when(idx == 0).then(pl.lit("-inf")).otherwise(pl.col("__m_left"))
    right = pl.when(idx == 0).then(pl.col("__e1")).otherwise(pl.col("__m_right"))
    d = d.with_columns(idx.alias("__idx"), left.alias("__left"), right.alias("__right"))

    # 4) Format the modified code exactly as the built-in does.
    tmpl = re.sub(r"\{(code|left|right|bin)\}", "{}", code_with_bin_name)
    fields = re.findall(r"\{(code|left|right|bin)\}", code_with_bin_name)
    fmap = {
        "code": pl.col(CODE),
        "left": pl.col("__left"),
        "right": pl.col("__right"),
        "bin": pl.col("__idx").cast(pl.String),
    }
    new_code = pl.format(tmpl, *[fmap[f] for f in fields]) if fields else pl.lit(code_with_bin_name)
    d = d.with_columns(
        pl.when(pl.col("__idx").is_not_null()).then(new_code).otherwise(pl.col(CODE)).alias(CODE)
    )
    if do_drop_numeric_value:
        d = d.with_columns(
            pl.when(pl.col("__idx").is_not_null()).then(None).otherwise(pl.col(NV)).alias(NV)
        )

    helper = ["__row", "__idx", "__left", "__right", "__e1", "__m_idx", "__m_left", "__m_right"]
    return d.sort("__row").drop([c for c in helper if c in d.collect_schema().names()])


@Stage.register(is_metadata=False)
def bin_numeric_values_fast_fntr(
    stage_cfg: DictConfig,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Same config surface as the built-in bin_numeric_values stage."""
    if code_modifiers is None:
        code_modifiers = []

    custom_bins = stage_cfg.get("custom_bins", {})
    if isinstance(custom_bins, DictConfig):
        custom_bins = OmegaConf.to_container(custom_bins)
    custom_bins_fp = stage_cfg.get("custom_bins_filepath")
    if custom_bins_fp:
        fp = resolve_pkg_path(custom_bins_fp) if custom_bins_fp.startswith(PKG_PFX) else Path(custom_bins_fp)
        if not fp.is_file():
            raise FileNotFoundError(f"custom_bins_filepath '{custom_bins_fp}' does not exist.")
        file_bins = OmegaConf.load(fp)
        if isinstance(file_bins, DictConfig):
            file_bins = OmegaConf.to_container(file_bins)
        custom_bins = {**file_bins, **custom_bins}

    do_drop_numeric_value = stage_cfg.get("drop_numeric_value", False)
    bin_with_columns = list(stage_cfg.get("bin_with_columns", ["values/quantiles"]))
    code_with_bin_name = stage_cfg.get("code_with_bin_name", "{code}//value_[{left},{right})")

    cm = code_metadata
    if custom_bins:
        struct_dtype = pl.Struct(dict.fromkeys(next(iter(custom_bins.values())).keys(), pl.Float32))
        s = pl.Series(
            [custom_bins.get(c, None) for c in cm[CodeMetadataSchema.code_name]], dtype=struct_dtype
        )
        cm = cm.with_columns(s.alias("__custom_bins"))
        bin_with_columns = ["__custom_bins", *bin_with_columns]

    join_cols = [CodeMetadataSchema.code_name, *code_modifiers]
    cols = [c for c in bin_with_columns if c in cm.columns]
    cm = cm.select(*join_cols, *cols)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        nd = df.collect_schema()[NV]
        local = cm.with_columns(
            *[pl.col(c).cast(pl.Struct({f.name: nd for f in cm.schema[c].fields})) for c in cols]
        )
        return _bin_frame(df, local, cols, code_with_bin_name, join_cols, do_drop_numeric_value)

    return fn


stage = bin_numeric_values_fast_fntr