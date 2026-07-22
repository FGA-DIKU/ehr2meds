"""Efficient replacement for the MEDS-Transforms `bin_numeric_values` stage.

Has been made specifically to produce the same output as the
MEDS-Transforms `bin_numeric_values` stage (same bin indices, same
`{code}//value_[{left},{right})` / `{bin}` formatting, same null and out-of-range
handling, same `drop_numeric_value` behavior), but computes the bin index with a
per-code `join_asof` instead of the built-in's per-row `list.explode().search_sorted().over("__idx")`.

This was done because the built-in stage is very slow on large shards, and can run out of memory.

NOTE: Attribution
-----------
The functionality is deliberately made to mirror MEDS-Transforms'
``bin_numeric_values`` (MIT licensed, v0.6.7). The binning
computation (``assign_value_bins`` and its helpers) is a reimplementation via
``join_asof`` rather than a copy of their per-row explode/window approach. Output was
verified equal to the built-in stage on v0.6.7 across both code templates,
``drop_numeric_value`` on/off, tied/duplicate edges, and codes without bins.
"""

import polars as pl
import re
from collections.abc import Callable
from meds import CodeMetadataSchema, DataSchema
from MEDS_transforms.stages import Stage
from MEDS_transforms.utils import PKG_PFX, resolve_pkg_path
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

CODE = DataSchema.code_name
VALUE = DataSchema.numeric_value_name


def edges_per_code(code_metadata: pl.DataFrame, bin_columns: list[str], key: list[str]) -> pl.LazyFrame:
    """One row per code (+ modifiers) holding its sorted list of bin edges.

    Extracted from the code_metadata stage. Edges are read from a struct column:
    field names are ignored, the (already sorted) struct values are the edges.
    If several ``bin_columns`` are provided, the first non-null struct for a code wins.
    Codes with no edges are dropped.
    """
    edge_list = pl.coalesce(
        [pl.when(pl.col(c).is_not_null()).then(pl.concat_list(pl.col(c).struct.unnest())) for c in bin_columns]
    )
    return code_metadata.lazy().select(*key, edge_list.alias("edges")).filter(pl.col("edges").is_not_null())


def bin_table(code_metadata: pl.DataFrame, bin_columns: list[str], key: list[str], value_dtype: pl.DataType) -> pl.LazyFrame:
    """One row per (code, bin) describing every bin as a half-open interval ``[left, right)``.

    A code's sorted edges ``e[1..n]`` are padded with sentinels into the boundary list
    ``[-inf, e[1], ..., e[n], +inf]``. Consecutive boundaries form the bins, so bin ``k`` is
    ``[boundary[k], boundary[k+1])``. ``edge`` is the bin's left boundary and is what values
    are matched against: a value's bin is the one whose ``edge`` is the largest ``<=`` it.
    Codes with no edges produce no rows, so their measurements are left unbinned.

        edges [0.0, 1.0, 2.0]  ->  bin 0  edge=-inf  [-inf, 0.0)
                                   bin 1  edge=0.0   [0.0, 1.0)
                                   bin 2  edge=1.0   [1.0, 2.0)
                                   bin 3  edge=2.0   [2.0, inf)
    """
    edges = edges_per_code(code_metadata, bin_columns, key)

    # Pad edges with -inf/+inf so all bins are pairs of boundaries.
    padded = edges.select(
        *key,
        pl.concat_list([pl.lit(float("-inf")), pl.col("edges"), pl.lit(float("inf"))])
        .cast(pl.List(value_dtype))
        .alias("boundaries"),
    )

    # Bin k spans [boundaries[k], boundaries[k+1]).
    # Only special cases are the first and the last bin.
    # Exploding the two lists together yields one row per bin.
    interval_ends = padded.select(
        *key,
        left_edge=pl.col("boundaries").list.slice(0, pl.col("boundaries").list.len() - 1),
        right_edge=pl.col("boundaries").list.slice(1),
    )
    return interval_ends.explode(["left_edge", "right_edge"]).select(
        *key,
        bin=pl.int_range(0, pl.len()).over(key),  # 0-based bin index within the code
        edge=pl.col("left_edge"),  # values are matched against the bin's left edge
        left=pl.col("left_edge").cast(pl.String),
        right=pl.col("right_edge").cast(pl.String),
    )


def render_code(template: str) -> pl.Expr:
    """Compile a code template into a string expression.

    The template mixes literal text with the placeholders ``{code}``, ``{left}``,
    ``{right}`` and ``{bin}``.
    These are split into pieces, and each placeholder is swapped for the
    matching column, and thereafter concatenated.

    For example
    ``"{code}//value_[{left},{right})"`` becomes
    ``code + "//value_[" + left + "," + right + ")"``.
    """
    placeholder_column = {
        "{code}": pl.col(CODE),
        "{left}": pl.col("left"),
        "{right}": pl.col("right"),
        "{bin}": pl.col("bin").cast(pl.String),
    }
    pieces = []
    # re.split keeps the delimiters (the placeholders) as separate items in the list.
    for piece in re.split(r"(\{code\}|\{left\}|\{right\}|\{bin\})", template):
        if piece in placeholder_column:
            pieces.append(placeholder_column[piece])
        elif piece:  # non-empty literal text
            pieces.append(pl.lit(piece))
    return pl.concat_str(pieces)


def assign_value_bins(
    data: pl.LazyFrame,
    code_metadata: pl.DataFrame,
    *,
    bin_columns: list[str],
    code_template: str,
    key: list[str],
    drop_numeric_value: bool,
) -> pl.LazyFrame:
    """Rewrite ``code`` to include the value's bin, matching the built-in stage's output.

    Args:
        data: MEDS data frame with ``code``, ``numeric_value`` and the ``key`` columns.
        code_metadata: per-code frame carrying the bin-edge struct column(s).
        bin_columns: struct columns holding edges (first non-null wins).
        code_template: e.g. ``"{code}//value_[{left},{right})"`` or ``"{code}//{bin}"``.
        key: join columns, ``["code", *code_modifiers]``.
        drop_numeric_value: if True, null out ``numeric_value`` on rows that were binned.

    Returns:
        The data frame with binned codes; unbinned rows (null value or code without edges)
        are returned unchanged. Helper columns are dropped.

    Example:
        >>> data = pl.LazyFrame({
        ...     "code": ["lab//A", "lab//A", "lab//A", "lab//A", "dx//1"],
        ...     "numeric_value": [-1.0, 0.5, 1.0, 3.0, None],
        ... })
        >>> code_metadata = pl.DataFrame(
        ...     {"code": ["lab//A", "dx//1"]},
        ... ).with_columns(pl.Series("values/quantiles", [{"a": 0.0, "b": 1.0, "c": 2.0}, None]))
        >>> assign_value_bins(
        ...     data, code_metadata, bin_columns=["values/quantiles"],
        ...     code_template="{code}//value_[{left},{right})", key=["code"],
        ...     drop_numeric_value=False,
        ... ).collect()["code"].to_list()
        ['lab//A//value_[-inf,0.0)', 'lab//A//value_[0.0,1.0)', 'lab//A//value_[1.0,2.0)', 'lab//A//value_[2.0,inf)', 'dx//1']
    """
    value_dtype = data.collect_schema()[VALUE]
    bins = bin_table(code_metadata, bin_columns, key, value_dtype)

    rows = data.with_row_index("_row")

    # The important change from the MEDS_transform version:
    # Match each value to its bin, (largest edge <= the value).
    # Do one as-of join instead of the built-in per-row explode + window.
    # This is the whole speedup. Codes with no bins are absent from
    # `bins`, so their rows get no match and pass through unchanged.
    matches = (
        rows.select("_row", *key, VALUE)
        .filter(pl.col(VALUE).is_not_null())
        .sort(VALUE)
        .join_asof(bins.sort(["edge", "bin"]), left_on=VALUE, right_on="edge", by=key, strategy="backward")
        .select("_row", "bin", "left", "right")
    )
    labelled = rows.join(matches, on="_row", how="left")

    # A matched row (bin is not null) gets its rewritten code
    was_binned = pl.col("bin").is_not_null()
    labelled = labelled.with_columns(pl.when(was_binned).then(render_code(code_template)).otherwise(pl.col(CODE)).alias(CODE))
    if drop_numeric_value:
        labelled = labelled.with_columns(pl.when(was_binned).then(None).otherwise(pl.col(VALUE)).alias(VALUE))

    working = ["_row", "bin", "left", "right"]
    return labelled.sort("_row").drop([c for c in working if c in labelled.collect_schema().names()])


def load_custom_bins(stage_cfg: DictConfig) -> dict:
    """Read inline ``custom_bins`` and/or a ``custom_bins_filepath`` YAML, inline taking
    precedence, matching the built-in stage."""
    inline = stage_cfg.get("custom_bins", {})
    if isinstance(inline, DictConfig):
        inline = OmegaConf.to_container(inline)

    fp = stage_cfg.get("custom_bins_filepath")
    if not fp:
        return inline or {}

    path = resolve_pkg_path(fp) if fp.startswith(PKG_PFX) else Path(fp)
    if not path.is_file():
        raise FileNotFoundError(f"custom_bins_filepath '{fp}' does not exist.")
    from_file = OmegaConf.load(path)
    if isinstance(from_file, DictConfig):
        from_file = OmegaConf.to_container(from_file)
    if not isinstance(from_file, dict):
        raise TypeError("custom_bins_filepath must point to a YAML file with a dictionary")
    return {**from_file, **inline}


@Stage.register(is_metadata=False)
def bin_numeric_values_fast_fntr(
    stage_cfg: DictConfig,
    code_metadata: pl.DataFrame,
    code_modifiers: list[str] | None = None,
) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Build the per-shard binning function.

    Recognized ``stage_cfg`` keys (same as the built-in stage):
        bin_with_columns:     struct columns holding edges. Default ``["values/quantiles"]``.
        code_with_bin_name:   code template. Default ``"{code}//value_[{left},{right})"``.
        drop_numeric_value:   null the value on binned rows. Default ``False``.
        custom_bins:          inline ``{code: {name: edge, ...}}`` mapping. Optional.
        custom_bins_filepath: YAML file with the same mapping. Optional.
    """
    code_modifiers = code_modifiers or []
    key = [CodeMetadataSchema.code_name, *code_modifiers]

    bin_columns = list(stage_cfg.get("bin_with_columns", ["values/quantiles"]))
    code_template = stage_cfg.get("code_with_bin_name", "{code}//value_[{left},{right})")
    drop_numeric_value = stage_cfg.get("drop_numeric_value", False)

    # Optional custom edges, which override the code_metadata stage's edges.
    custom_bins = load_custom_bins(stage_cfg)
    metadata = code_metadata
    if custom_bins:
        struct_dtype = pl.Struct(dict.fromkeys(next(iter(custom_bins.values())).keys(), pl.Float32))
        custom_series = pl.Series([custom_bins.get(c) for c in metadata[CodeMetadataSchema.code_name]], dtype=struct_dtype)
        metadata = metadata.with_columns(custom_series.alias("__custom_bins"))
        bin_columns = ["__custom_bins", *bin_columns]

    bin_columns = [c for c in bin_columns if c in metadata.columns]
    metadata = metadata.select(*key, *bin_columns)

    def bin_shard(df: pl.LazyFrame) -> pl.LazyFrame:
        return assign_value_bins(
            df,
            metadata,
            bin_columns=bin_columns,
            code_template=code_template,
            key=key,
            drop_numeric_value=drop_numeric_value,
        )

    return bin_shard


# Entry point
stage = bin_numeric_values_fast_fntr
