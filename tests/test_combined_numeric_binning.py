import polars as pl
from ehr2meds.meds_stages.aggregate_numeric_metadata import (
    VALUES,
    fit_numeric_metadata,
    mapper_fntr,
)
from ehr2meds.meds_stages.annotate_numeric_values import (
    BIN_EDGES,
    BIN_REPRESENTATIVES,
    DERIVED_COLUMNS,
    IS_CONSTANT,
    LOWER_BOUND,
    UPPER_BOUND,
    _combine_numeric_metadata,
    _load_external_metadata,
    annotate_numeric_values,
)
from omegaconf import DictConfig


def _mapped(values, code="L//A"):
    return mapper_fntr(DictConfig({}), None)(pl.LazyFrame({"code": [code] * len(values), "numeric_value": values})).collect()


def _fit(*shards, min_bins=2, max_bins=20):
    return fit_numeric_metadata(*shards, key=["code"], min_bins=min_bins, max_bins=max_bins)


def test_training_only_fit_and_clipping_preserve_raw_value_and_code():
    training = _mapped(list(range(100)))
    metadata = _fit(training)
    # Evaluation values are deliberately never supplied to the reducer.
    data = pl.LazyFrame({"code": ["L//A", "L//A"], "numeric_value": [-100.0, 1000.0]})
    out = annotate_numeric_values(data, metadata, key=["code"]).collect()

    assert metadata["numeric/p1"][0] == 0.99
    assert metadata["numeric/p99"][0] == 98.01
    assert out["code"].to_list() == ["L//A", "L//A"]
    assert out["numeric_value"].to_list() == [-100.0, 1000.0]
    assert out["numeric_value_normalized"].to_list() == [0.0, 1.0]
    assert out["numeric_value_was_clipped"].to_list() == [True, True]


def test_constant_rare_duplicate_edges_and_unseen_concepts():
    constant = _mapped([7.0], "L//CONSTANT")
    tied = _mapped([0.0] * 20 + [1.0] * 20, "L//TIED")
    metadata = _fit(constant, tied, min_bins=5, max_bins=10)

    constant_meta = metadata.filter(pl.col("code") == "L//CONSTANT").row(0, named=True)
    tied_meta = metadata.filter(pl.col("code") == "L//TIED").row(0, named=True)
    assert constant_meta["numeric/is_constant"] is True
    assert constant_meta["numeric/effective_bin_count"] == 1
    assert tied_meta["numeric/effective_bin_count"] < tied_meta["numeric/requested_bin_count"]

    data = pl.LazyFrame({"code": ["L//CONSTANT", "L//UNSEEN", "DX//1"], "numeric_value": [7.0, 3.0, None]})
    out = annotate_numeric_values(data, metadata, key=["code"]).collect()
    assert out["numeric_value_present"].to_list() == [True, True, False]
    assert out["numeric_value_normalized"].to_list() == [0.0, None, None]
    assert out["numeric_value_bin"].to_list() == [0, None, None]


def test_reduction_is_deterministic_across_shard_layouts():
    one_shard = _mapped([float(i % 7) for i in range(100)])
    first = pl.DataFrame({"code": ["L//A"], VALUES: [one_shard[VALUES][0][:37]]})
    second = pl.DataFrame({"code": ["L//A"], VALUES: [one_shard[VALUES][0][37:]]})
    pl.testing.assert_frame_equal(_fit(one_shard), _fit(second, first))


def test_custom_columns_have_canonical_final_types():
    metadata = _fit(_mapped([0.0, 1.0, 2.0]))
    out = annotate_numeric_values(pl.LazyFrame({"code": ["L//A"], "numeric_value": [1.0]}), metadata, key=["code"]).collect()
    for column, dtype in DERIVED_COLUMNS.items():
        assert out.schema[column] == dtype


def test_external_metadata_overrides_local_transforms_with_local_fallback():
    local = _fit(_mapped([0.0, 100.0], "L//A"), _mapped([10.0, 20.0], "L//LOCAL_ONLY"))
    external = pl.DataFrame(
        {
            "code": ["L//A"],
            LOWER_BOUND: [0.0],
            UPPER_BOUND: [10.0],
            BIN_EDGES: [[0.5]],
            BIN_REPRESENTATIVES: [[0.25, 0.75]],
            IS_CONSTANT: [False],
        }
    )
    combined = _combine_numeric_metadata(local, external, key=["code"])
    data = pl.LazyFrame({"code": ["L//A", "L//LOCAL_ONLY"], "numeric_value": [7.5, 15.0]})
    out = annotate_numeric_values(data, combined, key=["code"]).collect()

    assert out["numeric_value_normalized"].to_list() == [0.75, 0.5]
    assert out["numeric_value_bin"].to_list()[0] == 1
    assert out["numeric_value_binned"].to_list()[0] == 0.75


def test_external_metadata_path_accepts_file_or_metadata_directory(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata = _fit(_mapped([0.0, 1.0, 2.0]))
    filepath = metadata_dir / "codes.parquet"
    metadata.write_parquet(filepath)

    pl.testing.assert_frame_equal(_load_external_metadata(str(filepath)), metadata)
    pl.testing.assert_frame_equal(_load_external_metadata(str(metadata_dir)), metadata)
