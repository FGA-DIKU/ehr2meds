"""Known bugs in the pandas preMEDS stage, written as the behaviour we want.

Each test is a strict xfail: it fails today, and once a refactor fixes the bug the unexpected pass
fails the run, so the mark has to be removed and the fix is recorded. Their inputs are kept out of
the baseline in test_premeds_parity.py, so fixing a bug never requires re-recording it.
"""

import datetime as dt
import polars as pl
import pytest
from pathlib import Path
from tests.premeds_parity.compare import compare_dirs
from tests.premeds_parity.inputs import table_configs, write_inputs
from tests.premeds_parity.test_premeds_parity import run_premeds


def bug(reason: str):
    return pytest.mark.xfail(strict=True, reason=f"BUG: {reason}")


def run_csv_table(tmp_path: Path, population_csv: str, table_csv: str, table_cfg: dict, chunksize=100) -> pl.DataFrame:
    """Run one CSV table through preMEDS and return its parquet output."""
    (tmp_path / "population.csv").write_text(population_csv)
    (tmp_path / "table.csv").write_text(table_csv)
    tables = {"table": {"filename": str(tmp_path / "table.csv"), **table_cfg}}
    run_premeds(tmp_path / "out", tables, tmp_path / "population.csv", chunksize=chunksize)
    return pl.read_parquet(tmp_path / "out" / "table.parquet")


SIMPLE_TABLE = {
    "columns": {"pid": "subject_id", "diag": "code", "date": None, "time": None},
    "timestamp_columns": {"timestamp": {"date": "date", "time": "time"}},
}


def test_hh_mm_times_are_parsed(tmp_path):
    out = run_csv_table(tmp_path, "cpr,x\nP01,a\n", "pid,diag,date,time\nP01,A,2020-01-05,12:34\n", SIMPLE_TABLE)
    assert out["timestamp"].to_list() == [dt.datetime(2020, 1, 5, 12, 34)]


def test_common_date_formats_are_parsed(tmp_path):
    # ISO and compact dates are year-first; separated dates are day-first (Danish convention).
    dates = ["2020-01-05", "2020-01-05 13:45:00", "20200105", "05/01/2020", "5/1/2020", "05-01-2020", "05.01.2020"]
    table = "pid,diag,date,time\n" + "".join(f"P01,C{i},{d},\n" for i, d in enumerate(dates))
    out = run_csv_table(tmp_path, "cpr,x\nP01,a\n", table, SIMPLE_TABLE)
    assert out.sort("row_idx")["timestamp"].to_list() == [dt.datetime(2020, 1, 5)] * len(dates)


def test_null_keys_do_not_match_in_mappings(tmp_path):
    (tmp_path / "map.csv").write_text("contact_id,region_code\n1,1084\n,9999\n")
    table_cfg = {
        "columns": {**SIMPLE_TABLE["columns"], "contact_id": None},
        "timestamp_columns": SIMPLE_TABLE["timestamp_columns"],
        "mappings": [
            {
                "source_column": "contact_id",
                "via_file": str(tmp_path / "map.csv"),
                "join_on": "contact_id",
                "target_columns": {"region_code": None},
                "how": "left",
            }
        ],
    }
    table = "pid,diag,date,time,contact_id\nP01,A,2020-01-05,,1\nP01,B,2020-01-05,,\n"
    out = run_csv_table(tmp_path, "cpr,x\nP01,a\n", table, table_cfg)
    assert out.sort("row_idx")["region_code"].to_list() == ["1084", None]  # CSV columns are read as text


def test_numeric_subject_ids_are_kept_as_strings(tmp_path):
    out = run_csv_table(tmp_path, "cpr,x\n0101,a\n,b\n", "pid,diag,date,time\n0101,A,2020-01-05,10:00:00\n", SIMPLE_TABLE)
    assert out.height == 1
    mapping = pl.read_csv(tmp_path / "out" / "subject_id_mapping.csv", infer_schema=False)
    assert mapping["cpr"].to_list() == ["0101"]


def test_duplicates_are_removed_across_chunks(tmp_path):
    table = "pid,diag,date,time\nP01,A,2020-01-05,10:00:00\nP01,B,2020-01-05,10:00:00\nP01,A,2020-01-05,10:00:00\n"
    out = run_csv_table(tmp_path, "cpr,x\nP01,a\n", table, SIMPLE_TABLE, chunksize=2)
    assert out["code"].to_list() == ["A", "B"]


@pytest.mark.parametrize("write_file_type", ["parquet", "csv"])
def test_output_does_not_depend_on_chunksize(tmp_path, write_file_type):
    raw_dir = tmp_path / "raw"
    write_inputs(raw_dir)
    for chunksize in (4, 100):
        run_premeds(tmp_path / str(chunksize), table_configs(raw_dir), raw_dir / "population.csv", write_file_type, chunksize)
    assert not compare_dirs(tmp_path / "100", tmp_path / "4")
