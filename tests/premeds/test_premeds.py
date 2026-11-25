"""End-to-end tests of the raw -> preMEDS stage on small inputs."""

import datetime as dt
import logging
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ehr2meds.preMEDS.extractor import PREMEDSExtractor
from ehr2meds.preMEDS.mappings import apply_mapping
from omegaconf import OmegaConf
from pathlib import Path


def run_premeds(output: Path, tables: dict, population: Path, write_file_type="parquet", chunksize=100) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output": str(output)},
            "write_file_type": write_file_type,
            "chunksize": chunksize,
            "test": False,
            "num_workers": 1,
            "subject_id_mapping": {"file": str(population), "subject_id_col": "cpr", "mapping_id_col": None},
            "tables": tables,
        }
    )
    output.mkdir(parents=True)
    PREMEDSExtractor(cfg)()


def run_csv_table(tmp_path: Path, population_csv: str, table_csv: str, table_cfg: dict) -> pl.DataFrame:
    """Run one CSV table through preMEDS and return its parquet output."""
    (tmp_path / "population.csv").write_text(population_csv)
    (tmp_path / "table.csv").write_text(table_csv)
    tables = {"table": {"filename": str(tmp_path / "table.csv"), **table_cfg}}
    run_premeds(tmp_path / "out", tables, tmp_path / "population.csv")
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
    (tmp_path / "map.csv").write_text('contact_id,region_code\n1,1084\n,9999\n"  ",8888\n')
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
    table = 'pid,diag,date,time,contact_id\nP01,A,2020-01-05,,1\nP01,B,2020-01-05,,\nP01,C,2020-01-05,,"  "\n'
    out = run_csv_table(tmp_path, "cpr,x\nP01,a\n", table, table_cfg)
    assert out.sort("row_idx")["region_code"].to_list() == ["1084", None, None]  # CSV columns are read as text


def test_numeric_subject_ids_are_kept_as_strings(tmp_path):
    out = run_csv_table(tmp_path, "cpr,x\n0101,a\n,b\n", "pid,diag,date,time\n0101,A,2020-01-05,10:00:00\n", SIMPLE_TABLE)
    assert out.height == 1
    mapping = pl.read_parquet(tmp_path / "out" / "subject_id_mapping.parquet")
    assert mapping["cpr"].to_list() == ["0101"]


@pytest.mark.parametrize("time_type", [pa.time32("s"), pa.time64("us"), pa.duration("s")])
def test_parquet_time_columns_keep_their_time(tmp_path, time_type):
    # polars' own parquet reader misreads time32 columns, which silently dropped times on real data.
    seconds = [45296, 29700]  # 12:34:56, 08:15:00
    times = pa.array(seconds, pa.int64()).cast(pa.duration("s")) if time_type == pa.duration("s") else None
    if times is None:
        times = pa.array([dt.time(s // 3600, s // 60 % 60, s % 60) for s in seconds], time_type)
    pq.write_table(
        pa.table({"pid": ["P01", "P01"], "diag": ["A", "A"], "date": ["2020-01-05"] * 2, "time": times}),
        tmp_path / "t.parquet",
    )
    (tmp_path / "population.csv").write_text("cpr,x\nP01,a\n")
    run_premeds(
        tmp_path / "out", {"t": {"filename": str(tmp_path / "t.parquet"), **SIMPLE_TABLE}}, tmp_path / "population.csv"
    )
    out = pl.read_parquet(tmp_path / "out" / "t.parquet").sort("row_idx")
    assert out["timestamp"].to_list() == [dt.datetime(2020, 1, 5, 12, 34, 56), dt.datetime(2020, 1, 5, 8, 15)]


def test_csv_na_strings_are_missing(tmp_path):
    # NA, NULL, ... are missing, so a "NULL" code must not survive the mandatory-column filter.
    table = "pid,diag,date,time\nP01,A,2020-01-05,\nP01,NULL,2020-01-05,\nP01,NA,2020-01-05,\n"
    out = run_csv_table(tmp_path, "cpr,x\nP01,a\n", table, SIMPLE_TABLE)
    assert out["code"].to_list() == ["A"]


def test_worker_warnings_reach_the_parent_log(tmp_path, caplog):
    # spawn workers don't inherit logging, so this only passes if records are sent back to the parent.
    with caplog.at_level(logging.INFO):
        run_csv_table(tmp_path, "cpr,x\nP01,a\n", "pid,diag,date,time\nP01,A,not-a-date,\nP01,B,2020-01-05,\n", SIMPLE_TABLE)
    assert "match no date format" in caplog.text


@pytest.mark.parametrize("write_file_type", ["parquet", "csv"])
def test_output_does_not_depend_on_chunksize(tmp_path, write_file_type):
    # Codes and times whose inferred type would differ per chunk. No duplicates: those are only removed per chunk.
    rows = ["5,2020-01-05,08:00", "ALCC01,05/01/2020,", "0101,20200105,23:59:59", "5.0,2020-01-06,", "X,06.01.2020,12:30",
            "7,2020-01-07,", "0102,20200108,10:00"]  # fmt: skip
    table = "pid,diag,date,time\n" + "".join(f"P01,{row}\n" for row in rows)
    (tmp_path / "population.csv").write_text("cpr,x\nP01,a\n")
    (tmp_path / "table.csv").write_text(table)
    tables = {"table": {"filename": str(tmp_path / "table.csv"), **SIMPLE_TABLE}}
    for chunksize in (3, 100):
        run_premeds(tmp_path / str(chunksize), tables, tmp_path / "population.csv", write_file_type, chunksize)
    small, large = (tmp_path / str(size) / f"table.{write_file_type}" for size in (3, 100))
    assert (
        small.read_bytes() == large.read_bytes()
        if write_file_type == "csv"
        else pl.read_parquet(small).equals(pl.read_parquet(large))
    )


def test_padded_text_keys_match_numeric_mapping_keys():
    # Against an integer key, text "08" is the number 8. Text against text stays exact.
    df = pl.DataFrame({"c_spec": ["08", " 8 ", "8.0", "00", "0101"]})
    numeric_map = pl.DataFrame({"c_spec": [8, 0, 101], "specialty": ["ENDO", "MIXED", "X"]})
    out = apply_mapping(df, numeric_map, "c_spec", "c_spec", {"specialty": None}, how="left")
    assert out["specialty"].to_list() == ["ENDO", "ENDO", "ENDO", "MIXED", "X"]

    text_map = pl.DataFrame({"c_spec": ["101"], "specialty": ["X"]})
    out = apply_mapping(df.tail(1), text_map, "c_spec", "c_spec", {"specialty": None}, how="left")
    assert out["specialty"].to_list() == [None]
