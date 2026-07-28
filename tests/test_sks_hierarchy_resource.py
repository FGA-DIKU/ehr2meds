from __future__ import annotations

import polars as pl
from pathlib import Path

RESOURCE = Path(__file__).parents[1] / "ehr2meds" / "resources" / "sks_hierarchy.parquet"


def test_sks_resource_integrity_and_known_parents() -> None:
    hierarchy = pl.read_parquet(RESOURCE)

    assert hierarchy.height == 107_397
    assert hierarchy.select("record_type", "code", "valid_from", "valid_to").unique().height == hierarchy.height
    assert hierarchy.filter(pl.col("code") == pl.col("parent_code")).is_empty()
    assert hierarchy.filter(
        pl.col("parent_code").is_not_null() & ~pl.col("code").str.starts_with(pl.col("parent_code"))
    ).is_empty()

    parents = dict(
        hierarchy.filter(pl.col("code").is_in(["DC833A", "DC833", "DC83", "KABA00", "KABA"]))
        .select("code", "parent_code")
        .unique()
        .iter_rows()
    )
    assert parents == {
        "DC83": "D",
        "DC833": "DC83",
        "DC833A": "DC833",
        "KABA": "KAB",
        "KABA00": "KABA",
    }


def test_sks_resource_has_no_cycles() -> None:
    hierarchy = pl.read_parquet(RESOURCE)
    parent_by_code = dict(hierarchy.select("code", "parent_code").unique().iter_rows())

    for start in parent_by_code:
        seen = {start}
        parent = parent_by_code[start]
        while parent is not None:
            assert parent not in seen, f"Hierarchy cycle from {start!r} through {parent!r}"
            seen.add(parent)
            parent = parent_by_code.get(parent)
