"""Build the compact SOR history used for contact enrichment."""

from __future__ import annotations

import argparse
import polars as pl
from ehr2meds.preMEDS.mappings import normalize_sor_id
from pathlib import Path

SOURCE_COLUMNS = {
    "SorId": "sor_id",
    "FromDate": "valid_from",
    "ToDate": "valid_to",
    "PostalAddressRegionCode": "region",
    "PrioritizedEntitySpeciality1Name": "primary_specialty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sor_entity_csv", type=Path, help="Official SOR2 SOREntity.csv")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path(__file__).parents[1] / "resources" / "sor2_contact_mapping.parquet",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = (
        pl.read_csv(args.sor_entity_csv, separator=";", columns=list(SOURCE_COLUMNS), infer_schema=False)
        .rename(SOURCE_COLUMNS)
        .with_columns(
            normalize_sor_id(pl.col("sor_id")),
            pl.col("valid_from", "valid_to").str.strip_chars().str.to_datetime(strict=False, time_unit="us"),
            normalize_sor_id(pl.col("region")),
            pl.col("primary_specialty").str.strip_chars().replace("", None),
        )
        .drop_nulls("sor_id")
        .unique(maintain_order=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.write_parquet(args.output)
    print(f"Wrote {source.height:,} SOR history rows to {args.output}")
    print(f"Unique SOR IDs: {source['sor_id'].n_unique():,}")
    print(f"Rows with region: {source['region'].count():,}")
    print(f"Rows with primary specialty: {source['primary_specialty'].count():,}")


if __name__ == "__main__":
    main()
