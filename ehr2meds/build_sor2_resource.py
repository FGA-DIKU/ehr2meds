"""Build the compact SOR history used for contact enrichment."""

from __future__ import annotations

import argparse
import pandas as pd
from ehr2meds.preMEDS.utils import normalize_sor_id
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
    source = pd.read_csv(
        args.sor_entity_csv,
        sep=";",
        usecols=list(SOURCE_COLUMNS),
        dtype="string",
    ).rename(columns=SOURCE_COLUMNS)
    source["sor_id"] = normalize_sor_id(source["sor_id"])
    source["valid_from"] = pd.to_datetime(source["valid_from"], errors="coerce")
    source["valid_to"] = pd.to_datetime(source["valid_to"], errors="coerce")
    source["region"] = normalize_sor_id(source["region"])
    source["primary_specialty"] = source["primary_specialty"].str.strip().replace("", pd.NA)
    source = source.dropna(subset=["sor_id"]).drop_duplicates()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source.to_parquet(args.output, index=False)
    print(f"Wrote {len(source):,} SOR history rows to {args.output}")
    print(f"Unique SOR IDs: {source['sor_id'].nunique():,}")
    print(f"Rows with region: {source['region'].notna().sum():,}")
    print(f"Rows with primary specialty: {source['primary_specialty'].notna().sum():,}")


if __name__ == "__main__":
    main()
