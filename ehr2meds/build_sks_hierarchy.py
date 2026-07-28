"""Build a compact hierarchy resource from the official Danish SKS export."""

from __future__ import annotations

import argparse
import hashlib
import polars as pl
import urllib.request
from collections.abc import Iterable
from pathlib import Path

DEFAULT_SOURCE_URL = "https://filer.sundhedsdata.dk/sks/data/skscomplete/SKScomplete.txt"
RECORD_LENGTH = 214


def parse_sks_lines(lines: Iterable[str]) -> pl.DataFrame:
    """Parse the fixed-width SKS exchange format.

    Field positions follow Sundhedsdatastyrelsen's published SKS table
    specification. Empty and malformed lines are rejected instead of silently
    producing incomplete hierarchy entries.
    """
    records: list[dict[str, str | bool | None]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if not line:
            continue
        if len(line) < RECORD_LENGTH:
            raise ValueError(f"SKS line {line_number} has {len(line)} characters; expected at least {RECORD_LENGTH}")

        code = line[3:23].strip()
        if not code:
            raise ValueError(f"SKS line {line_number} has an empty code")

        records.append(
            {
                "record_type": line[0:3],
                "code": code,
                "valid_from": line[23:31],
                "changed_at": line[31:39],
                "valid_to": line[39:47],
                "description": line[47:167].strip(),
                "group_a": line[167:170].strip() or None,
                "group_b": line[170:173].strip() or None,
                "catalogue": line[179:182].strip() or None,
                "sex": line[182:183].strip() or None,
                "age_from": line[183:185].strip() or None,
                "age_to": line[185:187].strip() or None,
                "change_type": line[187:188].strip() or None,
                "requires_additional_registration": line[213:214] == "*",
            }
        )

    if not records:
        raise ValueError("The SKS source contained no records")

    return pl.DataFrame(records, infer_schema_length=None).with_columns(
        pl.col("valid_from").str.to_date("%Y%m%d"),
        pl.col("changed_at").str.to_date("%Y%m%d"),
        pl.col("valid_to").str.to_date("%Y%m%d"),
    )


def add_nearest_prefix_parent(sks: pl.DataFrame) -> pl.DataFrame:
    """Add the nearest shorter official SKS code prefix as a parent.

    Parent discovery uses every code in the SKS catalogue, not only codes with
    the same record type. Some procedure leaves and their grouping nodes are
    published under different record types. Codes without an official prefix
    ancestor intentionally receive a null parent.
    """
    all_codes = set(sks.get_column("code").unique().to_list())
    parent_by_code: dict[str, str | None] = {}
    for code in sorted(all_codes):
        parent_by_code[code] = next(
            (code[:length] for length in range(len(code) - 1, 0, -1) if code[:length] in all_codes),
            None,
        )

    parents = pl.DataFrame(
        {
            "code": list(parent_by_code),
            "parent_code": list(parent_by_code.values()),
        },
        schema={"code": pl.String, "parent_code": pl.String},
    )
    return sks.join(parents, on="code", how="left").select(
        "record_type",
        "code",
        "parent_code",
        "valid_from",
        "valid_to",
        "changed_at",
        "description",
        "group_a",
        "group_b",
        "catalogue",
        "sex",
        "age_from",
        "age_to",
        "change_type",
        "requires_additional_registration",
    )


def build_resource(source: Path, output: Path) -> pl.DataFrame:
    """Parse ``source``, derive parents, and write deterministic Parquet."""
    with source.open(encoding="cp1252") as input_file:
        hierarchy = add_nearest_prefix_parent(parse_sks_lines(input_file))

    hierarchy = hierarchy.sort("record_type", "code", "valid_from", "valid_to")
    output.parent.mkdir(parents=True, exist_ok=True)
    hierarchy.write_parquet(output, compression="zstd", statistics=True)
    return hierarchy


def download_source(url: str, destination: Path) -> str:
    """Download an SKS exchange file and return its SHA-256 digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            output_file.write(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing SKScomplete.txt file")
    parser.add_argument("--url", default=DEFAULT_SOURCE_URL, help="Official SKS download URL")
    parser.add_argument("--output", type=Path, required=True, help="Output Parquet path")
    parser.add_argument(
        "--download-to",
        type=Path,
        help="Where to download the source when --source is omitted",
    )
    args = parser.parse_args()

    source = args.source
    if source is None:
        if args.download_to is None:
            parser.error("--download-to is required when --source is omitted")
        source = args.download_to
        digest = download_source(args.url, source)
        print(f"Downloaded {source} (sha256={digest})")

    hierarchy = build_resource(source, args.output)
    print(f"Wrote {hierarchy.height} SKS records to {args.output}")


if __name__ == "__main__":
    main()
