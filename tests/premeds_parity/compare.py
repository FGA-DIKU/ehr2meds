"""Compare two preMEDS output directories: same files, same schemas, same rows in the same order.

Use it on real or synthetic runs too, e.g. run the pipeline on the pandas branch and the polars
branch, then:

    python tests/premeds_parity/compare.py <pandas_output_dir> <polars_output_dir>
"""

import difflib
import polars as pl
import sys
from pathlib import Path
from polars.testing import assert_frame_equal

OUTPUT_SUFFIXES = {".parquet", ".csv", ".txt"}


def _files(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*") if p.is_file() and p.suffix in OUTPUT_SUFFIXES}


def compare_dirs(expected: Path, actual: Path) -> list[str]:
    """Return one message per difference; an empty list means the outputs match."""
    expected_files, actual_files = _files(expected), _files(actual)
    problems = [f"missing: {p}" for p in sorted(expected_files - actual_files)]
    problems += [f"unexpected: {p}" for p in sorted(actual_files - expected_files)]

    for rel in sorted(expected_files & actual_files):
        if rel.suffix == ".parquet":
            # Compared as polars sees them, since the downstream MEDS stages read them with polars.
            try:
                assert_frame_equal(pl.read_parquet(expected / rel), pl.read_parquet(actual / rel), check_exact=True)
            except AssertionError as e:
                problems.append(f"{rel}: {e}")
        else:
            want, got = (expected / rel).read_text().splitlines(), (actual / rel).read_text().splitlines()
            if want != got:
                diff = difflib.unified_diff(want, got, "expected", "actual", lineterm="", n=1)
                problems.append(f"{rel}:\n" + "\n".join(list(diff)[:60]))
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    found = compare_dirs(Path(sys.argv[1]), Path(sys.argv[2]))
    print("\n\n".join(found) or "Outputs match.")
    sys.exit(1 if found else 0)
