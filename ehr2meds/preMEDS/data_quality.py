"""Summarize missing values in processed preMEDS tables."""

from collections import defaultdict
from pathlib import Path


class QualitySummary:
    """Aggregate null counts across processed chunks."""

    def __init__(self):
        self.results = defaultdict(lambda: [0, 0])

    def add(self, table, df):
        """Record null counts and total rows for each column."""
        for column in df.columns:
            counts = self.results[(table, column)]
            counts[0] += int(df[column].isna().sum())
            counts[1] += len(df)

    def merge(self, other):
        """Combine quality results from multiple workers."""
        for key, (nulls, total) in other.items():
            self.results[key][0] += nulls
            self.results[key][1] += total

    def to_dict(self):
        return dict(self.results)


def format_quality_summary(results):
    """Format columns containing null values."""
    rows = [
        (Path(table).name, column, nulls, total) for (table, column), (nulls, total) in sorted(results.items()) if nulls > 0
    ]

    if not rows:
        return "\nDATA QUALITY CHECK: No null values detected."

    lines = [
        "\nDATA QUALITY SUMMARY — preMEDS null values",
        f"{'Table':<30} {'Column':<25} {'Nulls':>12} {'Null %':>12}",
        "-" * 82,
    ]

    for table, column, nulls, total in rows:
        percentage = 100 * nulls / total if total else 0

        lines.append(f"{table:<30} {column:<25} {nulls:>12,} {percentage:>11.2f}%")

    return "\n".join(lines)
