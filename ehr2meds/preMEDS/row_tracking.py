from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


STAGES = {
    "Table mappings": "Mappings",
    "Subject ID mapping": "Subject ID",
    "Mandatory-column filtering": "Mandatory",
    "Deduplication": "Duplicates",
}


@dataclass
class RowTracker:
    """Track row-count changes for one processed chunk."""

    initial_rows: int
    current_rows: int = field(init=False)
    stage_changes: dict = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        self.current_rows = self.initial_rows

    def checkpoint(self, stage: str, df):
        """Record the net change since the previous checkpoint."""
        new_count = len(df)
        self.record_change(stage, new_count - self.current_rows)

    def record_change(self, stage: str, delta: int):
        """Record a known row-count change."""
        self.stage_changes[stage] += delta
        self.current_rows += delta

    def result(self):
        """Return serializable statistics for this chunk."""
        return {
            "input": self.initial_rows,
            "output": self.current_rows,
            "stages": dict(self.stage_changes),
        }


class RowSummary:
    """Aggregate chunk statistics into one result per table."""

    def __init__(self):
        self.results = defaultdict(
            lambda: {
                "input": 0,
                "output": 0,
                "stages": defaultdict(int),
            }
        )

    def add(self, table: str, result: dict):
        """Add statistics from one chunk or an aggregated worker result."""
        entry = self.results[table]
        entry["input"] += result["input"]
        entry["output"] += result["output"]

        for stage, delta in result["stages"].items():
            entry["stages"][stage] += delta

    def merge(self, other_results: dict):
        """Merge results returned by a multiprocessing worker."""
        for table, result in other_results.items():
            self.add(table, result)

    def to_dict(self):
        """Return a serializable copy of the aggregated results."""
        return {
            table: {
                "input": data["input"],
                "output": data["output"],
                "stages": dict(data["stages"]),
            }
            for table, data in self.results.items()
        }


def format_change(value):
    """Use + for positive, - for negative, and no sign for zero."""
    return f"{value:+,}" if value else "0"


def format_contribution(delta, total_change):
    """Show a stage's contribution to the net row loss."""
    formatted_delta = format_change(delta)

    if delta == 0:
        return "0 (0.0%)"

    if total_change >= 0:
        return f"{formatted_delta} (—)"

    percentage = 100 * delta / total_change

    if round(percentage, 1) == 0:
        percentage = 0.0

    return f"{formatted_delta} ({percentage:.1f}%)"


def format_final_summary(results: dict) -> str:
    """Build the final raw-to-preMEDS summary as a string."""
    if not results:
        return "No processing results recorded."

    headers = [
        "Table",
        "Raw",
        "preMEDS",
        "Mappings",
        "Subject ID",
        "Mandatory",
        "Duplicates",
        "Total change",
        "Loss %",
    ]

    rows = []

    for table, data in sorted(results.items()):
        raw = data["input"]
        premeds = data["output"]
        stages = data["stages"]

        total_change = premeds - raw
        loss_pct = 100 * (raw - premeds) / raw if raw else 0.0

        rows.append(
            [
                Path(table).name,
                f"{raw:,}",
                f"{premeds:,}",
                *[format_contribution(stages.get(stage, 0), total_change) for stage in STAGES],
                format_change(total_change),
                f"{loss_pct:.2f}%",
            ]
        )

    widths = [max(len(str(value)) for value in [header] + [row[i] for row in rows]) for i, header in enumerate(headers)]

    def format_row(row):
        return "  ".join(f"{value:<{widths[i]}}" if i == 0 else f"{value:>{widths[i]}}" for i, value in enumerate(row))

    table_width = sum(widths) + 2 * (len(headers) - 1)

    lines = [
        "=" * table_width,
        "RAW → preMEDS SUMMARY",
        "=" * table_width,
        "",
        "COLUMN EXPLANATIONS",
        "-------------------",
        "Table        : Input filename being processed.",
        "Raw          : Rows entering the processor.",
        "preMEDS      : Rows after processing.",
        "Mappings     : Net row change from table mappings, including joins.",
        "Subject ID   : Rows removed by subject ID mapping.",
        "Mandatory    : Rows removed due to null mandatory columns.",
        "Duplicates   : Duplicate rows removed, excluding row_idx.",
        "Total change : preMEDS minus Raw.",
        "Loss %       : Net row loss as a percentage of Raw.",
        "",
        "Stage percentages show contributions to net row loss.",
        "Positive changes add rows; negative changes remove rows.",
        "",
        format_row(headers),
        "-" * table_width,
    ]

    lines.extend(format_row(row) for row in rows)
    lines.append("=" * table_width)

    return "\n".join(lines)
