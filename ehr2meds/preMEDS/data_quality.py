"""Track row changes and missing values during preMEDS processing."""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

STAGES = (
    "Table mappings",
    "Subject ID mapping",
    "Mandatory-column filtering",
    "Deduplication",
)

@dataclass
class DataQualityTracker:
    """Track row-count changes for one processed chunk."""

    initial_rows: int
    current_rows: int = field(init=False)
    stage_changes: dict = field(default_factory=lambda: defaultdict(int))
    failed: bool = False

    def __post_init__(self):
        self.current_rows = self.initial_rows

    def checkpoint(self, stage: str, df):
        """Record the net change since the previous checkpoint."""
        if self.failed:
            return

        try:
            new_count = len(df)
            self.record_change(stage, new_count - self.current_rows)
        except Exception:
            self.failed = True
            logger.exception(
                "Data-quality tracking failed at stage '%s'.", stage
            )

    def record_change(self, stage: str, delta: int):
        """Record a known row-count change."""
        if self.failed:
            return

        try:
            self.stage_changes[stage] += delta
            self.current_rows += delta
        except Exception:
            self.failed = True
            logger.exception(
                "Data-quality tracking failed while recording stage '%s'.",
                stage,
            )

    def result(self):
        """Return statistics, or None if tracking failed."""
        if self.failed:
            return None

        try:
            return {
                "input": self.initial_rows,
                "output": self.current_rows,
                "stages": dict(self.stage_changes),
            }
        except Exception:
            self.failed = True
            logger.exception("Could not retrieve data-quality tracking results.")
            return None

class DataQualitySummary:
    """Aggregate row changes and null counts across chunks and workers."""

    def __init__(self):
        self.rows = defaultdict(
            lambda: {
                "input": 0,
                "output": 0,
                "stages": defaultdict(int),
            }
        )
        self.nulls = defaultdict(lambda: [0, 0])

    def add(self, table: str, row_result: dict, df):
        """Add row statistics and null counts from one processed chunk."""
        entry = self.rows[table]
        entry["input"] += row_result["input"]
        entry["output"] += row_result["output"]

        for stage, delta in row_result["stages"].items():
            entry["stages"][stage] += delta

        for column in df.columns:
            counts = self.nulls[(table, column)]
            counts[0] += int(df[column].isna().sum())
            counts[1] += len(df)

    def merge(self, other_results: dict):
        """Merge results returned by a multiprocessing worker."""
        for table, result in other_results["rows"].items():
            entry = self.rows[table]
            entry["input"] += result["input"]
            entry["output"] += result["output"]

            for stage, delta in result["stages"].items():
                entry["stages"][stage] += delta

        for key, (nulls, total) in other_results["nulls"].items():
            counts = self.nulls[key]
            counts[0] += nulls
            counts[1] += total

    def to_dict(self):
        """Return serializable copies of the aggregated results."""
        return {
            "rows": {
                table: {
                    "input": data["input"],
                    "output": data["output"],
                    "stages": dict(data["stages"]),
                }
                for table, data in self.rows.items()
            },
            "nulls": dict(self.nulls),
        }
    
    @classmethod
    def get_worker_summaries(cls, worker_results):
        """Combine worker summaries into one final result."""
        summary = cls()

        for worker_result in worker_results:
            summary.merge(worker_result)

        return summary.to_dict()


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


def format_table(headers, rows):
    """Format table columns with consistent spacing and alignment."""
    widths = [
        max(
            len(str(header)),
            *(len(str(row[i])) for row in rows),
        )
        for i, header in enumerate(headers)
    ]

    def format_row(row):
        return "  ".join(
            f"{str(value):<{widths[i]}}" if i == 0 else f"{str(value):>{widths[i]}}" for i, value in enumerate(row)
        )

    table_width = sum(widths) + 2 * (len(headers) - 1)

    lines = [
        format_row(headers),
        "-" * table_width,
    ]

    lines.extend(format_row(row) for row in rows)

    return "\n".join(lines)


def format_final_summary(results: dict) -> str:
    """Build the final raw-to-preMEDS row summary."""
    if not results:
        return "1. ROW SUMMARY\n--------------\nNo processing results recorded."

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

    lines = [
        "1. ROW SUMMARY",
        "--------------",
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
        format_table(headers, rows),
    ]

    return "\n".join(lines)


def format_quality_summary(results: dict) -> str:
    """Format columns containing null values."""
    rows = [
        [
            Path(table).name,
            column,
            f"{nulls:,}",
            f"{100 * nulls / total if total else 0:.2f}%",
        ]
        for (table, column), (nulls, total) in sorted(results.items())
        if nulls > 0
    ]

    lines = [
        "2. NULL SUMMARY",
        "---------------",
        "",
    ]

    if not rows:
        lines.append("No null values detected.")
    else:
        headers = [
            "Table",
            "Column",
            "Nulls",
            "Null %",
        ]
        lines.append(format_table(headers, rows))

    return "\n".join(lines)


def format_data_quality_summary(results: dict) -> str:
    """Combine row and null summaries into one final report."""
    row_summary = format_final_summary(results["rows"])
    null_summary = format_quality_summary(results["nulls"])

    table_width = max(
        80,
        max(
            (len(line) for line in row_summary.splitlines()),
            default=0,
        ),
        max(
            (len(line) for line in null_summary.splitlines()),
            default=0,
        ),
    )

    separator = "=" * table_width

    return "\n".join(
        [
            separator,
            "RAW → preMEDS DATA QUALITY SUMMARY",
            separator,
            "",
            row_summary,
            "",
            null_summary,
            "",
            separator,
        ]
    )
