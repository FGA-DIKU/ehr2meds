"""Plot date windows per patient and tokens per window chunk for natural-language EHR text.

Example:
    python -m ehr2meds.visualize_stuff.visualize_chunks_pr_patient \
        --data_dir /path/to/natural_language/pre_training/raw_data \
        --tokenizer /path/to/local/tokenizer_dir \
        --output chunks_pr_patient.png

The tokenizer is loaded from local files only, so this runs on an offline cluster: pass a directory
created with `AutoTokenizer.from_pretrained(name).save_pretrained(dir)`, or a Hub name that is
already in the local Hugging Face cache.
"""

import argparse
import matplotlib.pyplot as plt
import polars as pl
import re
from pathlib import Path

WINDOW_REGEX = r"^### Time"

WINDOW_BINS = [
    (0, 0, "0"),
    (1, 2, "1–2"),
    (3, 4, "3–4"),
    (5, 6, "5–6"),
    (7, 10, "7–10"),
    (11, 20, "11–20"),
    (21, 50, "21–50"),
    (51, 100, "51–100"),
    (101, 199, "101–199"),
    (200, None, "200+"),
]

TOKEN_BREAKS = [512, 1024, 2048, 4096, 8192, 16384, 32768]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=Path, required=True, help="Directory containing one subdirectory per split.")
    parser.add_argument("--splits", nargs="+", default=["train", "tuning"])
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer directory, or a Hub name already in the HF cache.")
    parser.add_argument("--text_col", default="ehr_text")
    parser.add_argument("--subject_col", default="subject_id")
    parser.add_argument("--window_regex", default=WINDOW_REGEX)
    parser.add_argument("--max_files", type=int, default=None, help="Read only the first N files per split (debugging).")
    parser.add_argument("--batch_size", type=int, default=1000, help="Chunks per tokenizer call.")
    parser.add_argument("--output", type=Path, default=None, help="Save the figure here instead of showing it.")
    return parser.parse_args()


def load_data(data_dir: Path, splits: list[str], max_files: int | None, columns: list[str]) -> pl.DataFrame:
    frames = []
    for split in splits:
        paths = sorted((data_dir / split).glob("*.parquet"))
        if max_files is not None:
            paths = paths[:max_files]
        if not paths:
            raise FileNotFoundError(f"No parquet files found in {data_dir / split}")
        print(f"{split}: reading {len(paths)} file(s)")
        frame = pl.concat([pl.read_parquet(path, columns=columns) for path in paths])
        print(split, frame.shape)
        frames.append(frame)
    return pl.concat(frames)


def split_into_windows(text: str | None, window_re: re.Pattern) -> list[str]:
    """Split text at each window marker; any text before the first marker is kept with the first window."""
    if not text:
        return []
    starts = [match.start() for match in window_re.finditer(text)]
    if not starts:
        return []
    starts[0] = 0
    ends = starts[1:] + [len(text)]
    return [text[start:end] for start, end in zip(starts, ends)]


def build_chunks(df: pl.DataFrame, text_col: str, subject_col: str, window_re: re.Pattern) -> pl.DataFrame:
    return (
        df.select(
            pl.col(subject_col),
            pl.col(text_col)
            .map_elements(lambda text: split_into_windows(text, window_re), return_dtype=pl.List(pl.String))
            .alias("chunk"),
        )
        .explode("chunk")
        .drop_nulls("chunk")
    )


def count_tokens(texts: list[str], tokenizer_name: str, batch_size: int) -> list[int]:
    from tqdm import tqdm
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    n_tokens = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
        encoded = tokenizer(texts[i : i + batch_size], add_special_tokens=False)["input_ids"]
        n_tokens.extend(len(ids) for ids in encoded)
    return n_tokens


def bin_label(n: int) -> str:
    for low, high, label in WINDOW_BINS:
        if n >= low and (high is None or n <= high):
            return label
    raise ValueError(f"No bin for {n}")


def plot(per_patient: pl.DataFrame, chunks: pl.DataFrame, window_regex: str, output: Path | None) -> None:
    window_labels = [label for _, _, label in WINDOW_BINS]
    window_dist = per_patient.with_columns(
        pl.col("n_windows").map_elements(bin_label, return_dtype=pl.String).alias("bin")
    )["bin"].value_counts()
    window_heights = [window_dist.filter(pl.col("bin") == label)["count"].sum() for label in window_labels]

    token_labels = [f"≤{b}" for b in TOKEN_BREAKS] + [f">{TOKEN_BREAKS[-1]}"]
    token_dist = chunks.select(pl.col("n_tokens").cut(TOKEN_BREAKS, labels=token_labels).alias("bin"))[
        "bin"
    ].value_counts()
    token_heights = [token_dist.filter(pl.col("bin") == label)["count"].sum() for label in token_labels]

    # Cumulative share for the "≤" bins, plain share for the overflow bin.
    pct = []
    running = 0
    for height in token_heights[:-1]:
        running += height
        pct.append(100 * running / chunks.height)
    pct.append(100 * token_heights[-1] / chunks.height)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout="constrained")
    axes[0].bar(window_labels, window_heights)
    axes[0].set_xlabel(f"Windows per patient ({window_regex})")
    axes[0].set_ylabel("Number of patients")
    axes[0].set_title(f"Windows per patient (n={per_patient.height})")
    axes[0].tick_params(axis="x", labelsize=8)

    bars = axes[1].bar(token_labels, token_heights)
    axes[1].set_xlabel(f"Tokens per window chunk ({window_regex})")
    axes[1].set_ylabel("Number of chunks")
    axes[1].set_title(f"Chunk length (n={chunks.height})")
    axes[1].set_ylim(0, max(token_heights) * 1.15)
    axes[1].tick_params(axis="x", labelrotation=30, labelsize=8)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")
    for bar, share in zip(bars, pct):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{share:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)
        print(f"Saved figure to {output}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    window_re = re.compile(args.window_regex, flags=re.MULTILINE)

    df = load_data(args.data_dir, args.splits, args.max_files, [args.subject_col, args.text_col])
    chunks = build_chunks(df, args.text_col, args.subject_col, window_re)
    chunks = chunks.with_columns(
        pl.Series("n_tokens", count_tokens(chunks["chunk"].to_list(), args.tokenizer, args.batch_size), dtype=pl.Int64)
    )

    # Patients with no windows have no chunks, so count windows from the full patient table.
    per_patient = df.group_by(args.subject_col).agg(
        pl.col(args.text_col).str.count_matches(f"(?m){args.window_regex}").fill_null(0).sum().alias("n_windows")
    )

    print(
        chunks.select(
            pl.len().alias("chunks"),
            pl.col("n_tokens").min().alias("min"),
            pl.col("n_tokens").median().alias("median"),
            pl.col("n_tokens").quantile(0.95).alias("p95"),
            pl.col("n_tokens").quantile(0.99).alias("p99"),
            pl.col("n_tokens").max().alias("max"),
        )
    )
    plot(per_patient, chunks, args.window_regex, args.output)


if __name__ == "__main__":
    main()
