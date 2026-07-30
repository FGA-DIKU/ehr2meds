"""Fit adaptive hierarchical code mappings on training event counts."""

from __future__ import annotations

import json
import polars as pl
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from ehr2meds.adaptive_code_mapping import (
    COUNT_COLUMN,
    MAPPED_CODE_COLUMN,
    MAPPED_COUNT_COLUMN,
    PROFILE_COLUMN,
    REASON_COLUMN,
)
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


@dataclass(frozen=True)
class HierarchyProfile:
    """How coarse a code is allowed to grow (``levels``), and by how much at a time."""

    minimum_count: int
    levels: tuple[int, ...]


def read_profiles(stage_cfg: DictConfig) -> tuple[dict[str, HierarchyProfile], dict[str, str]]:
    """Build the configured hierarchy profiles and namespace routing.

    ``hierarchies`` supplies every profile's character-position levels (see
    ``configs/MEDS/default_adaptive_code_mapping.yaml`` for the built-in ATC
    and SKS definitions). A pipeline config can override a single field of
    one profile, or add a new profile, without repeating the rest --
    Hydra/OmegaConf merges the pipeline's ``hierarchies`` onto the stage
    default field by field.
    """
    cfg = OmegaConf.to_container(stage_cfg, resolve=True)
    namespaces = {str(namespace): str(profile) for namespace, profile in cfg["namespaces"].items()}
    hierarchies = cfg["hierarchies"]
    default_minimum = int(cfg["minimum_count"])
    profiles = {
        name: HierarchyProfile(
            minimum_count=int(hierarchies[name].get("minimum_count", default_minimum)),
            levels=tuple(sorted({int(level) for level in hierarchies[name]["levels"]}, reverse=True)),
        )
        for name in sorted(set(namespaces.values()))
    }
    return profiles, namespaces


def split_code(code: str) -> tuple[str, str] | None:
    """Split a MEDS code into its first namespace and remaining payload."""
    namespace, separator, payload = code.partition("//")
    if not separator or not namespace or not payload:
        return None
    return namespace, payload


def candidate_ancestors(payload: str, profile: HierarchyProfile) -> list[str]:
    """Return nearest-to-broadest candidate ancestor payloads."""
    normalized = payload.strip().upper().replace(".", "")
    return [normalized[:length] for length in profile.levels if length < len(normalized)]


def make_record(
    code: str,
    count: int,
    profile_name: str | None,
    reason: str,
    mapped_code: str | None = None,
    mapped_count: int | None = None,
) -> dict:
    return {
        DataSchema.code_name: code,
        MAPPED_CODE_COLUMN: code if mapped_code is None else mapped_code,
        COUNT_COLUMN: count,
        MAPPED_COUNT_COLUMN: count if mapped_count is None else mapped_count,
        PROFILE_COLUMN: profile_name,
        REASON_COLUMN: reason,
    }


def fit_mapping(
    counts: Mapping[str, int],
    profiles: Mapping[str, HierarchyProfile],
    namespaces: Mapping[str, str],
) -> pl.DataFrame:
    """Fit a deterministic, disjoint adaptive hierarchy mapping."""
    records: dict[str, dict[str, object]] = {}
    pending_by_profile: dict[str, set[str]] = defaultdict(set)
    candidates_by_code: dict[str, list[str]] = {}

    for code in sorted(counts):
        count = int(counts[code])
        parsed = split_code(code)
        profile_name = namespaces.get(parsed[0]) if parsed else None
        if profile_name is None:
            records[code] = make_record(code, count, None, "unconfigured")
            continue

        profile = profiles[profile_name]
        if count >= profile.minimum_count:
            records[code] = make_record(code, count, profile_name, "retained")
            continue

        namespace, payload = parsed
        ancestors = [f"{namespace}//{candidate}" for candidate in candidate_ancestors(payload, profile)]
        if not ancestors:
            records[code] = make_record(code, count, profile_name, "no_hierarchy")
            continue

        candidates_by_code[code] = ancestors
        pending_by_profile[profile_name].add(code)

    for profile_name in sorted(pending_by_profile):
        profile = profiles[profile_name]
        pending = pending_by_profile[profile_name]

        by_candidate: dict[str, list[str]] = defaultdict(list)
        for code in sorted(pending):
            for candidate in candidates_by_code[code]:
                by_candidate[candidate].append(code)

        # Try candidates most-specific (longest) first. A code already
        # resolved by a longer candidate is skipped once a shorter one comes up.
        for candidate in sorted(by_candidate, key=lambda c: (-len(c), c)):
            members = [code for code in by_candidate[candidate] if code in pending]
            # If the target itself is observed, its events belong to the
            # target's denominator. Claim it too so it can't later move
            # farther upward.
            candidate_is_pending = candidate in pending
            if candidate_is_pending:
                members.append(candidate)

            mapped_count = sum(int(counts[code]) for code in members)
            candidate_needs_own_count = candidate in counts and not candidate_is_pending
            if candidate_needs_own_count:
                mapped_count += int(counts[candidate])

            if mapped_count < profile.minimum_count:
                continue
            for code in members:
                records[code] = make_record(
                    code, int(counts[code]), profile_name, "grouped", mapped_code=candidate, mapped_count=mapped_count
                )
                pending.remove(code)
            if candidate_needs_own_count:
                records[candidate][MAPPED_COUNT_COLUMN] = mapped_count

        for code in sorted(pending):
            records[code] = make_record(code, int(counts[code]), profile_name, "below_threshold")

    schema = {
        DataSchema.code_name: pl.String,
        MAPPED_CODE_COLUMN: pl.String,
        COUNT_COLUMN: pl.UInt64,
        MAPPED_COUNT_COLUMN: pl.UInt64,
        PROFILE_COLUMN: pl.String,
        REASON_COLUMN: pl.String,
    }
    return pl.DataFrame([records[code] for code in sorted(records)], schema=schema)


def combine_count_frames(*dfs: pl.DataFrame | pl.LazyFrame) -> dict[str, int]:
    """Sum mapped shard counts."""
    totals: dict[str, int] = defaultdict(int)
    for df in dfs:
        frame = df.collect() if isinstance(df, pl.LazyFrame) else df
        for code, count in frame.select(DataSchema.code_name, COUNT_COLUMN).iter_rows():
            totals[str(code)] += int(count)
    return dict(totals)


def summarize_mapping(mapping: pl.DataFrame) -> dict[str, object]:
    """Return a compact, JSON-friendly audit of a fitted mapping."""
    changed = pl.col(DataSchema.code_name) != pl.col(MAPPED_CODE_COLUMN)
    training_rows = pl.col(COUNT_COLUMN) > 0
    totals = mapping.select(
        pl.len().alias("metadata_source_codes"),
        training_rows.sum().alias("training_source_codes"),
        pl.col(MAPPED_CODE_COLUMN).n_unique().alias("output_codes"),
        changed.sum().alias("changed_source_codes"),
        pl.col(COUNT_COLUMN).sum().alias("training_events"),
        pl.col(COUNT_COLUMN).filter(changed).sum().alias("remapped_training_events"),
    ).to_dicts()[0]
    decisions = (
        mapping.group_by(
            pl.col(PROFILE_COLUMN).alias("profile"),
            pl.col(REASON_COLUMN).alias("reason"),
        )
        .agg(
            pl.len().alias("source_codes"),
            pl.col(COUNT_COLUMN).sum().alias("training_events"),
        )
        .sort("profile", "reason", nulls_last=True)
        .to_dicts()
    )
    return {
        "summary": totals,
        "decisions": decisions,
        "columns": {
            DataSchema.code_name: "original MEDS code",
            MAPPED_CODE_COLUMN: "code used after adaptive truncation",
            COUNT_COLUMN: "raw events in training data",
            MAPPED_COUNT_COLUMN: "training events represented by the mapped code",
            PROFILE_COLUMN: "hierarchy used",
            REASON_COLUMN: "mapping decision",
        },
    }


def add_unseen_metadata_codes(mapping: pl.DataFrame, code_metadata: pl.DataFrame) -> pl.DataFrame:
    """Carry forward metadata codes absent from training without fitting them."""
    unseen_codes = sorted(set(code_metadata.get_column(DataSchema.code_name).to_list()) - set(mapping.get_column(DataSchema.code_name)))
    if not unseen_codes:
        return mapping
    unseen = pl.DataFrame(
        {
            DataSchema.code_name: unseen_codes,
            MAPPED_CODE_COLUMN: unseen_codes,
            COUNT_COLUMN: [0] * len(unseen_codes),
            MAPPED_COUNT_COLUMN: [0] * len(unseen_codes),
            PROFILE_COLUMN: [None] * len(unseen_codes),
            REASON_COLUMN: ["unseen_training"] * len(unseen_codes),
        },
        schema=mapping.schema,
    )
    return pl.concat([mapping, unseen]).sort(DataSchema.code_name)


def mapper_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Count training events per code; ``train_only`` selects the shards."""
    read_profiles(stage_cfg)

    def mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.group_by(DataSchema.code_name)
            .len()
            .select(
                pl.col(DataSchema.code_name),
                pl.col("len").cast(pl.UInt64).alias(COUNT_COLUMN),
            )
            .sort(DataSchema.code_name)
        )

    return mapper


def reducer_fntr(stage_cfg: DictConfig) -> Callable[..., pl.LazyFrame]:
    """Fit one global mapping and write its reusable mapping and audit files."""
    profiles, namespaces = read_profiles(stage_cfg)
    configured_output = stage_cfg.get("mapping_output_filepath")
    output_filepath = (
        Path(str(configured_output))
        if configured_output
        else Path(str(stage_cfg.reducer_output_dir)) / "adaptive_code_mapping.parquet"
    )
    configured_summary = stage_cfg.get("mapping_summary_output_filepath")
    if configured_summary:
        summary_filepath = Path(str(configured_summary))
    else:
        summary_filepath = output_filepath.with_suffix(".summary.json")
    code_metadata_filepath = Path(str(stage_cfg.metadata_input_dir)) / "codes.parquet"
    code_metadata = (
        pl.read_parquet(code_metadata_filepath)
        if code_metadata_filepath.is_file()
        else pl.DataFrame(schema={DataSchema.code_name: pl.String})
    )

    def reducer(*dfs: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
        counts = combine_count_frames(*dfs)
        mapping = fit_mapping(counts, profiles=profiles, namespaces=namespaces)
        mapping = add_unseen_metadata_codes(mapping, code_metadata)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        mapping.write_parquet(output_filepath)
        summary_filepath.parent.mkdir(parents=True, exist_ok=True)
        summary_filepath.write_text(
            json.dumps(summarize_mapping(mapping), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        # MEDS-Transforms scans prior code metadata lazily, and its merge helper
        # requires both sides of the join to use the same eager/lazy type.
        return mapping.lazy()

    return reducer


stage = Stage.register(
    map_fn=mapper_fntr,
    reduce_fn=reducer_fntr,
    default_config=Path("configs/MEDS/default_adaptive_code_mapping.yaml"),
)
