"""Fit adaptive hierarchical code mappings on training event counts."""

from __future__ import annotations

import json
import polars as pl
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from meds import DataSchema
from MEDS_transforms.stages import Stage
from omegaconf import DictConfig, OmegaConf
from pathlib import Path


@dataclass(frozen=True)
class HierarchyProfile:
    """Adaptive truncation settings for one MEDS namespace."""

    minimum_count: int
    levels: tuple[int, ...]


def read_profiles(stage_cfg: DictConfig) -> dict[str, HierarchyProfile]:
    """Read hierarchy profiles keyed by MEDS namespace."""
    cfg = OmegaConf.to_container(stage_cfg, resolve=True)
    hierarchies = cfg["hierarchies"]
    default_minimum = int(cfg["minimum_count"])
    return {
        str(namespace): HierarchyProfile(
            minimum_count=int(hierarchy.get("minimum_count", default_minimum)),
            levels=tuple(sorted({int(level) for level in hierarchy["levels"]}, reverse=True)),
        )
        for namespace, hierarchy in sorted(hierarchies.items())
    }


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
    *,
    columns: Mapping[str, str],
) -> dict:
    return {
        DataSchema.code_name: code,
        columns["mapped_code"]: code if mapped_code is None else mapped_code,
        columns["count"]: count,
        columns["mapped_count"]: count if mapped_count is None else mapped_count,
        columns["profile"]: profile_name,
        columns["reason"]: reason,
    }


def resolve_pending_codes(
    counts: Mapping[str, int],
    profiles: Mapping[str, HierarchyProfile],
    pending_by_profile: Mapping[str, set[str]],
    candidates_by_code: Mapping[str, list[str]],
    records: Mapping[str, dict[str, object]],
    columns: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Resolve below-threshold codes against their candidate ancestors."""
    resolved_records = {code: record.copy() for code, record in records.items()}

    for profile_name, profile_pending in pending_by_profile.items():
        profile = profiles[profile_name]
        pending = set(profile_pending)

        by_candidate: dict[str, list[str]] = defaultdict(list)
        for code in sorted(pending):
            for candidate in candidates_by_code[code]:
                by_candidate[candidate].append(code)

        # Resolve longest candidates first so codes cannot later move to broader ancestors.
        for candidate in sorted(by_candidate, key=lambda c: (-len(c), c)):
            members = [code for code in by_candidate[candidate] if code in pending]
            # Include an observed target in its own aggregate and prevent further truncation.
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
                resolved_records[code] = make_record(
                    code,
                    int(counts[code]),
                    profile_name,
                    "grouped",
                    mapped_code=candidate,
                    mapped_count=mapped_count,
                    columns=columns,
                )
                pending.remove(code)
            if candidate_needs_own_count:
                resolved_records[candidate][columns["mapped_count"]] = mapped_count

        for code in sorted(pending):
            resolved_records[code] = make_record(code, int(counts[code]), profile_name, "below_threshold", columns=columns)

    return resolved_records


def fit_mapping(
    counts: Mapping[str, int],
    profiles: Mapping[str, HierarchyProfile],
    columns: Mapping[str, str],
) -> pl.DataFrame:
    """Fit a deterministic, disjoint adaptive hierarchy mapping."""
    records: dict[str, dict[str, object]] = {}
    pending_by_profile: dict[str, set[str]] = defaultdict(set)
    candidates_by_code: dict[str, list[str]] = {}

    for code, count in counts.items():
        count = int(count)
        parsed = split_code(code)
        profile_name = parsed[0] if parsed and parsed[0] in profiles else None
        if profile_name is None:
            records[code] = make_record(code, count, None, "unconfigured", columns=columns)
            continue

        profile = profiles[profile_name]
        if count >= profile.minimum_count:
            records[code] = make_record(code, count, profile_name, "retained", columns=columns)
            continue

        namespace, payload = parsed
        ancestors = [f"{namespace}//{candidate}" for candidate in candidate_ancestors(payload, profile)]
        if not ancestors:
            records[code] = make_record(code, count, profile_name, "no_hierarchy", columns=columns)
            continue

        candidates_by_code[code] = ancestors
        pending_by_profile[profile_name].add(code)

    records = resolve_pending_codes(counts, profiles, pending_by_profile, candidates_by_code, records, columns)

    schema = {
        DataSchema.code_name: pl.String,
        columns["mapped_code"]: pl.String,
        columns["count"]: pl.UInt64,
        columns["mapped_count"]: pl.UInt64,
        columns["profile"]: pl.String,
        columns["reason"]: pl.String,
    }
    return pl.DataFrame([records[code] for code in sorted(records)], schema=schema)


def combine_count_frames(
    *dfs: pl.DataFrame | pl.LazyFrame,
    columns: Mapping[str, str],
) -> dict[str, int]:
    """Sum mapped shard counts."""
    totals: dict[str, int] = defaultdict(int)
    for df in dfs:
        frame = df.collect() if isinstance(df, pl.LazyFrame) else df
        for code, count in frame.select(DataSchema.code_name, columns["count"]).iter_rows():
            totals[str(code)] += int(count)
    return dict(totals)


def summarize_mapping(mapping: pl.DataFrame, columns: Mapping[str, str]) -> dict[str, object]:
    """Return a compact, JSON-friendly audit of a fitted mapping."""
    changed = pl.col(DataSchema.code_name) != pl.col(columns["mapped_code"])
    training_rows = pl.col(columns["count"]) > 0
    totals = mapping.select(
        metadata_source_codes=pl.len(),
        training_source_codes=training_rows.sum(),
        output_codes=pl.col(columns["mapped_code"]).n_unique(),
        changed_source_codes=changed.sum(),
        training_events=pl.col(columns["count"]).sum(),
        remapped_training_events=pl.col(columns["count"]).filter(changed).sum(),
    ).to_dicts()[0]
    decisions = (
        mapping.group_by(
            profile=pl.col(columns["profile"]),
            reason=pl.col(columns["reason"]),
        )
        .agg(
            source_codes=pl.len(),
            training_events=pl.col(columns["count"]).sum(),
        )
        .sort("profile", "reason", nulls_last=True)
        .to_dicts()
    )
    return {
        "summary": totals,
        "decisions": decisions,
        "columns": {
            DataSchema.code_name: "original MEDS code",
            columns["mapped_code"]: "code used after adaptive truncation",
            columns["count"]: "raw events in training data",
            columns["mapped_count"]: "training events represented by the mapped code",
            columns["profile"]: "hierarchy used",
            columns["reason"]: "mapping decision",
        },
    }


def add_unseen_metadata_codes(
    mapping: pl.DataFrame,
    code_metadata: pl.DataFrame,
    columns: Mapping[str, str],
) -> pl.DataFrame:
    """Carry forward metadata codes absent from training without fitting them."""
    unseen_codes = sorted(
        set(code_metadata.get_column(DataSchema.code_name).to_list()) - set(mapping.get_column(DataSchema.code_name))
    )
    if not unseen_codes:
        return mapping
    unseen = pl.DataFrame(
        {
            DataSchema.code_name: unseen_codes,
            columns["mapped_code"]: unseen_codes,
            columns["count"]: [0] * len(unseen_codes),
            columns["mapped_count"]: [0] * len(unseen_codes),
            columns["profile"]: [None] * len(unseen_codes),
            columns["reason"]: ["unseen_training"] * len(unseen_codes),
        },
        schema=mapping.schema,
    )
    return pl.concat([mapping, unseen]).sort(DataSchema.code_name)


def mapper_fntr(stage_cfg: DictConfig) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Count training events per code; ``train_only`` selects the shards."""
    read_profiles(stage_cfg)
    count_column = stage_cfg.columns["count"]

    def mapper(df: pl.LazyFrame) -> pl.LazyFrame:
        return (
            df.group_by(DataSchema.code_name)
            .len()
            .select(
                pl.col(DataSchema.code_name),
                **{count_column: pl.col("len").cast(pl.UInt64)},
            )
            .sort(DataSchema.code_name)
        )

    return mapper


def reducer_fntr(stage_cfg: DictConfig) -> Callable[..., pl.LazyFrame]:
    """Fit one global mapping and write its reusable mapping and audit files."""
    profiles = read_profiles(stage_cfg)
    columns = stage_cfg.columns
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
        counts = combine_count_frames(*dfs, columns=columns)
        mapping = fit_mapping(counts, profiles=profiles, columns=columns)
        mapping = add_unseen_metadata_codes(mapping, code_metadata, columns)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        mapping.write_parquet(output_filepath)
        summary_filepath.parent.mkdir(parents=True, exist_ok=True)
        summary_filepath.write_text(
            json.dumps(summarize_mapping(mapping, columns), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        # Match the lazy type expected by MEDS-Transforms' metadata merge.
        return mapping.lazy()

    return reducer


stage = Stage.register(
    map_fn=mapper_fntr,
    reduce_fn=reducer_fntr,
    default_config=Path("configs/MEDS/default_adaptive_code_mapping.yaml"),
)
