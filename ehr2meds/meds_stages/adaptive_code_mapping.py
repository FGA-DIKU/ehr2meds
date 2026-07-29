"""Shared adaptive hierarchical code-mapping logic."""

from __future__ import annotations

import polars as pl
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from meds import DataSchema
from MEDS_transforms.utils import PKG_PFX, resolve_pkg_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path

CODE_COLUMN = DataSchema.code_name
MAPPED_CODE_COLUMN = "adaptive/mapped_code"
COUNT_COLUMN = "adaptive/count"
MAPPED_COUNT_COLUMN = "adaptive/mapped_count"
PROFILE_COLUMN = "adaptive/profile"
REASON_COLUMN = "adaptive/reason"
MEMBER_COUNT_COLUMN = "adaptive/member_count"

# Character-position level widths for the code systems supported out of the
# box. A config normally only needs to route MEDS namespaces to these names.
BUILTIN_HIERARCHIES: dict[str, dict[str, object]] = {
    "sks_diagnosis": {
        "levels": [4, 3, 2, 1],
    },
    "sks_operation": {
        "levels": [7, 6, 5, 4, 3, 2, 1],
    },
    "sks_other_procedure": {
        "levels": [8, 7, 6, 5, 4, 3, 2, 1],
    },
    "atc": {
        "levels": [1, 3, 4, 5, 7],
    },
}


@dataclass(frozen=True)
class HierarchyProfile:
    """Validated configuration for one hierarchy."""

    name: str
    minimum_count: int
    levels: tuple[int, ...] = ()
    minimum_canonical_length: int = 1
    uppercase: bool = True
    remove_dots: bool = True


def _plain(value: object) -> object:
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def read_profiles(stage_cfg: DictConfig) -> tuple[dict[str, HierarchyProfile], dict[str, str]]:
    """Build the referenced built-in/custom profiles and validate routing.

    ``hierarchies`` is optional. Entries override built-ins with the same name
    or define a new custom profile.
    """
    raw_overrides = _plain(stage_cfg.get("hierarchies", {}))
    raw_namespaces = _plain(stage_cfg.get("namespaces", {}))
    if not isinstance(raw_overrides, dict):
        raise TypeError("hierarchies must be a mapping")
    if not isinstance(raw_namespaces, dict) or not raw_namespaces:
        raise ValueError("namespaces must be a non-empty mapping")

    default_minimum = int(stage_cfg.get("minimum_count", 100))
    if default_minimum < 1:
        raise ValueError("minimum_count must be at least 1")

    namespaces = {str(namespace): str(profile) for namespace, profile in raw_namespaces.items()}
    requested_profiles = set(namespaces.values())
    undefined = sorted(requested_profiles - set(BUILTIN_HIERARCHIES) - set(raw_overrides))
    if undefined:
        raise ValueError(f"namespaces reference undefined hierarchy profiles: {undefined}")

    profiles: dict[str, HierarchyProfile] = {}
    for name in sorted(requested_profiles):
        raw = dict(BUILTIN_HIERARCHIES.get(name, {}))
        override = raw_overrides.get(name, {})
        if not isinstance(override, dict):
            raise TypeError(f"hierarchies[{name!r}] must be a mapping")
        raw.update(override)

        levels = tuple(sorted({int(level) for level in raw.get("levels", [])}, reverse=True))
        if not levels or min(levels) < 1:
            raise ValueError(f"hierarchies[{name!r}].levels must contain positive integers")

        minimum_count = int(raw.get("minimum_count", default_minimum))
        if minimum_count < 1:
            raise ValueError(f"hierarchies[{name!r}].minimum_count must be at least 1")

        minimum_length = int(raw.get("minimum_canonical_length", 1))
        if minimum_length < 1:
            raise ValueError(f"hierarchies[{name!r}].minimum_canonical_length must be at least 1")

        profiles[str(name)] = HierarchyProfile(
            name=str(name),
            minimum_count=minimum_count,
            levels=levels,
            minimum_canonical_length=minimum_length,
            uppercase=bool(raw.get("uppercase", True)),
            remove_dots=bool(raw.get("remove_dots", True)),
        )

    if any("//" in namespace or not namespace for namespace in namespaces):
        raise ValueError("namespace names must be non-empty and cannot contain '//'")
    return profiles, namespaces


def resolve_resource_path(filepath: str) -> Path:
    """Resolve normal and package resource paths."""
    return resolve_pkg_path(filepath) if filepath.startswith(PKG_PFX) else Path(filepath)


def load_frame(filepath: str, label: str) -> pl.DataFrame:
    """Load a JSON or Parquet mapping resource."""
    path = resolve_resource_path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"{label} filepath '{filepath}' does not exist")
    match path.suffix.lower():
        case ".parquet":
            return pl.read_parquet(path)
        case ".json":
            return pl.read_json(path)
        case _:
            raise ValueError(f"{label} filepath must point to a JSON or Parquet file")


def normalize_payload(payload: str, profile: HierarchyProfile) -> str:
    """Normalize a payload for hierarchy lookup."""
    result = payload.strip()
    if profile.uppercase:
        result = result.upper()
    if profile.remove_dots:
        result = result.replace(".", "")
    return result


def split_code(code: str) -> tuple[str, str] | None:
    """Split a MEDS code into its first namespace and remaining payload."""
    namespace, separator, payload = code.partition("//")
    if not separator or not namespace or not payload:
        return None
    return namespace, payload


def candidate_ancestors(payload: str, profile: HierarchyProfile) -> list[str]:
    """Return nearest-to-broadest candidate ancestor payloads."""
    normalized = normalize_payload(payload, profile)
    return [
        normalized[:length]
        for length in profile.levels
        if length < len(normalized) and length >= profile.minimum_canonical_length
    ]


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
            records[code] = {
                CODE_COLUMN: code,
                MAPPED_CODE_COLUMN: code,
                COUNT_COLUMN: count,
                MAPPED_COUNT_COLUMN: count,
                PROFILE_COLUMN: None,
                REASON_COLUMN: "unconfigured",
            }
            continue

        profile = profiles[profile_name]
        if count >= profile.minimum_count:
            records[code] = {
                CODE_COLUMN: code,
                MAPPED_CODE_COLUMN: code,
                COUNT_COLUMN: count,
                MAPPED_COUNT_COLUMN: count,
                PROFILE_COLUMN: profile_name,
                REASON_COLUMN: "retained",
            }
            continue

        namespace, payload = parsed
        ancestors = [f"{namespace}//{candidate}" for candidate in candidate_ancestors(payload, profile)]
        if not ancestors:
            records[code] = {
                CODE_COLUMN: code,
                MAPPED_CODE_COLUMN: code,
                COUNT_COLUMN: count,
                MAPPED_COUNT_COLUMN: count,
                PROFILE_COLUMN: profile_name,
                REASON_COLUMN: "no_hierarchy",
            }
            continue
        candidates_by_code[code] = ancestors
        pending_by_profile[profile_name].add(code)

    for profile_name in sorted(pending_by_profile):
        profile = profiles[profile_name]
        pending = pending_by_profile[profile_name]
        lengths = sorted(
            {len(candidate) for code in pending for candidate in candidates_by_code[code]},
            reverse=True,
        )
        for length in lengths:
            groups: dict[str, list[str]] = defaultdict(list)
            for code in sorted(pending):
                for candidate in candidates_by_code[code]:
                    if len(candidate) == length:
                        groups[candidate].append(code)
                        break

            for candidate in sorted(groups):
                members = [code for code in groups[candidate] if code in pending]
                # If the target is itself observed, its events belong to the
                # target denominator. Claim a pending target with its
                # descendants so it cannot subsequently move farther upward.
                if candidate in pending:
                    members.append(candidate)

                mapped_count = sum(int(counts[code]) for code in members)
                exact_target_already_assigned = candidate in counts and candidate not in set(members)
                if exact_target_already_assigned:
                    mapped_count += int(counts[candidate])

                if mapped_count < profile.minimum_count:
                    continue
                for code in members:
                    records[code] = {
                        CODE_COLUMN: code,
                        MAPPED_CODE_COLUMN: candidate,
                        COUNT_COLUMN: int(counts[code]),
                        MAPPED_COUNT_COLUMN: mapped_count,
                        PROFILE_COLUMN: profile_name,
                        REASON_COLUMN: "grouped",
                    }
                    pending.remove(code)
                if exact_target_already_assigned and candidate in records:
                    records[candidate][MAPPED_COUNT_COLUMN] = mapped_count

        for code in sorted(pending):
            records[code] = {
                CODE_COLUMN: code,
                MAPPED_CODE_COLUMN: code,
                COUNT_COLUMN: int(counts[code]),
                MAPPED_COUNT_COLUMN: int(counts[code]),
                PROFILE_COLUMN: profile_name,
                REASON_COLUMN: "below_threshold",
            }

    schema = {
        CODE_COLUMN: pl.String,
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
        for code, count in frame.select(CODE_COLUMN, COUNT_COLUMN).iter_rows():
            totals[str(code)] += int(count)
    return dict(totals)


def summarize_mapping(mapping: pl.DataFrame) -> dict[str, object]:
    """Return a compact, JSON-friendly audit of a fitted mapping."""
    changed = pl.col(CODE_COLUMN) != pl.col(MAPPED_CODE_COLUMN)
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
            CODE_COLUMN: "original MEDS code",
            MAPPED_CODE_COLUMN: "code used after adaptive truncation",
            COUNT_COLUMN: "raw events in training data",
            MAPPED_COUNT_COLUMN: "training events represented by the mapped code",
            PROFILE_COLUMN: "hierarchy used",
            REASON_COLUMN: "mapping decision",
        },
    }


def add_unseen_metadata_codes(mapping: pl.DataFrame, code_metadata: pl.DataFrame) -> pl.DataFrame:
    """Carry forward metadata codes absent from training without fitting them."""
    unseen_codes = sorted(set(code_metadata.get_column(CODE_COLUMN).to_list()) - set(mapping.get_column(CODE_COLUMN)))
    if not unseen_codes:
        return mapping
    unseen = pl.DataFrame(
        {
            CODE_COLUMN: unseen_codes,
            MAPPED_CODE_COLUMN: unseen_codes,
            COUNT_COLUMN: [0] * len(unseen_codes),
            MAPPED_COUNT_COLUMN: [0] * len(unseen_codes),
            PROFILE_COLUMN: [None] * len(unseen_codes),
            REASON_COLUMN: ["unseen_training"] * len(unseen_codes),
        },
        schema=mapping.schema,
    )
    return pl.concat([mapping, unseen]).sort(CODE_COLUMN)


def prepare_mapping(
    local_metadata: pl.DataFrame,
    external_mapping_filepath: str | None,
    external_mapping_mode: str,
) -> pl.DataFrame:
    """Select a local mapping and optionally overlay or replace it externally."""
    required_local = {CODE_COLUMN, MAPPED_CODE_COLUMN}
    if not required_local.issubset(local_metadata.columns):
        if external_mapping_filepath:
            local = pl.DataFrame(schema={CODE_COLUMN: pl.String, MAPPED_CODE_COLUMN: pl.String})
        else:
            missing = sorted(required_local - set(local_metadata.columns))
            raise ValueError(f"local adaptive metadata is missing columns: {missing}")
    else:
        local = local_metadata.select(CODE_COLUMN, MAPPED_CODE_COLUMN)

    if not external_mapping_filepath:
        return local
    external = load_frame(str(external_mapping_filepath), "external mapping")
    missing = required_local - set(external.columns)
    if missing:
        raise ValueError(f"external mapping is missing columns: {sorted(missing)}")
    external = external.select(CODE_COLUMN, MAPPED_CODE_COLUMN)
    if external.get_column(CODE_COLUMN).n_unique() != external.height:
        raise ValueError("external mapping must contain at most one row per code")

    mode = str(external_mapping_mode).lower()
    if mode == "replace":
        return external
    if mode != "overlay":
        raise ValueError("external_mapping_mode must be 'overlay' or 'replace'")
    return pl.concat([external, local]).unique(CODE_COLUMN, keep="first", maintain_order=True)


def apply_mapping(data: pl.LazyFrame, mapping: pl.DataFrame) -> pl.LazyFrame:
    """Rewrite codes through a frozen mapping while preserving row order and schema."""
    temporary = "_adaptive_mapped_code"
    lookup = mapping.select(CODE_COLUMN, pl.col(MAPPED_CODE_COLUMN).alias(temporary)).lazy()
    return (
        data.join(lookup, on=CODE_COLUMN, how="left", maintain_order="left")
        .with_columns(pl.coalesce(temporary, CODE_COLUMN).alias(CODE_COLUMN))
        .drop(temporary)
    )


def collapse_code_metadata(metadata: pl.DataFrame, mapping: pl.DataFrame) -> pl.DataFrame:
    """Rewrite and deterministically collapse code metadata."""
    temporary = "_adaptive_mapped_code"
    mapped = (
        metadata.join(
            mapping.select(CODE_COLUMN, pl.col(MAPPED_CODE_COLUMN).alias(temporary)),
            on=CODE_COLUMN,
            how="left",
        )
        .with_columns(
            pl.coalesce(temporary, CODE_COLUMN).alias(temporary),
            (pl.col(CODE_COLUMN) == pl.coalesce(temporary, CODE_COLUMN)).alias("_adaptive_exact"),
        )
        .sort(temporary, "_adaptive_exact", CODE_COLUMN, descending=[False, True, False])
    )

    technical = {
        CODE_COLUMN,
        temporary,
        "_adaptive_exact",
        MAPPED_CODE_COLUMN,
        COUNT_COLUMN,
        MAPPED_COUNT_COLUMN,
        PROFILE_COLUMN,
        REASON_COLUMN,
        MEMBER_COUNT_COLUMN,
    }
    preserved = [column for column in metadata.columns if column not in technical]
    aggregations = [pl.col(column).drop_nulls().first().alias(column) for column in preserved]
    if COUNT_COLUMN in metadata.columns:
        aggregations.append(pl.col(COUNT_COLUMN).fill_null(0).sum().alias(COUNT_COLUMN))
    aggregations.append(pl.len().cast(pl.UInt32).alias(MEMBER_COUNT_COLUMN))

    collapsed = mapped.group_by(temporary, maintain_order=True).agg(*aggregations).rename({temporary: CODE_COLUMN})

    if "description" in collapsed.columns:
        generic = pl.concat_str(
            pl.lit("Adaptive aggregation "),
            pl.col(CODE_COLUMN),
            pl.lit(" ("),
            pl.col(MEMBER_COUNT_COLUMN),
            pl.lit(" source codes)"),
        )
        collapsed = collapsed.with_columns(
            pl.when(pl.col(MEMBER_COUNT_COLUMN) > 1).then(generic).otherwise(pl.col("description")).alias("description")
        )
    if "parent_codes" in collapsed.columns:
        collapsed = collapsed.with_columns(
            pl.when(pl.col(MEMBER_COUNT_COLUMN) > 1)
            .then(pl.lit(None, dtype=pl.List(pl.String)))
            .otherwise(pl.col("parent_codes"))
            .alias("parent_codes")
        )
    return collapsed.sort(CODE_COLUMN)


def add_missing_observed_metadata(metadata: pl.DataFrame, observed_codes: Sequence[str]) -> pl.DataFrame:
    """Ensure finalized metadata covers every code present in transformed data."""
    missing_codes = sorted(set(observed_codes) - set(metadata.get_column(CODE_COLUMN).to_list()))
    if not missing_codes:
        return metadata

    columns: dict[str, pl.Series] = {}
    for name, dtype in metadata.schema.items():
        if name == CODE_COLUMN:
            columns[name] = pl.Series(name, missing_codes, dtype=pl.String)
        elif name == "description":
            columns[name] = pl.Series(
                name,
                [f"Code observed outside adaptive-training metadata: {code}" for code in missing_codes],
                dtype=pl.String,
            )
        elif name == COUNT_COLUMN:
            columns[name] = pl.Series(name, [0] * len(missing_codes), dtype=dtype)
        elif name == MEMBER_COUNT_COLUMN:
            columns[name] = pl.Series(name, [1] * len(missing_codes), dtype=dtype)
        else:
            columns[name] = pl.Series(name, [None] * len(missing_codes), dtype=dtype)
    return pl.concat([metadata, pl.DataFrame(columns)]).sort(CODE_COLUMN)
