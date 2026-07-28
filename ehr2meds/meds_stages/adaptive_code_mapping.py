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

# Stable hierarchy definitions for the code systems supported out of the box.
# A config normally only needs to route MEDS namespaces to these names.
BUILTIN_HIERARCHIES: dict[str, dict[str, object]] = {
    "sks_diagnosis": {
        "kind": "table",
        "record_types": ["dia"],
        "leading_marker": "D",
        "minimum_canonical_length": 2,
        "synthetic_prefix_lengths": [2, 3],
    },
    "sks_operation": {
        "kind": "table",
        "record_types": ["opr"],
        "leading_marker": "K",
        "minimum_canonical_length": 2,
    },
    "sks_other_procedure": {
        "kind": "table",
        "record_types": ["pro", "und"],
        "minimum_canonical_length": 2,
    },
    "atc": {
        "kind": "levels",
        "levels": [1, 3, 4, 5, 7],
        "minimum_canonical_length": 1,
    },
}


@dataclass(frozen=True)
class HierarchyProfile:
    """Validated configuration for one hierarchy."""

    name: str
    kind: str
    minimum_count: int
    levels: tuple[int, ...] = ()
    record_types: tuple[str, ...] = ()
    leading_marker: str | None = None
    minimum_canonical_length: int = 1
    synthetic_prefix_lengths: tuple[int, ...] = ()
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
        kind = str(raw.get("kind", "")).lower()
        if kind not in {"levels", "table"}:
            raise ValueError(f"hierarchies[{name!r}].kind must be 'levels' or 'table'")

        levels = tuple(sorted({int(level) for level in raw.get("levels", [])}, reverse=True))
        if kind == "levels" and (not levels or min(levels) < 1):
            raise ValueError(f"hierarchies[{name!r}].levels must contain positive integers")

        record_types = tuple(str(value) for value in raw.get("record_types", []))
        if kind == "table" and not record_types:
            raise ValueError(f"hierarchies[{name!r}].record_types cannot be empty")

        minimum_count = int(raw.get("minimum_count", default_minimum))
        if minimum_count < 1:
            raise ValueError(f"hierarchies[{name!r}].minimum_count must be at least 1")

        synthetic = tuple(sorted({int(level) for level in raw.get("synthetic_prefix_lengths", [])}, reverse=True))
        minimum_length = int(raw.get("minimum_canonical_length", 1))
        if minimum_length < 1 or any(level < minimum_length for level in synthetic):
            raise ValueError(f"hierarchies[{name!r}] prefix lengths must be at least minimum_canonical_length")

        marker = raw.get("leading_marker")
        profiles[str(name)] = HierarchyProfile(
            name=str(name),
            kind=kind,
            minimum_count=minimum_count,
            levels=levels,
            record_types=record_types,
            leading_marker=str(marker).upper() if marker else None,
            minimum_canonical_length=minimum_length,
            synthetic_prefix_lengths=synthetic,
            uppercase=bool(raw.get("uppercase", True)),
            remove_dots=bool(raw.get("remove_dots", True)),
        )

    if any("//" in namespace or not namespace for namespace in namespaces):
        raise ValueError("namespace names must be non-empty and cannot contain '//'")
    return profiles, namespaces


def resolve_resource_path(filepath: str) -> Path:
    """Resolve normal, package, and bundled-resource paths."""
    if filepath == "builtin://sks":
        return Path(__file__).resolve().parents[1] / "resources" / "sks_hierarchy.parquet"
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


@dataclass
class TableHierarchy:
    """Code-table lookup shared by table-based profiles."""

    parent_by_code: dict[str, str | None]
    allowed_by_profile: dict[str, set[str]]


def load_table_hierarchy(
    profiles: Mapping[str, HierarchyProfile],
    hierarchy_filepath: str | None,
    external_hierarchy_filepath: str | None = None,
) -> TableHierarchy:
    """Load the base hierarchy and overlay optional external parent rows."""
    table_profiles = {name: profile for name, profile in profiles.items() if profile.kind == "table"}
    if not table_profiles:
        return TableHierarchy({}, {})
    if not hierarchy_filepath:
        raise ValueError("hierarchy_filepath is required when a table hierarchy is configured")

    base = load_frame(str(hierarchy_filepath), "hierarchy")
    required = {"record_type", "code", "parent_code"}
    missing = required - set(base.columns)
    if missing:
        raise ValueError(f"hierarchy is missing columns: {sorted(missing)}")

    pairs = base.select("code", "parent_code").unique()
    conflicts = pairs.group_by("code").agg(pl.col("parent_code").n_unique().alias("_n")).filter(pl.col("_n") > 1)
    if conflicts.height:
        raise ValueError("hierarchy contains codes with conflicting parents")
    parent_by_code = dict(pairs.iter_rows())
    allowed_by_profile = {
        name: set(base.filter(pl.col("record_type").is_in(profile.record_types)).get_column("code").unique().to_list())
        for name, profile in table_profiles.items()
    }

    if external_hierarchy_filepath:
        external = load_frame(str(external_hierarchy_filepath), "external hierarchy")
        missing = {"code", "parent_code"} - set(external.columns)
        if missing:
            raise ValueError(f"external hierarchy is missing columns: {sorted(missing)}")
        if external.get_column("code").n_unique() != external.height:
            raise ValueError("external hierarchy must contain at most one row per code")

        has_profile = "profile" in external.columns
        for row in external.iter_rows(named=True):
            code = str(row["code"])
            parent = None if row["parent_code"] is None else str(row["parent_code"])
            parent_by_code[code] = parent
            if has_profile and row["profile"] is not None:
                profile_name = str(row["profile"])
                if profile_name not in table_profiles:
                    raise ValueError(f"external hierarchy references unknown table profile {profile_name!r}")
                allowed_by_profile[profile_name].add(code)
            else:
                for allowed in allowed_by_profile.values():
                    allowed.add(code)

    return TableHierarchy(parent_by_code, allowed_by_profile)


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


def _display_candidate(canonical: str, marker_was_added: bool, profile: HierarchyProfile) -> str:
    if marker_was_added and profile.leading_marker and canonical.startswith(profile.leading_marker):
        return canonical[len(profile.leading_marker) :]
    return canonical


def candidate_ancestors(
    payload: str,
    profile: HierarchyProfile,
    table: TableHierarchy,
) -> list[tuple[str, str]]:
    """Return nearest-to-broadest ``(payload, reason)`` candidates."""
    normalized = normalize_payload(payload, profile)
    if profile.kind == "levels":
        return [
            (normalized[:length], "structural")
            for length in profile.levels
            if length < len(normalized) and length >= profile.minimum_canonical_length
        ]

    allowed = table.allowed_by_profile.get(profile.name, set())
    canonical = normalized
    marker_was_added = False
    if canonical not in allowed and profile.leading_marker and f"{profile.leading_marker}{canonical}" in allowed:
        canonical = f"{profile.leading_marker}{canonical}"
        marker_was_added = True
    if canonical not in allowed:
        return []

    candidates: dict[str, str] = {}
    seen = {canonical}
    parent = table.parent_by_code.get(canonical)
    while parent is not None:
        if parent in seen:
            raise ValueError(f"hierarchy cycle detected at {parent!r}")
        seen.add(parent)
        if len(parent) >= profile.minimum_canonical_length:
            candidates[parent] = "official"
        parent = table.parent_by_code.get(parent)

    for length in profile.synthetic_prefix_lengths:
        if profile.minimum_canonical_length <= length < len(canonical):
            candidates.setdefault(canonical[:length], "synthetic")

    ordered = sorted(candidates.items(), key=lambda item: (-len(item[0]), item[0]))
    return [(_display_candidate(candidate, marker_was_added, profile), reason) for candidate, reason in ordered]


def fit_mapping(
    counts: Mapping[str, int],
    profiles: Mapping[str, HierarchyProfile],
    namespaces: Mapping[str, str],
    table: TableHierarchy,
) -> pl.DataFrame:
    """Fit a deterministic, disjoint adaptive hierarchy mapping."""
    records: dict[str, dict[str, object]] = {}
    pending_by_profile: dict[str, set[str]] = defaultdict(set)
    candidates_by_code: dict[str, list[tuple[str, str]]] = {}

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
        ancestors = [
            (f"{namespace}//{candidate}", reason) for candidate, reason in candidate_ancestors(payload, profile, table)
        ]
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
            {len(candidate) for code in pending for candidate, _ in candidates_by_code[code]},
            reverse=True,
        )
        for length in lengths:
            groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
            for code in sorted(pending):
                for candidate, reason in candidates_by_code[code]:
                    if len(candidate) == length:
                        groups[candidate].append((code, reason))
                        break

            for candidate in sorted(groups):
                members = [(code, reason) for code, reason in groups[candidate] if code in pending]
                mapped_count = sum(int(counts[code]) for code, _ in members)
                if mapped_count < profile.minimum_count:
                    continue
                reason = "synthetic" if any(member_reason == "synthetic" for _, member_reason in members) else "grouped"
                for code, _ in members:
                    records[code] = {
                        CODE_COLUMN: code,
                        MAPPED_CODE_COLUMN: candidate,
                        COUNT_COLUMN: int(counts[code]),
                        MAPPED_COUNT_COLUMN: mapped_count,
                        PROFILE_COLUMN: profile_name,
                        REASON_COLUMN: reason,
                    }
                    pending.remove(code)

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
