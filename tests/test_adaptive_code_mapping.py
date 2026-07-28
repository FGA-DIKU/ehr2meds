from __future__ import annotations

import ehr2meds.meds_stages.adaptive_code_mapping as adaptive
import json
import polars as pl
import pytest
from ehr2meds.meds_stages.adaptive_code_mapping import (
    CODE_COLUMN,
    COUNT_COLUMN,
    MAPPED_CODE_COLUMN,
    MEMBER_COUNT_COLUMN,
    REASON_COLUMN,
    HierarchyProfile,
    TableHierarchy,
    add_missing_observed_metadata,
    add_unseen_metadata_codes,
    apply_mapping,
    candidate_ancestors,
    collapse_code_metadata,
    fit_mapping,
    prepare_mapping,
    read_profiles,
    summarize_mapping,
)
from ehr2meds.meds_stages.fit_adaptive_code_mapping import mapper_fntr, reducer_fntr
from omegaconf import OmegaConf
from pathlib import Path


def profiles() -> dict[str, HierarchyProfile]:
    return {
        "diagnosis": HierarchyProfile(
            name="diagnosis",
            kind="table",
            minimum_count=50,
            record_types=("dia",),
            leading_marker="D",
            minimum_canonical_length=2,
            synthetic_prefix_lengths=(3, 2),
        ),
        "operation": HierarchyProfile(
            name="operation",
            kind="table",
            minimum_count=50,
            record_types=("opr",),
            leading_marker="K",
            minimum_canonical_length=2,
        ),
        "atc": HierarchyProfile(
            name="atc",
            kind="levels",
            minimum_count=50,
            levels=(7, 5, 4, 3, 1),
        ),
    }


def hierarchy() -> TableHierarchy:
    parent_by_code = {
        "D": None,
        "DC83": "D",
        "DC833": "DC83",
        "DC833A": "DC833",
        "DC833B": "DC833",
        "DC833C": "DC833",
        "K": None,
        "KA": "K",
        "KAB": "KA",
        "KABA": "KAB",
        "KABA00": "KABA",
    }
    return TableHierarchy(
        parent_by_code,
        {
            "diagnosis": {code for code in parent_by_code if code.startswith("D")},
            "operation": {code for code in parent_by_code if code.startswith("K")},
        },
    )


def test_candidate_ancestors_use_real_and_structural_boundaries() -> None:
    configured = profiles()
    table = hierarchy()

    assert candidate_ancestors("C833A", configured["diagnosis"], table) == [
        ("C833", "official"),
        ("C83", "official"),
        ("C8", "synthetic"),
        ("C", "synthetic"),
    ]
    assert candidate_ancestors("KABA00", configured["operation"], table) == [
        ("KABA", "official"),
        ("KAB", "official"),
        ("KA", "official"),
    ]
    assert candidate_ancestors("N02BA01", configured["atc"], table) == [
        ("N02BA", "structural"),
        ("N02B", "structural"),
        ("N02", "structural"),
        ("N", "structural"),
    ]
    assert candidate_ancestors("FLAT", configured["diagnosis"], table) == []


def test_fit_mapping_keeps_common_codes_and_groups_rare_siblings() -> None:
    counts = {
        "RD//C833A": 30,
        "RD//C833B": 25,
        "RD//C833C": 100,
        "RC//C833A": 20,
        "RC//C833B": 20,
        "OTHER//C833A": 10,
    }
    mapping = fit_mapping(
        counts,
        profiles=profiles(),
        namespaces={"RD": "diagnosis", "RC": "diagnosis"},
        table=hierarchy(),
    )
    result = {row[CODE_COLUMN]: row for row in mapping.to_dicts()}

    assert result["RD//C833A"][MAPPED_CODE_COLUMN] == "RD//C833"
    assert result["RD//C833B"][MAPPED_CODE_COLUMN] == "RD//C833"
    assert result["RD//C833A"][REASON_COLUMN] == "grouped"
    assert result["RD//C833A"]["adaptive/mapped_count"] == 55
    assert result["RD//C833C"][MAPPED_CODE_COLUMN] == "RD//C833C"
    assert result["RD//C833C"][REASON_COLUMN] == "retained"

    # Distinct MEDS namespaces never contribute counts to one another.
    assert result["RC//C833A"][MAPPED_CODE_COLUMN] == "RC//C833A"
    assert result["RC//C833A"][REASON_COLUMN] == "below_threshold"
    assert result["OTHER//C833A"][MAPPED_CODE_COLUMN] == "OTHER//C833A"
    assert result["OTHER//C833A"][REASON_COLUMN] == "unconfigured"


def test_fit_mapping_can_use_synthetic_diagnosis_family() -> None:
    counts = {
        "RD//C833A": 20,
        "RD//C849A": 20,
        "RD//C859A": 20,
    }
    table = hierarchy()
    table.parent_by_code.update(
        {
            "DC849A": "DC84",
            "DC84": "D",
            "DC859A": "DC85",
            "DC85": "D",
        }
    )
    table.allowed_by_profile["diagnosis"].update({"DC849A", "DC84", "DC859A", "DC85"})

    mapping = fit_mapping(
        counts,
        profiles=profiles(),
        namespaces={"RD": "diagnosis"},
        table=table,
    )
    assert mapping.get_column(MAPPED_CODE_COLUMN).to_list() == ["RD//C8", "RD//C8", "RD//C8"]
    assert mapping.get_column(REASON_COLUMN).to_list() == ["synthetic", "synthetic", "synthetic"]


def test_mapping_summary_is_compact_and_reviewable() -> None:
    mapping = fit_mapping(
        {"RD//C833A": 30, "RD//C833B": 25, "RD//C833C": 100},
        profiles=profiles(),
        namespaces={"RD": "diagnosis"},
        table=hierarchy(),
    )
    audit = summarize_mapping(mapping)

    assert audit["summary"] == {
        "metadata_source_codes": 3,
        "training_source_codes": 3,
        "output_codes": 2,
        "changed_source_codes": 2,
        "training_events": 155,
        "remapped_training_events": 55,
    }
    assert {row["reason"] for row in audit["decisions"]} == {"grouped", "retained"}


def test_fit_stage_emits_mapping_and_readable_summary(monkeypatch) -> None:
    written: dict[str, object] = {}

    def capture_mapping(frame: pl.DataFrame, path: Path) -> None:
        written["mapping_path"] = str(path)
        written["mapping"] = frame

    def capture_summary(path: Path, text: str, **_kwargs) -> None:
        written["summary_path"] = str(path)
        written["summary"] = text

    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    monkeypatch.setattr(Path, "mkdir", lambda _path, **_kwargs: None)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", capture_mapping)
    monkeypatch.setattr(Path, "write_text", capture_summary)
    cfg = OmegaConf.create(
        {
            "minimum_count": 10,
            "namespaces": {"RM": "atc"},
            "metadata_input_dir": "metadata",
            "reducer_output_dir": "output",
        }
    )

    reducer_fntr(cfg)(
        pl.DataFrame(
            {
                "code": ["RM//N02BA01", "RM//N02BA02"],
                COUNT_COLUMN: [6, 5],
            }
        )
    ).collect()

    assert written["mapping_path"] == str(Path("output/adaptive_code_mapping.parquet"))
    assert written["summary_path"] == str(Path("output/adaptive_code_mapping.summary.json"))
    summary = json.loads(str(written["summary"]))
    assert summary["summary"]["training_events"] == 11
    assert summary["summary"]["output_codes"] == 1
    assert summary["decisions"] == [
        {
            "profile": "atc",
            "reason": "grouped",
            "source_codes": 2,
            "training_events": 11,
        }
    ]


def test_mapper_counts_events_not_subjects() -> None:
    cfg = OmegaConf.create(
        {
            "minimum_count": 2,
            "hierarchies": {
                "atc": {
                    "kind": "levels",
                    "levels": [1, 3, 4, 5, 7],
                }
            },
            "namespaces": {"RM": "atc"},
        }
    )
    data = pl.DataFrame(
        {
            "subject_id": [1, 1, 1, 2],
            "code": ["RM//N02BA01", "RM//N02BA01", "OTHER//X", "RM//N02BA01"],
        }
    )
    counted = mapper_fntr(cfg)(data.lazy()).collect()
    assert counted.to_dicts() == [
        {"code": "OTHER//X", COUNT_COLUMN: 1},
        {"code": "RM//N02BA01", COUNT_COLUMN: 3},
    ]


def test_builtin_profile_only_needs_namespace_routing() -> None:
    cfg = OmegaConf.create({"minimum_count": 25, "namespaces": {"RM": "atc"}})
    configured, namespaces = read_profiles(cfg)

    assert namespaces == {"RM": "atc"}
    assert configured["atc"].levels == (7, 5, 4, 3, 1)
    assert configured["atc"].minimum_count == 25


def test_builtin_profile_can_be_partially_overridden() -> None:
    cfg = OmegaConf.create(
        {
            "minimum_count": 25,
            "namespaces": {"RD": "sks_diagnosis"},
            "hierarchies": {"sks_diagnosis": {"synthetic_prefix_lengths": []}},
        }
    )
    configured, _ = read_profiles(cfg)

    assert configured["sks_diagnosis"].record_types == ("dia",)
    assert configured["sks_diagnosis"].synthetic_prefix_lengths == ()


def test_apply_mapping_preserves_rows_and_unmapped_codes() -> None:
    data = pl.DataFrame({"row": [0, 1, 2], "code": ["RD//A", "RD//B", "OTHER//X"]})
    mapping = pl.DataFrame(
        {
            "code": ["RD//A", "RD//B"],
            MAPPED_CODE_COLUMN: ["RD//PARENT", "RD//PARENT"],
        }
    )
    result = apply_mapping(data.lazy(), mapping).collect()
    assert result.to_dicts() == [
        {"row": 0, "code": "RD//PARENT"},
        {"row": 1, "code": "RD//PARENT"},
        {"row": 2, "code": "OTHER//X"},
    ]


def test_unseen_training_codes_are_carried_forward_without_fitting() -> None:
    mapping = fit_mapping(
        {"RD//C833A": 20},
        profiles=profiles(),
        namespaces={"RD": "diagnosis"},
        table=hierarchy(),
    )
    metadata = pl.DataFrame({"code": ["RD//C833A", "RD//C833B", "OTHER//X"]})
    completed = add_unseen_metadata_codes(mapping, metadata)
    result = {row[CODE_COLUMN]: row for row in completed.to_dicts()}

    assert result["RD//C833B"][MAPPED_CODE_COLUMN] == "RD//C833B"
    assert result["RD//C833B"][COUNT_COLUMN] == 0
    assert result["RD//C833B"][REASON_COLUMN] == "unseen_training"
    assert result["OTHER//X"][REASON_COLUMN] == "unseen_training"


def test_external_mapping_overlay_and_replace(monkeypatch) -> None:
    local = pl.DataFrame(
        {
            "code": ["A", "B"],
            MAPPED_CODE_COLUMN: ["LOCAL_A", "LOCAL_B"],
        }
    )
    external = pl.DataFrame(
        {
            "code": ["A", "C"],
            MAPPED_CODE_COLUMN: ["EXTERNAL_A", "EXTERNAL_C"],
        }
    )
    monkeypatch.setattr(adaptive, "load_frame", lambda filepath, label: external)

    overlay = prepare_mapping(local, "external.parquet", "overlay")
    assert dict(overlay.iter_rows()) == {"A": "EXTERNAL_A", "C": "EXTERNAL_C", "B": "LOCAL_B"}
    replacement = prepare_mapping(local, "external.parquet", "replace")
    assert dict(replacement.iter_rows()) == {"A": "EXTERNAL_A", "C": "EXTERNAL_C"}


def test_collapse_metadata_produces_one_row_per_mapped_code() -> None:
    metadata = pl.DataFrame(
        {
            "code": ["RD//A", "RD//B", "OTHER//X"],
            "description": ["A", "B", "Other"],
            "parent_codes": [None, None, None],
            COUNT_COLUMN: [20, 30, None],
            MAPPED_CODE_COLUMN: ["RD//PARENT", "RD//PARENT", None],
        },
        schema_overrides={"parent_codes": pl.List(pl.String)},
    )
    mapping = metadata.select("code", MAPPED_CODE_COLUMN)
    collapsed = collapse_code_metadata(metadata, mapping)

    assert collapsed.get_column("code").to_list() == ["OTHER//X", "RD//PARENT"]
    parent = collapsed.filter(pl.col("code") == "RD//PARENT").to_dicts()[0]
    assert parent[COUNT_COLUMN] == 50
    assert parent[MEMBER_COUNT_COLUMN] == 2
    assert parent["description"] == "Adaptive aggregation RD//PARENT (2 source codes)"
    assert parent["parent_codes"] is None


def test_missing_held_out_codes_are_added_to_final_metadata() -> None:
    metadata = pl.DataFrame(
        {
            "code": ["RD//TRAIN"],
            "description": ["Training code"],
            "parent_codes": [None],
            COUNT_COLUMN: [10],
            MEMBER_COUNT_COLUMN: [1],
        },
        schema_overrides={"parent_codes": pl.List(pl.String)},
    )
    completed = add_missing_observed_metadata(metadata, ["RD//TRAIN", "RD//HELD_OUT"])

    assert completed.get_column("code").to_list() == ["RD//HELD_OUT", "RD//TRAIN"]
    held_out = completed.filter(pl.col("code") == "RD//HELD_OUT").to_dicts()[0]
    assert held_out[COUNT_COLUMN] == 0
    assert held_out[MEMBER_COUNT_COLUMN] == 1


def test_profile_validation_rejects_unknown_namespace_profile() -> None:
    cfg = OmegaConf.create(
        {
            "minimum_count": 10,
            "hierarchies": {"atc": {"kind": "levels", "levels": [1, 3, 7]}},
            "namespaces": {"RM": "missing"},
        }
    )
    with pytest.raises(ValueError, match="undefined hierarchy"):
        read_profiles(cfg)
