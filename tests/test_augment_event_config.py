import yaml
from ehr2meds.meds_stages.augment_event_config import augment_event_config


def test_augment_event_config_adds_provenance_only_to_events(tmp_path):
    source = tmp_path / "events.yaml"
    output = tmp_path / "events.augmented.yaml"
    source.write_text(
        """
subject_id_col: subject_id
table:
  subject_id_col: patient_id
  transforms:
    parsed_time: $timestamp
  diagnosis:
    code: f"D//{$diagnosis}"
    time: $parsed_time
  static:
    code: STATIC
    time: null
""".lstrip(),
        encoding="utf-8",
    )

    augment_event_config(source, output)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert config["subject_id_col"] == "subject_id"
    assert config["table"]["transforms"] == {"parsed_time": "$timestamp"}
    for event_name in ("diagnosis", "static"):
        assert config["table"][event_name]["source_row_id"] == "$source_row_id"
        assert config["table"][event_name]["source_row_index"] == "$source_row_index"


def test_augment_event_config_is_idempotent_and_rejects_conflicts(tmp_path):
    source = tmp_path / "events.yaml"
    output = tmp_path / "events.augmented.yaml"
    source.write_text(
        "table:\n  event:\n    code: X\n    time: null\n    source_row_id: $source_row_id\n",
        encoding="utf-8",
    )

    augment_event_config(source, output)
    augment_event_config(output, output)

    conflict = tmp_path / "conflict.yaml"
    conflict.write_text(
        "table:\n  event:\n    code: X\n    time: null\n    source_row_id: $other_id\n",
        encoding="utf-8",
    )
    try:
        augment_event_config(conflict, output)
    except ValueError as error:
        assert "conflicting expression" in str(error)
    else:
        raise AssertionError("A conflicting event field must not be overwritten")
