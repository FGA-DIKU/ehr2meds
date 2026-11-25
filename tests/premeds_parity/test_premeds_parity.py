"""pandas vs polars parity check for the raw -> preMEDS stage.

Compares the polars port against output recorded with the pandas version. Delete this folder once the port is
merged; tests/premeds keeps the tests that don't depend on pandas. Regenerate the baseline only from a known-good run:

    UPDATE_PREMEDS_BASELINE=1 pytest tests/premeds_parity
"""

import os
import pytest
import shutil
from pathlib import Path
from tests.premeds.test_premeds import run_premeds
from tests.premeds_parity.compare import compare_dirs
from tests.premeds_parity.inputs import table_configs, write_inputs

BASELINE = Path(__file__).parent / "baseline"


@pytest.mark.parametrize("write_file_type", ["csv", "parquet"])
def test_premeds_output_matches_baseline(tmp_path, capsys, write_file_type):
    raw_dir, output = tmp_path / "raw", tmp_path / "out"
    write_inputs(raw_dir)
    run_premeds(output, table_configs(raw_dir), raw_dir / "population.csv", write_file_type)
    # The summary and subject mapping don't depend on the output format, so only the parquet baseline keeps them.
    if write_file_type == "parquet":
        (output / "data_quality_summary.txt").write_text(capsys.readouterr().out)
    else:
        (output / "subject_id_mapping.csv").unlink()

    expected = BASELINE / write_file_type
    if os.environ.get("UPDATE_PREMEDS_BASELINE"):
        shutil.rmtree(expected, ignore_errors=True)
        shutil.copytree(output, expected)
        pytest.skip(f"baseline written to {expected}")

    problems = compare_dirs(expected, output)
    assert not problems, "\n\n".join(problems)
