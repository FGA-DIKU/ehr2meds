"""Pin intended raw -> preMEDS outputs so a rewrite of the preMEDS stage can be checked against them.

Known bugs are excluded from these inputs and tracked in test_known_bugs.py, so fixing one does not
break the baseline. Regenerate the baseline only from a known-good implementation:

    UPDATE_PREMEDS_BASELINE=1 pytest tests/premeds_parity
"""

import os
import pytest
import shutil
from ehr2meds.preMEDS.extractor import PREMEDSExtractor
from omegaconf import OmegaConf
from pathlib import Path
from tests.premeds_parity.compare import compare_dirs
from tests.premeds_parity.inputs import table_configs, write_inputs

BASELINE = Path(__file__).parent / "baseline"


def run_premeds(output: Path, tables: dict, population: Path, write_file_type="parquet", chunksize=100) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output": str(output)},
            "write_file_type": write_file_type,
            "chunksize": chunksize,
            "test": False,
            "num_workers": 1,
            "subject_id_mapping": {"file": str(population), "subject_id_col": "cpr", "mapping_id_col": None},
            "tables": tables,
        }
    )
    output.mkdir(parents=True)
    PREMEDSExtractor(cfg)()


# Single chunk: chunked output currently depends on chunksize (see test_known_bugs.py).
@pytest.mark.parametrize("write_file_type", ["csv", "parquet"])
def test_premeds_output_matches_baseline(tmp_path, capsys, write_file_type):
    raw_dir, output = tmp_path / "raw", tmp_path / "out"
    write_inputs(raw_dir)
    run_premeds(output, table_configs(raw_dir), raw_dir / "population.csv", write_file_type)
    (output / "data_quality_summary.txt").write_text(capsys.readouterr().out)

    expected = BASELINE / write_file_type
    if os.environ.get("UPDATE_PREMEDS_BASELINE"):
        shutil.rmtree(expected, ignore_errors=True)
        shutil.copytree(output, expected)
        pytest.skip(f"baseline written to {expected}")

    problems = compare_dirs(expected, output)
    assert not problems, "\n\n".join(problems)
