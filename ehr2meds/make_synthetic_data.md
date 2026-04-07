  # Synthetic data workflow

End-to-end flow: generate tabular synthetic “raw” data, convert to PREMEDS, then to MEDS for downstream use (e.g. BONSAI).

**Prerequisites:** from the repository root (the folder that contains the `ehr2meds/` package), install the package in editable mode:

```bash
pip install -e .
```

Run all commands below from that same directory unless noted.

---

### 1. Synthetic data generation

Uses `ehr2meds/generate_synthetic_raw_data.py`. YAML configs live under `configs/synthetic_generation/` (pass **only the filename** with `--config`).

```bash
python ehr2meds/generate_synthetic_raw_data.py --config synthetic_generation/fetal_SP.yaml
```

Output paths are set inside each YAML (for example `paths.output` in `fetal_SP.yaml`).

---

### 2. Raw → PREMEDS

Run the PREMEDS conversion with a config that points at your synthetic raw data and desired PREMEDS output:

```bash
python ehr2meds/convert_raw_to_premeds.py --config preMEDS/fetal_SP.yaml     
```

Adjust `--config` if you use a different PREMEDS profile.

---

### 3. PREMEDS → MEDS

Invoke the MEDS transform shell script with four arguments: PREMEDS input directory, pipeline config, event conversion config, and MEDS output directory. Optional fifth argument: `do_unzip=true` or `do_unzip=false` for `.csv.gz` inputs.

**Template:**

```bash
bash ehr2meds/MEDS/MEDS_transform/run.sh \
  <PREMEDS_DIR> \
  <PIPELINE_CONFIG_FP> \
  <EVENT_CONFIG_FP> \
  <MEDS_OUTPUT_DIR> \
  [do_unzip=true|do_unzip=false]
```

**Example (paths relative to repo root):**

```bash
bash ehr2meds/convert_premeds_to_meds.sh \
  ehr2meds/data/preMEDS/fetal_data/SP \
  ehr2meds/configs/MEDS/default_pipeline.yaml \
  ehr2meds/configs/MEDS/default_event.yaml \
  ehr2meds/data/MEDS/SP
```
---

**Next step:** use the `data` directory produced under the MEDS output folder as input to the BONSAI (or other) pipeline setup.