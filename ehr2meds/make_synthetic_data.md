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
python ehr2meds/generate_synthetic_raw_data.py --config-name synthetic_generation/fetal_SP &&
python ehr2meds/generate_synthetic_raw_data.py --config-name synthetic_generation/fetal_SDS &&
python ehr2meds/generate_synthetic_raw_data.py --config-name synthetic_generation/dst &&
python ehr2meds/generate_synthetic_raw_data.py --config-name synthetic_generation/skin_cancer

```

Output paths are set inside each YAML (for example `paths.output` in `fetal_SP.yaml`).

---

### 2. Raw → PREMEDS

Run the PREMEDS conversion with a config that points at your synthetic raw data and desired PREMEDS output:

```bash
python ehr2meds/convert_raw_to_premeds.py --config-name preMEDS/fetal_SP     
```

Adjust `--config-name` if you use a different PREMEDS profile.

Optionally you can also normalise the data after converting to preMEDS, using 
```bash
python ehr2meds/normalize_premeds.py --config-name preMEDS/normalize     
```
---

### 3. PREMEDS → MEDS

Invoke the MEDS transform shell script with four arguments: PREMEDS input directory, pipeline config, event conversion config, and MEDS output directory. Optional fifth argument: `do_unzip=true` or `do_unzip=false` for `.csv.gz` inputs.

**Template:**

```bash
bash ehr2meds/convert_premeds_to_meds.sh \
  <PREMEDS_DIR> \
  <PIPELINE_CONFIG_FP> \
  <EVENT_CONFIG_FP> \
  <MEDS_OUTPUT_DIR> \
  [do_unzip=true|do_unzip=false]
```

**Example using env paths:**

```bash
source .env && bash ehr2meds/convert_premeds_to_meds.sh \
  ${EHR2MEDS_DATA}/preMEDS/fetal_data/SP \
  ${EHR2MEDS_CONFIGS}/MEDS/default_pipeline.yaml \
  ${EHR2MEDS_CONFIGS}/MEDS/default_event.yaml \
  ${EHR2MEDS_DATA}/MEDS/SP
```

**Example using relative paths:**
```bash
bash ehr2meds/convert_premeds_to_meds.sh \
  data/preMEDS/fetal_data/SP \
  configs/MEDS/default_pipeline.yaml \
  configs/MEDS/default_event.yaml \
  data/MEDS/SP
```
---

**Next step:** use the `data` directory produced under the MEDS output folder as input to the BONSAI (or other) pipeline setup.