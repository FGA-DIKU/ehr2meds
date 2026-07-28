
# EHR2MEDS

## Overview

EHR2MEDS is a tool that formats dumps of Electronic Health Records (EHR) and converts them to MEDS (Medical Event Data Set).

0. **Prerequities**
   You need to install the correct packages using
   ```bash
   pip install -e .
   ```

1. **Raw → PREMEDS Conversion:**  
   Converts raw EHR data into a preliminary format called preMEDS.  
   Run 

   ```bash
   python ehr2meds/convert_raw_to_premeds.py --config-name <config_path>
   ```

   example:

   ```bash
   python ehr2meds/convert_raw_to_premeds.py --config-name preMEDS/DeepFetal/fetal_synth_full
   ```

   Example configuration files can be found in the [configs/preMEDS](./configs/preMEDS).
   
   The main functionalities of this is to 
   * Map subject ID hashes to integer values to ensure compatibility with MEDS
   * (optional) rename the raw column names to streamline the data input
   * Fill missing values from different data sources
   * Align timestamp inputs to one type
   * Connect visit ids etc with subject ids for the register data

2. **PREMEDS → MEDS Conversion:**  
   Transforms preMEDS data into a finalized MEDS cohort format.  
   You need to run [convert_premeds_to_meds.sh](./ehr2meds/convert_premeds_to_meds.sh) to run the MEDS conversion.

   ```bash
      bash ehr2meds/MEDS/MEDS_transform/run.sh \
      <PREMEDS_DIR> \
      <PIPELINE_CONFIG_FP> \
      <EVENT_CONFIG_FP> \
      <MEDS_OUTPUT_DIR> \
      [do_unzip=true|do_unzip=false]
   ```
   Example:

   ```bash
   source .env && bash ehr2meds/convert_premeds_to_meds.sh \
   ${EHR2MEDS_DATA}/preMEDS/DeepFetal/fetal_synth_full \
   ${EHR2MEDS_CONFIGS}/MEDS/default_pipeline.yaml \
   ${EHR2MEDS_CONFIGS}/MEDS/fetal_ngc_event.yaml \
   ${EHR2MEDS_DATA}/MEDS/DeepFetal/fetal_synth_full
   ```

   Example configuration files can be found in the [configs/MEDS](./configs/MEDS).

### Custom MEDS stages

The package includes the following stages to be used in MEDS pipeline configurations: 
| Stage | Purpose |
| --- | --- |
| `augment_event_config` | Adds shared columns, such as `row_idx`, to every event definition so they do not need to be repeated throughout the event configuration. |
| `aggregate_numeric_metadata` | Fits per-code numeric normalization bounds and adaptive quantile bins. It supports training-only fitting, an optional date cutoff (for OOT settings), hard plausibility limits (filtering values greater or lower than biological limits), and writes reusable numeric metadata. |
| `annotate_numeric_values` | Applies the fitted metadata (from `aggregate_numeric_metadata`) to create new columns based on the numeric values. It adds normalized values, bin indices, and binned representatives. External numeric metadata can override locally fitted metadata. |
| `fit_adaptive_code_mapping` | Fits a hierarchy-aware code mapping from training-event counts. It supports ATC levels, the bundled Danish SKS hierarchy, synthetic diagnosis prefix families, and external hierarchy overlays. |
| `apply_adaptive_code_mapping` | Applies the frozen local and/or external adaptive mapping to every data split while retaining the MEDS event namespace. |
| `finalize_adaptive_code_metadata` | Rewrites and collapses `codes.parquet` to match the adaptively transformed data vocabulary. |
| `join_numeric_bins` | Optionally creates the "joined representation" of numeric values, such as `LAB_CODE//bin_3`, from the numeric bin index. |
| `bin_numeric_values_fast` | A faster, memory-efficient replacement for the standard MEDS-Transforms discrete binning stage. It rewrites codes using bin indices or interval labels. |

Adaptive truncation uses raw training-event counts, not distinct-subject
counts. Its normal configuration is deliberately small: set `minimum_count`
and map each MEDS namespace to one of `sks_diagnosis`, `sks_operation`,
`sks_other_procedure`, or `atc` under `namespaces`. Built-in definitions hold
the hierarchy details.

The three adaptive stages correspond to different MEDS-Transforms data flows:

1. `fit_adaptive_code_mapping` globally reduces **training shards** into one
   frozen mapping.
2. `apply_adaptive_code_mapping` maps **every data shard** without learning
   from tuning or test data.
3. `finalize_adaptive_code_metadata` reconciles **code metadata** after the
   transformed data vocabulary is known.

`extract_code_metadata` is the standard upstream MEDS stage, not part of the
adaptive implementation. The fit stage writes the full reviewable mapping as
Parquet and a compact `*.summary.json` audit containing vocabulary sizes,
affected event counts, and decisions by hierarchy and reason. Later per-code
metadata stages must follow metadata reconciliation.

Custom profiles remain available through the optional `hierarchies` mapping.
For example, a fixed-length hierarchy can be defined as
`my_codes: {kind: levels, levels: [2, 4, 6]}` and selected with
`namespaces: {MY_NAMESPACE: my_codes}`. An entry named like a built-in profile
overrides only the specified built-in fields.

`mapping_filepath` can point to a frozen JSON or Parquet mapping with `code`
and `adaptive/mapped_code` columns. In the default `overlay` mode, external
rows override locally fitted rows. Set `external_mapping_mode: replace` to use
only the external mapping. `external_hierarchy_filepath` similarly overlays
parent relations before local fitting; it accepts `code`, `parent_code`, and
an optional `profile` column.

For combined numeric encoding, use `aggregate_numeric_metadata` followed by
`annotate_numeric_values`. Add `join_numeric_bins` afterwards only when the final
model input should contain joined lab-and-bin codes.

Shared numeric column names and stage defaults are defined in
`configs/MEDS/default_numeric_values.yaml`; pipeline configurations only
need to specify dataset- or run-specific overrides. Any setting can be overridden
under the relevant pipeline stage. The `numeric_value_column_groups` lists control
which transform, optional bound, and derived columns are used; column names and
derived outputs are configured through `numeric_value_columns`.
