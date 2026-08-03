
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
| `annotate_numeric_values` | Applies the fitted metadata (from `aggregate_numeric_metadata`) to create new columns based on the numeric values. It adds normalized values, bin indices, and binned representatives. If no local fit exists, falls back to external numeric metadata instead. |
| `fit_adaptive_code_mapping` | Fits a code mapping from training-event counts by climbing character-position levels (ATC and SKS diagnosis/operation/procedure) only as far as needed to clear a minimum count. |
| `apply_adaptive_code_mapping` | Applies the frozen local mapping (or, if none was fitted, an external one) to every data split while retaining the MEDS event namespace. |
| `finalize_adaptive_code_metadata` | Rewrites and collapses `codes.parquet` to match the adaptively transformed data vocabulary. |
| `join_numeric_bins` | Optionally creates the "joined representation" of numeric values, such as `LAB_CODE//bin_3`, from the numeric bin index. |
| `bin_numeric_values_fast` | A faster, memory-efficient replacement for the standard MEDS-Transforms discrete binning stage. It rewrites codes using bin indices or interval labels. |

### Adaptive code mapping

Adaptive mapping uses raw training-event counts, not distinct-subject counts:

1. `fit_adaptive_code_mapping` creates a frozen Parquet mapping and
   `*.summary.json` audit from all training shards.
2. `apply_adaptive_code_mapping` maps every data shard.
3. `finalize_adaptive_code_metadata` reconciles `codes.parquet` with the mapped
   vocabulary.

Run the standard `extract_code_metadata` stage before these stages. Run later
per-code metadata stages after `finalize_adaptive_code_metadata`.

#### Hierarchy configuration

Set `minimum_count` and character-position widths under `hierarchies`. ATC and
SKS defaults are in
[`default_adaptive_code_mapping.yaml`](./configs/MEDS/default_adaptive_code_mapping.yaml).

Override a built-in namespace or add a new one as needed:

```yaml
hierarchies:
  MY_NAMESPACE:
    levels: [2, 4, 6]
```

SKS defaults exclude level 1 because an ICD-10/SKS leading letter can span
clinical chapters; for example, `D` covers parts of both neoplasm and blood
disorder chapters. ATC keeps level 1 because it represents the 14 official
anatomical groups. Either default can be overridden.

### Numeric-value encoding

Use `aggregate_numeric_metadata` followed by `annotate_numeric_values`. Add
`join_numeric_bins` only for joined lab-and-bin model inputs.

Defaults are defined in
[`default_numeric_values.yaml`](./configs/MEDS/default_numeric_values.yaml).

- `numeric_value_columns` names the source and derived columns.
- `numeric_value_column_groups` selects transforms, bounds, and derived columns.

### Using externally fitted metadata

To use externally fitted artifacts, omit `fit_adaptive_code_mapping` and/or
`aggregate_numeric_metadata`, then configure their consumers:

```yaml
- extract_code_metadata
- apply_adaptive_code_mapping:
    mapping_filepath: ${oc.env:EXTERNAL_MAPPING_FP}
- finalize_adaptive_code_metadata:
    mapping_filepath: ${oc.env:EXTERNAL_MAPPING_FP}
- annotate_numeric_values:
    numeric_metadata_filepath: ${oc.env:EXTERNAL_NUMERIC_METADATA_FP}
```

External files are fallbacks; a local fit takes precedence if its fit stage is
present. Code mappings may be JSON or Parquet and must contain `code` and
`adaptive/mapped_code`.
