
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
| `join_numeric_bins` | Optionally creates the "joined representation" of numeric values, such as `LAB_CODE//bin_3`, from the numeric bin index. |
| `bin_numeric_values_fast` | A faster, memory-efficient replacement for the standard MEDS-Transforms discrete binning stage. It rewrites codes using bin indices or interval labels. |

For combined numeric encoding, use `aggregate_numeric_metadata` followed by
`annotate_numeric_values`. Add `join_numeric_bins` afterwards only when the final
model input should contain joined lab-and-bin codes.

Shared numeric column names and stage defaults are defined in
`configs/MEDS/default_numeric_values.yaml`; pipeline configurations only
need to specify dataset- or run-specific overrides. Any setting can be overridden
under the relevant pipeline stage. The `numeric_value_column_groups` lists control
which transform, optional bound, and derived columns are used; column names and
derived outputs are configured through `numeric_value_columns`.
