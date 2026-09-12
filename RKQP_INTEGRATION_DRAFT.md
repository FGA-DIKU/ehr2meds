# RKQP quality-register integration draft

## Scope and source structure

The combined table contains one row per registered patient and 495 columns. `supertype` identifies the source registry:

| `supertype` | Registry | Rows |
| --- | --- | ---: |
| `LYFO` | Danish National Lymphoma Registry | 27,061 |
| `CLL` | Danish CLL Registry | 7,884 |
| `MM` | DaMyDa | 14,805 |

The table is sparse by registry. Common columns occur first, followed approximately by LYFO-specific, CLL-specific, and MM-specific blocks. The RKQP table should remain a distinct preMEDS input table; the `supertype` value should be retained for provenance but should not be added to every clinical token.

## Proposed namespaces

All quality-register-only concepts use an `RKQP_` prefix. This makes later vocabulary expansion explicit and prevents an apparently identical registry-derived value from silently colliding with an event from the administrative or laboratory feeds.

| Namespace | Purpose | Representation |
| --- | --- | --- |
| `RKQP_REGISTRY` | Source provenance | categorical |
| `RKQP_SUBTYPE` | Harmonized disease subtype | categorical |
| `RKQP_STAGE` | Named staging systems | `RKQP_STAGE//{system}//{value}` |
| `RKQP_PS` | Performance status | categorical |
| `RKQP_FEATURE` | Diagnosis findings and symptoms | `RKQP_FEATURE//{feature}//{value}` |
| `RKQP_BIOMARKER` | Molecular, FISH, and pathology findings | categorical or numeric |
| `RKQP_LABTEST` | Quality-register measurements | numeric value, with unit in code |
| `RKQP_TREATMENT` | Treatment component or regimen | categorical |
| `RKQP_RESPONSE` | Assessed treatment response | categorical |
| `RKQP_PROCEDURE` | Surgery, radiotherapy, or transplant | categorical |
| `RKQP_COMPLICATION` | Disease/treatment complication | categorical |
| `RKQP_REGION` | Register-provided region | categorical |

The namespace describes the clinical concept, while line number and component name belong in the payload. For example:

```text
RKQP_STAGE//ANN_ARBOR//3
RKQP_PS//2
RKQP_TREATMENT//LINE_1//REGIMEN//CHOP
RKQP_RESPONSE//LINE_1//CR
RKQP_LABTEST//HB_MMOL_L
```

## Cross-registry harmonization

These columns are already harmonized in the source table and should be used directly:

| Canonical concept | Source column | Treatment |
| --- | --- | --- |
| Diagnosis time | `date_diagnosis` | time anchor only |
| Sex | `sex` | omit: already supplied by the patient table |
| Broad subtype | `subtype` | `RKQP_SUBTYPE//{value}` |
| Performance status at diagnosis | `PS_diagnosis` | `RKQP_PS//{value}` |
| Hemoglobin | `HB_diagnosis` | numeric `RKQP_LABTEST//HB_MMOL_L` |
| Albumin | `ALB_diagnosis` | numeric `RKQP_LABTEST//ALBUMIN_G_L` |
| Creatinine | `CREA_diagnosis` | numeric `RKQP_LABTEST//CREATININE_UMOL_L` |
| LDH | `LDH_diagnosis` | numeric; unit must be confirmed |
| IgA | `IgA_diagnosis` | numeric; consolidated unit must be confirmed |
| IgG | `IgG_diagnosis` | numeric; consolidated unit must be confirmed |
| IgM | `IgM_diagnosis` | numeric; consolidated unit must be confirmed |

Do not separately emit `ALB_gL_diagnosis`/`ALB_uM_diagnosis`, `KREA_uM_diagnosis`/`KREA_mM_diagnosis`, or the unit-specific immunoglobulin columns when their consolidated counterpart is used. That would duplicate the same measurement.

The following similarly named variables represent the same general concept but must retain their named staging system:

| Columns | Canonical representation |
| --- | --- |
| `AA_stage_diagnosis` | `RKQP_STAGE//ANN_ARBOR//{value}` |
| `binet` | `RKQP_STAGE//BINET//{value}` |
| `ISS_diagnosis` | `RKQP_STAGE//ISS//{value}` |
| `RISS_diagnosis` | `RKQP_STAGE//RISS//{value}` |
| `IPI_score_diagnosis`, `aaIPI_score_diagnosis`, `IPS_score_diagnosis`, `FLIPI_diagnosis`, `FLIPI2_diagnosis` | distinct named score payloads, not one merged value |

`PS`, `PS_1st_line`, and `PS_2nd_line` use the same 0–4 encoding as `PS_diagnosis`, but belong at different time anchors. They share `RKQP_PS` rather than becoming distinct vocabularies.

## Numeric fields for the first pass

Only direct measurements are represented with `numeric_value`. Coded categories such as stage, performance status, yes/no flags, and response are categorical even when stored as numbers.

| Code | Column | Anchor | Status |
| --- | --- | --- | --- |
| `RKQP_LABTEST//HB_MMOL_L` | `HB_diagnosis` | diagnosis | include |
| `RKQP_LABTEST//WBC_10E9_L` | `WBC_diagnosis` | diagnosis | include after invalid/outlier handling is confirmed |
| `RKQP_LABTEST//PLATELETS_10E9_L` | `TRC_diagnosis` | diagnosis | include |
| `RKQP_LABTEST//LYMPHOCYTES_PERCENT` | `lymphocyte_percentage_diagnosis` | diagnosis | include |
| `RKQP_LABTEST//ALC_10E9_L` | `ALC_diagnosis` | diagnosis | include |
| `RKQP_LABTEST//ALBUMIN_G_L` | `ALB_diagnosis` | diagnosis | include |
| `RKQP_LABTEST//CALCIUM_MMOL_L` | `CA2_diagnosis` | diagnosis | include |
| `RKQP_LABTEST//CORRECTED_CALCIUM_MMOL_L` | `CA_albumin_corrected_diagnosis` | diagnosis | include |
| `RKQP_LABTEST//CREATININE_UMOL_L` | `CREA_diagnosis` | diagnosis | include |
| `RKQP_LABTEST//BETA2_MICROGLOBULIN_MG_L` | `B2M_diagnosis` | diagnosis | parse numeric strings; retain text for inequalities |
| `RKQP_LABTEST//LDH` | `LDH_diagnosis` | diagnosis | include after unit is confirmed |
| `RKQP_LABTEST//BILIRUBIN` | `bilirubin_diagnosis` | diagnosis | include after unit is confirmed |
| `RKQP_LABTEST//ALAT` | `ALAT_diagnosis` | diagnosis | include after unit is confirmed |
| `RKQP_LABTEST//BASP` | `BASP_diagnosis` | diagnosis | include after analyte/unit is confirmed |
| `RKQP_LABTEST//IGA` | `IgA_diagnosis` | diagnosis | include after consolidated unit is confirmed |
| `RKQP_LABTEST//IGG` | `IgG_diagnosis` | diagnosis | include after consolidated unit is confirmed |
| `RKQP_LABTEST//IGM` | `IgM_diagnosis` | diagnosis | include after consolidated unit is confirmed |

The observed `WBC_diagnosis` maximum is 31,124,925 and includes `-1`, so it needs explicit invalid-value handling before numeric metadata is fitted. Numeric plausibility bounds should be configured only after unit confirmation.

## Time anchoring

The quality-register row must be expanded into events at clinically meaningful times:

| Field group | Preferred anchor | Fallback |
| --- | --- | --- |
| Diagnosis, subtype, stage, baseline labs and biomarkers | `date_diagnosis` | none |
| First-line chemotherapy | `date_chemo_start_1st_line` | `date_treatment_1st_line` |
| First-line immunotherapy | `date_immuno_start_1st_line` | `date_treatment_1st_line` |
| First-line radiotherapy | `date_RT_1st_line` | `date_treatment_1st_line` |
| First-line response | `date_response_1st_line` | no event if missing |
| First-line surgery | `date_surgery_1st_line` | no event if missing |
| First-line transplant | `date_stem_cell_infusion_1st_line` or `date_transplant` | no event if missing |
| Second-line baseline/relapse | `date_relapse_confirmed_2nd_line` | `date_treatment_2nd_line` |
| Second-line treatment | component-specific start date | `date_treatment_2nd_line` or `date_treatment_2nd_line_start` |
| Second-line response | `date_response_2nd_line` | no event if missing |
| Follow-up/death | actual event date | exclude from first pretraining draft |

Dates are anchors and should not themselves be encoded as categorical tokens.

## Conservative first-pass selection

The first runnable version should include:

1. Registry provenance, subtype, stage, performance status, symptoms, disease extent, and direct baseline measurements.
2. Directly recorded biomarkers and FISH results.
3. Optionally, first- and second-line treatment components/regimens at their treatment dates for longitudinal pretraining.
4. Optionally and separately, response at the corresponding response date.
5. Optionally, surgery, radiotherapy, and transplant at their specific dates.

The following are excluded initially:

- Identifiers: `patientid`, `hospital_id`, `SHAK_resource_1st_line`, `shak`, municipality fields.
- Redundant demographics: `sex`, `date_birth`, derived age and age-threshold flags.
- Derived durations/outcomes: `time_OS`, `OS`, `time_to_death`, `time_to_treatment`, `time_to_TFS`, `TFS`, and `time_1st_to_2nd_treatment`.
- Prediction labels: `treated_within_90_days`, `died_within_90_days`, `dead`, and `treated`.
- Duplicate derived scores such as `*_score_minus_*` until there is a specific use case.
- Free text or very high-cardinality values until reviewed, including specified “other” fields.
- Treatment-plan flags until it is clear whether they describe intent at diagnosis or treatment actually delivered.

Excluding future-derived labels is important even though downstream windowing should normally prevent future events from entering a prediction context. It avoids making leakage dependent on every downstream task being configured perfectly.

## Implementation shape

The initial shaping implementation is `ehr2meds/prepare_rkkp.py`. It emits a long event table with source and timing provenance. Its default is deliberately baseline-only:

```bash
python ehr2meds/prepare_rkkp.py merged_rkkp.parquet rkkp_events.parquet
```

Post-diagnosis treatment events are opt-in:

```bash
python ehr2meds/prepare_rkkp.py merged_rkkp.parquet rkkp_events.parquet \
  --include-treatments
```

Responses require a separate explicit flag because a recorded response date does not prove that the value was available prospectively at that date:

```bash
python ehr2meds/prepare_rkkp.py merged_rkkp.parquet rkkp_events.parquet \
  --include-treatments --include-responses
```

Every output row retains `source_column`, `time_source`, and `temporal_confidence`. These fields make later leakage audits possible. The script also writes a JSON QC summary and removes WBC values outside `(0, 1000]` by default; the upper limit is configurable with `--wbc-max`.

For prediction-oriented fine-tuning, use the baseline-only output unless a task-specific censoring audit establishes that post-diagnosis events cannot cross the prediction boundary. Treatments are useful longitudinal pretraining signal, but they should not be enabled merely to improve separation in a baseline prediction task. Responses are even higher risk because registry abstraction may be retrospective.

Use the opt-in preMEDS configuration `configs/preMEDS/lymphoma_ngc_rkkp.yaml`. It inherits every existing lymphoma input and adds the shaped event file as `quality_registers`. Set `RKQP_EVENTS_FILE` before running preMEDS:

```bash
export RKQP_EVENTS_FILE=/absolute/path/to/rkkp_events.parquet
python ehr2meds/convert_raw_to_premeds.py --config-name preMEDS/lymphoma_ngc_rkkp
```

The RKQP event fragment is `configs/MEDS/rkkp_event.yaml`. Opt it into the unchanged lymphoma MEDS pipeline by exporting its absolute path:

```bash
export RKQP_EVENT_CONVERSION_CONFIG_FP="$EHR2MEDS_CONFIGS/MEDS/rkkp_event.yaml"
bash ehr2meds/convert_premeds_to_meds.sh \
  "$EHR2MEDS_DATA/preMEDS/lymphoma_ngc_rkkp" \
  "$EHR2MEDS_CONFIGS/MEDS/lymphoma_pipeline.yaml" \
  "$EHR2MEDS_CONFIGS/MEDS/lymphoma_ngc_event.yaml" \
  "$EHR2MEDS_DATA/MEDS/lymphoma_ngc_rkkp"
```

If `RKQP_EVENT_CONVERSION_CONFIG_FP` is unset, `lymphoma_pipeline.yaml` uses only the existing lymphoma event configuration and behaves as before.

Small bounded counts such as `n_regions_diagnosis`, `n_extranodal_regions_diagnosis`, treatment cycle counts, and radiotherapy fraction counts are exact categorical payloads. Continuous measurements and doses use `numeric_value` and therefore participate in the existing train-only, per-code numeric fitting. The QC report contains registry-stratified quantiles for every numeric source column and flags fields whose registry medians differ by at least five-fold.

Units that remain uncertain are deliberately represented as `UNIT_UNKNOWN`. The registry-stratified QC report should be reviewed before treating any flagged measurements as one shared numerical distribution.
