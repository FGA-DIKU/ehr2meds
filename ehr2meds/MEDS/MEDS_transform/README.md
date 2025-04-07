# MEDS Data Processing Pipeline

This repository contains the necessary scripts and configuration files to process PREMEDS data using Azure Machine Learning. The pipeline extracts data from a raw PREMEDS dataset and transforms it into a MEDS cohort format using a multi-step process.

## Prerequisites

- **Azure ML Workspace:** When on azure, activate conda environment with sdkv2. E.g. conda activate azureml_py310_sdkv2

## Configuration Files

### Main Config File

The main configuration file defines the global parameters for your job including the input/output data paths, as well as the file paths for your pipeline and event configuration files. It can also include a default compute target (which can be overridden via the command line).

Example: `config/config.yaml`

```yaml
# Main configuration file for PREMEDS job

# Directory containing raw PREMEDS data (input)
input_uri: ".../paths/your_preMEDS_input/"

# Directory where processed MEDS data will be stored (output)
output_uri: ".../paths/your_MEDS_output/"

# File path to the pipeline configuration for data processing
pipeline_config_path: ".../configs/pipeline.yaml"

# File path to the event conversion configuration
event_config_path: ".../configs/event.yaml"

# Optional compute target (can be overridden via command line)
compute_target: "CPU-20-LP"
```

### Pipeline Config File

The pipeline configuration file specifies the steps and environment variables needed for the data extraction and transformation process.

Example: `pipeline.yaml`

```yaml
defaults:
  - _extract
  - _self_

description: |-
  This pipeline extracts the preMEDS dataset in longitudinal, sparse form from an input dataset meeting
  select criteria and converts them to the flattened MEDS format. You can control the key arguments to this
  pipeline by setting environment variables:

  export EVENT_CONVERSION_CONFIG_FP=event_configs.yaml       # Path to your event conversion config
  export PREMEDS_INPUT_DIR=../preMEDS_w_meta                    # Path to the output dir of the pre-MEDS step
  export PREMEDS_MEDS_COHORT_DIR=../MEDS_new_data               # Path to where you want the dataset to live

event_conversion_config_fp: ${oc.env:EVENT_CONVERSION_CONFIG_FP}

input_dir: ${oc.env:PREMEDS_INPUT_DIR}
cohort_dir: ${oc.env:PREMEDS_MEDS_COHORT_DIR}

etl_metadata:
  dataset_name: preMEDS
  dataset_version: 1.0

stage_configs:
  shard_events:
    infer_schema_length: 999999999

stages:
  - shard_events
  - split_and_shard_subjects
  - convert_to_sharded_events
  - merge_to_MEDS_cohort
  - extract_code_metadata
  - finalize_MEDS_metadata
  - finalize_MEDS_data
```

### Event Config File

The event configuration file maps the columns and formats for various event types. Customize these mappings based on your dataset.
!IMPORTANT: Specify the time_format correctly. Otherwise, meds-transforms will remove rows not matching the time_format.!

Example: `event.yaml`

```yaml
subject_id_col: subject_id

subject:
  subject_id_col: subject_id
  dob:
    code: 
      - DOB
    time: col(birthdate)
    time_format: "%Y-%m-%d %H:%M:%S"
  dod:
    code: 
      - DOD
    time: col(deathdate)
    time_format: "%Y-%m-%d %H:%M:%S"
  gender:
    code:
      - "GENDER"
      - col(gender)
    time: null    
admissions:
  admissions:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d %H:%M:%S"

diagnosis:
  diagnosis:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d"

medication:
  medication:
    code:
      - col(code)
    time: col(timestamp)
    time_format:  "%Y-%m-%d %H:%M:%S"

procedure:
  procedure:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d %H:%M:%S"

labtest_norm_min_max:
  labtest_norm_min_max:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d %H:%M:%S"
    numeric_value: col(numeric_value)

register_medication:
  register_medication:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d %H:%M:%S"

register_diagnosis:
  register_diagnosis:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d %H:%M:%S"
    
register_procedures_other:
  register_procedures_other:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d %H:%M:%S"

register_procedures_surgery:
  register_procedures_surgery:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d %H:%M:%S"

register_contacts:
  register_contacts:
    code:
      - col(code)
    time: col(timestamp)
    time_format: "%Y-%m-%d %H:%M:%S"
```

When executed, `run.py` will submit a job to your Azure ML workspace. The job calls the `run.sh` bash script with the following parameters:

- **Input Directory:** Path to raw PREMEDS data.
- **Pipeline Config File Path:** Path to the pipeline configuration.
- **Event Config File Path:** Path to the event conversion configuration.
- **Output Directory:** Path to store the processed MEDS data.
