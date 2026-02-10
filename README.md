
# EHR2MEDS

## Overview

EHR2MEDS is a tool that formats dumps of Electronic Health Records (EHR) and converts them to MEDS (Medical Event Data Set) to be compatible with [CORE-BEHRT](https://github.com/FGA-DIKU/EHR).

1. **Raw → PREMEDS Conversion:**  
   Converts raw EHR data into a preliminary format called preMEDS.  
   You need to run [main_azure.py](./ehr2meds/PREMEDS/main_azure.py) to run the PREMEDS conversion.
   For instructions on how to run on azure, see the [Azure README](./ehr2meds/PREMEDS/azure/README.md).
   Example:

   ```bash
   python -m ehr2meds.PREMEDS.azure premeds <compute> -c <config_path>
   ```

   example:

   ```bash
   python -m ehr2meds.PREMEDS.azure premeds CPU-20-LP -c ehr2meds/example_configs/premeds/minimal.yaml
   ```

   Example configuration files can be found in the [PREMEDS/configs](./ehr2meds/PREMEDS/configs).

2. **PREMEDS → MEDS Conversion:**  
   Transforms preMEDS data into a finalized MEDS cohort format.  
   You need to run [run.sh](./ehr2meds/MEDS/MEDS_transform/run.sh) to run the MEDS conversion.
   For instructions ons how to run on azure, see the [MEDS Transform README](./ehr2meds/MEDS/MEDS_transform/README.md).

   ```bash
   bash -m ehr2meds/MEDS/MEDS_transform/run.sh <input_dir> <pipeline_config> <event_config> <output_dir> 
   ```

   Example:

   ```bash
   bash -m ehr2meds/MEDS/MEDS_transform/run.sh outputs/preMEDS ehr2meds/example_configs/meds/pipeline.yaml ehr2meds/example_configs/meds/event_minimal.yaml outputs/MEDS
   ```

   Example configuration files can be found in the [MEDS/MEDS_transform/configs](./ehr2meds/MEDS/MEDS_transform/configs).

3. **Normalization:**  
   (Optional) Normalizes lab test data before MEDS conversion.  
   You need to run [main_normalise.py](./ehr2meds/PREMEDS/main_normalise.py) to run the normalization.
   Example:

   ```bash
   python -m ehr2meds.PREMEDS.azure normalise <compute> -c <config_path>
   ```

   Example:

   ```bash
   python -m ehr2meds.PREMEDS.azure normalise CPU-20-LP -c ehr2meds/example_configs/premeds/normalise.yaml
   ```

   Example configuration files can be found in the [PREMEDS/configs](./ehr2meds/PREMEDS/configs).
