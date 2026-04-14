
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
   python ehr2meds/convert_raw_to_premeds.py --config-name preMEDS/fetal_SP     
   ```

   Example configuration files can be found in the [configs/preMEDS](./ehr2meds/configs/preMEDS).
   
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
      ${EHR2MEDS_DATA}/preMEDS/fetal_data/SP \
      ${EHR2MEDS_CONFIGS}/MEDS/default_pipeline.yaml \
      ${EHR2MEDS_CONFIGS}/MEDS/default_event.yaml \
      ${EHR2MEDS_DATA}/MEDS/SP
   ```

   Example configuration files can be found in the [configs/MEDS](./ehr2meds/configs/MEDS).

3. **Normalization:**  
   (Optional) Normalizes lab test data before MEDS conversion.  
   You need to run [normalise_premeds.py](./ehr2meds/normalise_premeds.py) to run the normalization.
   Example:

   ```bash
   python ehr2meds/normalise_premeds.py --config-name <config_path> 
   ```

   Example:

   ```bash
   python ehr2meds/normalise_premeds.py --config-name preMEDS/normalise     
   ```

   Example configuration files can be found in the [configs/preMEDS](./ehr2meds/configs/preMEDS).
