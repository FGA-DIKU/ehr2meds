# PREMEDS Pipeline

The **PREMEDS** pipeline is designed to process and extract relevant data from sp and register data for further transformation to MEDS.

## Overview

The pipeline is built from several modules:

- **extractor.py**  
  Contains the `PREMEDSExtractor` class, which coordinates loading patient data, processing various medical concepts, and handling register-specific concepts.
  
- **register.py**  
  Implements register-specific processing, including mapping, column unrolling, and data cleanup.
  
- **azure.py**  
  Provides data loaders for both local file system and Azure-based datasets. The data loader is selected via a factory function based on the environment.
  
- **data_handling.py**  
  Defines the `DataConfig` and `DataHandler` classes that standardize data loading and saving operations.

## Configuration

The pipeline is controlled via a YAML configuration file (for example, `MEDS_local.yaml`). This file determines the environment, file locations, processing options, and how the data should be transformed.

Below is an updated explanation—both in plain text and in an excerpt from the configuration file—that describes in more detail what processing steps are applied for regular concepts versus register concepts, and how you should set these in your YAML config.

---

### Concepts Processing

**What Happens:**  
For each concept (e.g., admissions, diagnosis, labtest, etc.), the pipeline follows these steps:

1. **Data Loading:**  
   Depending on the file size and configuration, data is either loaded entirely (if small) or in chunks (if large). The file is read from the location specified in the `dump_path` (and optionally, the filename in the config).
2. **Column Renaming:**  
   The pipeline uses the `rename_columns` mapping to rename and subset the original columns. This step standardizes the column names across all concepts. For example, you might map “Noteret_dato” to “timestamp” or “CPR_hash” to “subject_id.”
3. **Data Cleaning:**  
   Additional cleaning is applied—this can include filling missing values (`fillna` settings), converting numeric columns (via the `numeric_columns` key), and even applying custom transformations like adding a code prefix (`code_prefix`).
4. **Saving Results:**  
   After processing (and potentially after processing each chunk), the formatted data is saved into a standardized format (like CSV or Parquet) in the output directory defined in `paths.output_dir`.

**How to Set in Config:**  
Under the `concepts` section, you provide keys such as:

- `filename`: Name of the dataset file (e.g., `CPMI_Diagnoser.parquet`).
- `rename_columns`: A mapping that defines the column renaming (for example, `{ Diagnosekode: code, Noteret_dato: timestamp, CPR_hash: subject_id }`).
- `fillna` (optional): Instructions for filling missing values in certain columns.
- `numeric_columns` (optional): List of columns to be explicitly converted to numeric types.
- `code_prefix` (optional): A prefix to add to code values.
- Other keys that control special processing (for example, `use_adm_move` in admissions).

A simplified excerpt of the config for concepts might be:

```yaml
concepts:
  diagnosis:
    filename: CPMI_Diagnoser.parquet
    rename_columns:
      Diagnosekode: code
      Diagnose: fill_code
      Noteret_dato: timestamp
      CPR_hash: subject_id
    fillna:
      code:
        column: fill_code
        regex: '\((D[^)]+)\)'
    code_prefix: "D_"

  labtest:
    filename: CPMI_Labsvar.parquet
    rename_columns:
      BestOrd: code
      Prøvetagningstidspunkt: timestamp
      Bestillingsdato: fill_timestamp
      Resultatværdi: numeric_value
      CPR_hash: subject_id
    fillna:
      timestamp:
        column: fill_timestamp
    code_prefix: "L_"
    numeric_columns: [numeric_value]
```

---

### Register Concepts Processing

**Steps Involved:**

1. **Data Loading:**  
   Register concept files are loaded (they might come in different file formats such as ASC files). Like standard concepts, they can be processed in chunks or loaded entirely based on configuration.

2. **Column Renaming:**  
   The register concept data is first renamed using the dedicated `rename_columns` mapping (e.g., mapping “dw_ek_kontakt” to “contact_id”).

3. **Mapping via External Link Files:**  
   Register concepts commonly require further processing via one or more external mapping steps. These steps use a list of mapping definitions provided under the `mappings` key:
    - **source_column:** The column in the register data that serves as the lookup key.
    - **via_file:** The file (link dataset) that contains the mapping information.
    - **join_on:** The column in the mapping file to join on.
    - **target_column:** The column in the mapping file that holds the value to be merged.
    - **rename_to:** How to rename the merged target column in the output.
    - **drop_source (optional):** Whether to drop the source column after merging.

   The register processing function will iterate through these mappings and merge the link data into the register dataset.  
   **Config Setting:**  
   - Under `register_concepts`, provide a `mappings` list with each mapping’s parameters.

4. **Unrolling Columns (if applicable):**  
   Just as with standard concepts, you can optionally include an `unroll_columns` key in the register concepts configuration. This is used when certain columns need to be expanded (for example, if a register record holds multiple dates or codes in one cell).

5. **Saving the Processed Data:**  
   After all mapping and transformation steps, the processed register concept data is saved to the output directory.

Example snippet in config for a register concept:

```yaml
register_concepts:
  register_diagnosis:
    filename: diagnoser.asc
    rename_columns:
      diagnosekode: code
      diagnosetype: code_type
      dw_ek_kontakt: contact_id
    code_prefix: "RD_"
    mappings:
      - source_column: contact_id
        via_file: kontakter.parquet
        join_on: dw_ek_kontakt
        target_column: PID
        rename_to: subject_id
      - source_column: contact_id
        via_file: kontakter.parquet
        join_on: dw_ek_kontakt
        target_column: dato_slut
        rename_to: timestamp
        drop_source: true
    # Optionally unroll any multi-value columns if needed:
    unroll_columns:
      column: code
      delimiter: ";"
```
