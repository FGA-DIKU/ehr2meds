# Running PREMEDS on Azure

## Prerequisites

- Azure ML workspace set up
- Azure ML SDK v2 installed (`pip install azure-ai-ml`)
- Valid Azure credentials configured
- PREMEDS package installed

## Basic Usage

To run a PREMEDS job on Azure, use the following command:

```bash
python -m ehr2meds.PREMEDS.azure premeds <compute> [options]
```

### Required Arguments

- `compute`: Name of the Azure compute target to use (e.g., "CPU-20-LP")

### Optional Arguments

- `-c, --config`: Path to configuration file (default: ./configs/premeds.yaml)
- `-e, --experiment`: Name of the experiment (default: premeds_runs)
- `-o, --register_output`: Register outputs as Azure ML assets (format: `<output_id>=<asset_name>`)
- `-lsm, --log_system_metrics`: Enable logging of system metrics

### Example

```bash
# Basic run
python -m ehr2meds.PREMEDS.azure premeds CPU-20-LP

# With custom config and experiment name
python -m ehr2meds.PREMEDS.azure premeds CPU-20-LP -c configs/MEDS_azure.yaml -e premeds

# Register output as Azure ML asset
python -m ehr2meds.PREMEDS.azure premeds CPU-20-LP -o output_dir=premeds_v01
```

## Configuration File

The configuration file should specify:

- `paths.input_dir`: Input directory containing EHR data
- `paths.output_dir`: Output directory for processed data

Example config (yaml):

```yaml
paths:
  input_dir: "researcher_data:/path/to/input"
  output_dir: "researcher_data:/path/to/output"
```

Path formats supported:

- `azureml:*` - Azure ML asset paths
- `researcher_data:/path` - Paths on researcher data store
- `sp_data:/path` - Paths on sp data store
