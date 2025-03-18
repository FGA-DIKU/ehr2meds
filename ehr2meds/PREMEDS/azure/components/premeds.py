"""
Component for running the PREMEDS pipeline on Azure.
"""

# Input/Output configurations
INPUTS = {
    "input_dir": {
        "type": "uri_folder",
        "key": "paths.input_dir",
        "description": "Input directory containing EHR data",
    }
}

OUTPUTS = {
    "output_dir": {
        "type": "uri_folder",
        "key": "paths.output_dir",
        "description": "Output directory for processed data",
    }
}

if __name__ == "__main__":
    from util import run_main
    from ehr2meds.PREMEDS import main_azure

    run_main(main_azure.run_pre_MEDS, INPUTS, OUTPUTS)
