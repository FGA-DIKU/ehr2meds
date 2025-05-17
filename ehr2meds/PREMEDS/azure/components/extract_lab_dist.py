"""
Component for running the extraction of lab distribution on Azure.
"""

# Input/Output configurations
INPUTS = {
    "input_data": {
        "type": "uri_file",
        "key": "paths.input",
        "description": "Input file containing lab test data",
    },
}

OUTPUTS = {
    "output": {
        "type": "uri_folder",
        "key": "paths.output_dir",
        "description": "Output directory for lab distribution data",
    }
}

if __name__ == "__main__":
    from ehr2meds.PREMEDS.azure.util import run_main
    from ehr2meds.PREMEDS import extract_lab_dist

    run_main(extract_lab_dist.my_app, INPUTS, OUTPUTS)
