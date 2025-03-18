"""
Component for running the normalization pipeline on Azure.
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
        "description": "Output directory for normalized data",
    }
}

if __name__ == "__main__":
    from ehr2meds.PREMEDS.azure.util import run_main
    from ehr2meds.PREMEDS import main_normalise

    run_main(main_normalise.my_app, INPUTS, OUTPUTS)
