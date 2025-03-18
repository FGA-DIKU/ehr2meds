"""
Component for running the PREMEDS pipeline on Azure.
"""

# Input/Output configurations
INPUTS = {
    "concepts": {
        "type": "uri_folder",
        "key": "paths.concepts",
        "description": "Input directory containing SP-dumps data",
    },
    "register_concepts": {
        "type": "uri_folder",
        "key": "paths.register_concepts",
        "optional": True,
        "description": "Input directory containing FSEID data (optional)",
    },
    "pid_link": {
        "type": "uri_file",
        "key": "paths.pid_link",
        "description": "Path to mapping.csv file",
    },
}

OUTPUTS = {
    "output": {
        "type": "uri_folder",
        "key": "paths.output",
        "description": "Output directory for processed data",
    }
}

if __name__ == "__main__":
    from util import run_main
    from ehr2meds.PREMEDS import main_azure

    run_main(main_azure.run_pre_MEDS, INPUTS, OUTPUTS)
