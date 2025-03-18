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


def main(config_path: str) -> None:
    """
    Main function for running the PREMEDS pipeline.

    :param config_path: Path to configuration file.
    """
    from ehr2meds.PREMEDS import run_pipeline

    run_pipeline(config_path)


if __name__ == "__main__":
    from ..util import run_main

    run_main(main, INPUTS, OUTPUTS)
