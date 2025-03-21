#!/usr/bin/env python3
import argparse
import os

import yaml
from azure.ai.ml import Input, MLClient, Output, command
from azure.identity import DefaultAzureCredential


def main():
    # Set up command line arguments.
    parser = argparse.ArgumentParser(description="Run PREMEDS job with Azure ML.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file."
    )
    parser.add_argument(
        "--compute",
        type=str,
        default="CPU-20-LP",
        help="Optional compute target.",
    )
    args = parser.parse_args()

    # Load the configuration from the provided YAML file.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract values from the configuration.
    input_uri = config.get("input_uri")
    output_uri = config.get("output_uri")
    pipeline_config_fp = config.get("pipeline_config_path")
    event_config_fp = config.get("event_config_path")

    # Use the command-line compute target if provided; otherwise, use the config file value.
    compute_target = args.compute

    # Create the ML client using the default Azure credential.
    ml_client = MLClient.from_config(DefaultAzureCredential())

    # Define inputs and outputs using the paths from the configuration.
    inputs = {"input_dir": Input(type="uri_folder", path=input_uri)}
    outputs = {
        "output_dir": Output(path=output_uri, type="uri_folder", mode="rw_mount")
    }

    # Get the directory where the run.py script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Full path to run.sh
    run_sh_path = os.path.join(script_dir, "run.sh")

    # Define and submit the job.
    job = command(
        code=".",  # Folder with source code.
        # Pass the input, pipeline config, event config, and output paths as arguments.
        command=f'bash {run_sh_path} ${{inputs.input_dir}} "{pipeline_config_fp}" "{event_config_fp}" ${{outputs.output_dir}}',
        inputs=inputs,
        outputs=outputs,
        environment="MEDS_transform@latest",
        compute=compute_target,
    )

    ml_client.create_or_update(job, experiment_name="MEDS")


if __name__ == "__main__":
    main()
