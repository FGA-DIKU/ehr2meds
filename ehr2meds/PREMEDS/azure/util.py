"""
Utility functions for running Azure ML jobs.
This code is largely taken from corebehrt https://github.com/FGA-DIKU/EHR
"""

import argparse
from datetime import datetime
import yaml
import importlib
from os.path import join
from typing import Tuple, Dict, Any, Optional, Callable

# Azure imports
from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential

AZURE_CONFIG_FILE = "azure_job_config.yaml"


def ml_client() -> MLClient:
    """Returns the Azure MLClient."""
    return MLClient.from_config(DefaultAzureCredential())


def create_job(
    name: str,
    config: dict,
    compute: str,
    register_output: dict = dict(),
    log_system_metrics: bool = False,
) -> command:
    """Creates the Azure command/job object."""
    # Load component
    component = importlib.import_module(f"ehr2meds.PREMEDS.azure.components.{name}")

    return setup_job(
        name,
        inputs=component.INPUTS,
        outputs=component.OUTPUTS,
        config=config,
        compute=compute,
        register_output=register_output,
        log_system_metrics=log_system_metrics,
    )


def setup_job(
    job: str,
    inputs: dict,
    outputs: dict,
    config: dict,
    compute: str,
    register_output: dict = dict(),
    log_system_metrics: bool = False,
) -> command:
    """Sets up the Azure job."""
    # Prepare command
    cmd = f"python -m ehr2meds.PREMEDS.azure.components.{job}"

    # Save config for job
    with open(AZURE_CONFIG_FILE, "w") as cfg_file:
        yaml.dump(config, cfg_file)

    # Prepare input and output paths
    input_values, input_cmds = prepare_job_command_args(config, inputs, "inputs")
    output_values, output_cmds = prepare_job_command_args(
        config, outputs, "outputs", register_output=register_output
    )

    # Add input and output arguments to cmd
    cmd += input_cmds + output_cmds

    # Add log_system_metrics if set
    if log_system_metrics:
        cmd += " --log_system_metrics"

    # Create job
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return command(
        code=".",
        command=cmd,
        inputs=input_values,
        outputs=output_values,
        environment="MEDS:2",
        compute=compute,
        name=f"{job}_{ts}",
    )


def run_job(job: command, experiment: str) -> None:
    """Starts the given job in the given experiment."""
    ml_client().create_or_update(job, experiment_name=experiment)


def run_main(main: Callable, inputs: Dict, outputs: Dict) -> None:
    """Implements a wrapper for running scripts on the cluster."""
    args = parse_args(inputs | outputs)
    prepare_config(args, inputs, outputs)
    main(AZURE_CONFIG_FILE)


def prepare_config(args: dict, inputs: dict, outputs: dict) -> None:
    """Prepares the config on the cluster."""
    with open(AZURE_CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    # Update input arguments in config file
    for arg, arg_cfg in (inputs | outputs).items():
        if args[arg] is None:
            if arg_cfg.get("optional", False):
                continue
            else:
                raise Exception(f"Missing argument '{arg}'")
        _cfg = cfg
        cfg_path = arg_cfg.get("key", f"paths.{arg}").split(".")
        for step in cfg_path[:-1]:
            _cfg = _cfg[step]
        _cfg[cfg_path[-1]] = args[arg]

    # Save updated config
    with open(AZURE_CONFIG_FILE, "w") as f:
        yaml.dump(cfg, f)


def parse_args(args: Dict) -> Dict[str, Any]:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    for arg in args:
        parser.add_argument(f"--{arg}", type=str)
    parser.add_argument("--log_system_metrics", action="store_true", default=False)
    return vars(parser.parse_args())


def prepare_job_command_args(
    config: dict, args: dict, _type: str, register_output: dict = dict()
) -> Tuple[dict, str]:
    """Prepare job arguments and command string."""
    assert _type in ("inputs", "outputs")

    job_args = dict()
    cmd = ""
    azure_arg_cls = Input if _type == "inputs" else Output

    for arg, arg_cfg in args.items():
        if value := get_path_from_cfg(config, arg, arg_cfg):
            job_args[arg] = azure_arg_cls(path=value, type=arg_cfg["type"])
            cmd += f" --{arg} ${{{_type}.{arg}}}"

            # Register output if needed
            if _type == "outputs" and arg in register_output:
                job_args[arg].name = register_output[arg]

    return job_args, cmd


def get_path_from_cfg(cfg: dict, arg: str, arg_cfg: dict) -> Optional[str]:
    """Get path from config."""
    steps = arg_cfg.get("key", f"paths.{arg}").split(".")

    for step in steps:
        if step not in cfg:
            if arg_cfg.get("optional", False):
                return None
            else:
                raise Exception(f"Missing required config item '{'.'.join(steps)}'")
        cfg = cfg[step]

    return map_azure_path(cfg)


def map_azure_path(path: str) -> str:
    """Maps path to correct Azure format."""
    if ":" not in path or path.startswith("azureml:"):
        return path

    dstore, tail = path.split(":", 1)
    dstore = dstore.replace("-", "_")

    if dstore in ("researcher_data", "sp_data"):
        return join("azureml://datastores", dstore, "paths", tail)
    else:
        return f"azureml:{path}"
