import sys
from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig

ws = Workspace.from_config()

# The first argument is the job name; any additional arguments will be passed to the script
job_name = sys.argv[1]
job_args = sys.argv[2:]

env = Environment.get(ws, "MEDS", version="10")

ct = "CPU-20-LP"
src = ScriptRunConfig(
    source_directory=".",
    script=f"{job_name}.py",
    compute_target=ct,
    environment=env,
    arguments=job_args,  # Pass additional arguments here
)
Experiment(ws, name="MEDS").submit(src)
