import sys
from azureml.core import Workspace, Environment, Experiment, ScriptRunConfig

ws = Workspace.from_config()

job_name = sys.argv[1]

env = Environment.get(ws, "MEDS", version="10")

ct = "CPU-20-LP"
src = ScriptRunConfig(
    source_directory=".", script=job_name + ".py", compute_target=ct, environment=env
)
Experiment(ws, name="MEDS").submit(src)
