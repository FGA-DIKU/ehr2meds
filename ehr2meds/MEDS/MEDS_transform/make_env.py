from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

mlc = MLClient.from_config(DefaultAzureCredential())

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="../envs/MEDS_transform.yaml",
    name="MEDS_transform",
    description="Environment for MEDS-transforms",
)
mlc.environments.create_or_update(env)
