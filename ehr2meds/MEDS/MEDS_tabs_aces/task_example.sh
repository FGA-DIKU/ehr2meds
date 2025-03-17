#!/bin/bash
#BSUB -J meds_task               # Job name
#BSUB -q p1                      # Queue name
#BSUB -n 8                       # Number of CPU cores
#BSUB -R "span[hosts=1]"         # All cores on the same host
#BSUB -R "rusage[mem=5GB]"       # Memory per core
#BSUB -M 5GB                     # Memory limit per process
#BSUB -W 5:00                    # Wall-clock time limit
#BSUB -o run_%J.out              # Standard output file
#BSUB -e run_%J.err              # Error file
##BSUB -B                        # Send email when job starts
##BSUB -N                        # Send email when job finishes

set -e  # Exit immediately if a command exits with a non-zero status

# Perform the editable installation
#conda init
#conda activate meds_tabs
#source meds_tabs activate
cd MEDS-DEV
pip install -e .
cd ../
echo "Editable installation completed successfully."


#source MEDS activate

export HYDRA_FULL_ERROR=1

# Set MEDS_ROOT_DIR and DATASET_NAME
export MEDS_ROOT_DIR=$1
export DATASET_NAME=$2
export TASK_NAME=$3


# export MEDS_ROOT_DIR="./MEDS_data_labs/"  # Replace with your actual MEDS root directory
# export DATASET_NAME="CPH_data"  # Replace with your actual dataset name
# #export TASK_NAME="readmission/general_hospital/30d"
# export TASK_NAME="cancer/all_cancers" #"cancer/breast" #"mortality/in_icu/first_24h" 


echo "MEDS_ROOT_DIR: $MEDS_ROOT_DIR"
echo "DATASET_NAME: $DATASET_NAME"

# Run the command
MEDS-DEV/src/MEDS_DEV/helpers/extract_task.sh "$MEDS_ROOT_DIR" "$DATASET_NAME" "$TASK_NAME"