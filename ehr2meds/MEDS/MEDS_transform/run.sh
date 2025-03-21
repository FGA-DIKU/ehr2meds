#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Updated Script for PREMEDS Data Processing
#
# This script processes PREMEDS data through several steps including raw data
# conversion, event conversion, and pipeline execution. All required paths (raw
# data, pipeline configuration, event conversion configuration, and output directory)
# are provided via command-line arguments.
#
# Usage:
#   $0 <PREMEDS_RAW_DIR> <PIPELINE_CONFIG_FP> <EVENT_CONFIG_FP> <PREMEDS_OUTPUT_DIR> [do_unzip=true|do_unzip=false]
#
# Arguments:
#   PREMEDS_INPUT_DIR         Directory containing raw PREMEDS data files.
#   PIPELINE_CONFIG_FP      File path for the pipeline configuration.
#   EVENT_CONFIG_FP         File path for the event conversion configuration.
#   MEDS_OUTPUT_DIR      Output directory for processed MEDS data.
#   (OPTIONAL) do_unzip flag: Set do_unzip=true to unzip CSV.GZ files before processing.
#
# Example:
#   bash run.sh /data/raw configs/pipeline.yaml configs/event.yaml /data/output do_unzip=true
# -----------------------------------------------------------------------------

# Exit immediately if any command exits with a non-zero status.
set -e

# Function to display help message
function display_help() {
    echo "Usage: $0 <PREMEDS_INPUT_DIR> <PIPELINE_CONFIG_FP> <EVENT_CONFIG_FP> <MEDS_OUTPUT_DIR> [do_unzip=true|do_unzip=false]"
    echo
    echo "This script processes PREMEDS data through several steps, including raw data conversion,"
    echo "event conversion, and pipeline execution. It uses the provided pipeline and event conversion"
    echo "configuration files for processing."
    echo
    echo "Arguments:"
    echo "  PREMEDS_INPUT_DIR         Directory containing raw PREMEDS data files."
    echo "  PIPELINE_CONFIG_FP      File path for the pipeline configuration."
    echo "  EVENT_CONVERSION_CONFIG_FP    File path for the event conversion configuration."
    echo "  MEDS_OUTPUT_DIR         Output directory for processed MEDS data."
    echo "  (OPTIONAL) do_unzip flag: Set do_unzip=true to unzip CSV.GZ files before processing."
    echo
    echo "Options:"
    echo "  -h, --help          Display this help message and exit."
    exit 1
}

# Determine the project root directory
# First get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Then go up 3 directories to the project root (adjust if your directory structure is different)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
echo "Project root detected at: $PROJECT_ROOT"

# Unset SLURM_CPU_BIND in case you're running on a SLURM node with parallelism.
unset SLURM_CPU_BIND

# Check if help is requested.
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    display_help
fi

# Check for mandatory parameters (we now require 4 arguments)
if [ "$#" -lt 4 ]; then
    echo "Error: Incorrect number of arguments provided."
    display_help
fi

# Assign mandatory parameters
export PREMEDS_INPUT_DIR=$1

# Handle config paths - make them absolute from project root if they're relative
if [[ "$2" != /* ]]; then
    # If path doesn't start with /, it's relative - resolve against project root
    export PIPELINE_CONFIG_FP="$PROJECT_ROOT/$2"
    echo "Resolved pipeline config: $PIPELINE_CONFIG_FP"
else
    # Path is already absolute
    export PIPELINE_CONFIG_FP="$2"
fi

if [[ "$3" != /* ]]; then
    # If path doesn't start with /, it's relative - resolve against project root
    export EVENT_CONVERSION_CONFIG_FP="$PROJECT_ROOT/$3"
    echo "Resolved event conversion config: $EVENT_CONVERSION_CONFIG_FP"
else
    # Path is already absolute
    export EVENT_CONVERSION_CONFIG_FP="$3"
fi

export MEDS_OUTPUT_DIR=$4
shift 4

# Handle optional do_unzip flag
_DO_UNZIP_ARG_STR=""
if [ "$#" -ge 1 ]; then
  case "$1" in
    do_unzip=*)
      _DO_UNZIP_ARG_STR="$1"
      shift 1
      ;;
  esac
fi

DO_UNZIP="false"
if [ -n "$_DO_UNZIP_ARG_STR" ]; then
  case "$_DO_UNZIP_ARG_STR" in
    do_unzip=true)
      DO_UNZIP="true"
      ;;
    do_unzip=false)
      DO_UNZIP="false"
      ;;
    *)
      echo "Error: Invalid do_unzip value. Use 'do_unzip=true' or 'do_unzip=false'."
      exit 1
      ;;
  esac
  echo "Setting DO_UNZIP=${DO_UNZIP}"
fi

# If unzipping is enabled, unzip all .csv.gz files under the raw data directory.
if [ "$DO_UNZIP" == "true" ]; then
  GZ_FILES="${PREMEDS_INPUT_DIR}/*/*.csv.gz"
  if compgen -G "$GZ_FILES" > /dev/null; then
    echo "Unzipping csv.gz files matching ${GZ_FILES}."
    for file in $GZ_FILES; do
      gzip -d --force "$file"
    done
  else
    echo "No csv.gz files found to unzip at ${GZ_FILES}."
  fi
else
  echo "Skipping unzipping of csv.gz files."
fi

# Export configuration file paths so that downstream tools can access them.
export EVENT_CONVERSION_CONFIG_FP
export PIPELINE_CONFIG_FP

echo "Running extraction pipeline."
# Execute the PREMEDS transform runner with the pipeline configuration.
# Any additional arguments are forwarded.
MEDS_transform-runner "pipeline_config_fp=${PIPELINE_CONFIG_FP}" "$@"

echo "Starting cleanup..."
# Example cleanup: List any empty files or directories.
find . -empty -exec ls -ld {} \;
echo "Cleanup complete."
