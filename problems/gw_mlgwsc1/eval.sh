#!/bin/bash

# Parse command line arguments with defaults
foreground_path=$1
foreground_events_path=$2
background_events_path=$3
injection_path=$4
output_path=$5

# Evaluate 
# Determine the path based on hostname
hostname=$(hostname)
if [ "$hostname" = "login05" ]; then
    # Dongfang
    python /work1/hewang/HW/ml-mock-data-challenge-1/evaluate.py \
        --injection-file ${injection_path} \
        --foreground-events ${foreground_events_path} \
        --foreground-files ${foreground_path} \
        --background-events ${background_events_path} \
        --output-file ${output_path} \
        --verbose
elif [ "$hostname" = "1f7472014c87-" ]; then
    # A800
    python /home/nvme0n1/data/ml-mock-data-challenge-1/evaluate.py \
        --injection-file ${injection_path} \
        --foreground-events ${foreground_events_path} \
        --foreground-files ${foreground_path} \
        --background-events ${background_events_path} \
        --output-file ${output_path} \
        --verbose
elif [ "$hostname" = "Gravitation-wave" ]; then
    # A6000
    python /home/main/gwtoolkit_project/gwtoolkit/benchmark/ml-mock-data-challenge-1/evaluate.py \
        --injection-file ${injection_path} \
        --foreground-events ${foreground_events_path} \
        --foreground-files ${foreground_path} \
        --background-events ${background_events_path} \
        --output-file ${output_path} \
        --verbose
else
    # Generic path - use environment variable or default
    ML_CHALLENGE_PATH=${ML_CHALLENGE_PATH:-"/path/to/ml-mock-data-challenge-1"}
    if [ ! -d "$ML_CHALLENGE_PATH" ]; then
        echo "Error: ML_CHALLENGE_PATH not found: $ML_CHALLENGE_PATH"
        echo "Please set ML_CHALLENGE_PATH environment variable to the correct path"
        echo "Example: export ML_CHALLENGE_PATH=/your/path/to/ml-mock-data-challenge-1"
        exit 1
    fi
    python ${ML_CHALLENGE_PATH}/evaluate.py \
        --injection-file ${injection_path} \
        --foreground-events ${foreground_events_path} \
        --foreground-files ${foreground_path} \
        --background-events ${background_events_path} \
        --output-file ${output_path} \
        --verbose
fi

# Plot results
# Determine the path based on hostname
hostname=$(hostname)
if [ "$hostname" = "login05" ]; then
    # Dongfang
    python /work1/hewang/HW/ml-mock-data-challenge-1/contributions/sensitivity_plot.py \
        --files ${output_path} \
        --output ${output_path}.png \
        --no-tex
elif [ "$hostname" = "1f7472014c87-" ]; then
    # A800
    python /home/nvme0n1/data/ml-mock-data-challenge-1/contributions/sensitivity_plot.py \
        --files ${output_path} \
        --output ${output_path}.png \
        --no-tex
elif [ "$hostname" = "Gravitation-wave" ]; then
    # A6000
    python /home/main/gwtoolkit_project/gwtoolkit/benchmark/ml-mock-data-challenge-1/contributions/sensitivity_plot.py \
        --files ${output_path} \
        --output ${output_path}.png \
        --no-tex
else
    # Generic path - use environment variable or default
    ML_CHALLENGE_PATH=${ML_CHALLENGE_PATH:-"/path/to/ml-mock-data-challenge-1"}
    if [ ! -d "$ML_CHALLENGE_PATH" ]; then
        echo "Error: ML_CHALLENGE_PATH not found: $ML_CHALLENGE_PATH"
        echo "Please set ML_CHALLENGE_PATH environment variable to the correct path"
        echo "Example: export ML_CHALLENGE_PATH=/your/path/to/ml-mock-data-challenge-1"
        exit 1
    fi
    python ${ML_CHALLENGE_PATH}/contributions/sensitivity_plot.py \
        --files ${output_path} \
        --output ${output_path}.png \
        --no-tex
fi
