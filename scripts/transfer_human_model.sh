#!/bin/bash

base_dir=""
python_args=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d)
            base_dir="$2"
            shift 2
            ;;
        -m)
            human_model="$2"
            shift 2
            ;;
        *)
            python_args+=("$1")
            shift
            ;;
    esac
done

# Get base directory: use argument if provided, otherwise use env var, otherwise prompt
if [[ -z "$base_dir" ]]; then
    if [[ -n "$HUMOTO_DATASET_DIR" ]]; then
        base_dir="$HUMOTO_DATASET_DIR"
    else
        read -p "Enter base directory: " base_dir
    fi
fi

if [[ -z "$human_model" ]]; then
    read -p "Enter human model path: " human_model
fi

# Check if directory exists
if [[ ! -d "$base_dir" ]]; then
    echo "Error: Directory '$base_dir' not found"
    exit 1
fi

if [[ ! -f "$human_model" ]]; then
    echo "Error: File '$human_model' not found"
    exit 1
fi

# Ensure trailing slash
base_dir="${base_dir%/}/"

echo "Processing directory: $base_dir"

# Process all subdirectories
find "$base_dir" -mindepth 1 -type d | while read -r dir; do
    relative_path=${dir#"$base_dir/"}
    last_folder=$(echo "$relative_path" | awk -F'/' '{print $NF}')
    echo "-------------$last_folder---------------"
    python transfer_human_model.py -d "$relative_path" -m "$human_model" "${python_args[@]}"
done
