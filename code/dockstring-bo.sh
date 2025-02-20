#!/bin/bash

# Define the script name
PYTHON_SCRIPT="dockstring-bo.py"  # Replace with your actual script name

# Default parameters that stay constant
N_INIT=10
BUDGET=1

# Loop through all combinations
for radius in 2 4; do
    for sparse in "true" "false"; do
        for target in "PARP1" "F2"; do
            echo "Running with: radius=$radius, sparse=$sparse, target=$target"
            
            # Convert 'true'/'false' string to --sparse flag
            sparse_flag=""
            if [ "$sparse" = "true" ]; then
                sparse_flag="--sparse"
            fi
            
            python $PYTHON_SCRIPT \
                --radius $radius \
                $sparse_flag \
                --target $target \
                --n_init $N_INIT \
                --budget $BUDGET
            
            echo "Completed run"
            echo "------------------------"
        done
    done
done