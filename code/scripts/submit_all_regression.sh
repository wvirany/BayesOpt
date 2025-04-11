#!/bin/bash

# Run from "code/" directory with ./submit_all_regression.sh

TARGETS=("PARP1" "F2" "ESR2" "PGR")
N_TRAINS=(100 1000 10000)
RADII=(2 4)
SPARSE_OPTIONS=("true" "false")
COUNT="true"

for TARGET in "${TARGETS[@]}"; do
    for N_TRAIN in "${N_TRAINS[@]}"; do
        for RADIUS in "${RADII[@]}"; do
            for SPARSE in "${SPARSE_OPTIONS[@]}"; do

            # Derived parameters
            if [ "$SPARSE" = "true" ]; then
                SPARSE_NAME="sparse"
                SPARSE_FLAG="--sparse"
            else
                SPARSE_NAME="compressed"
                SPARSE_FLAG=""
            fi

            if [ "$COUNT" = "true" ]; then
                COUNT_FLAG="--count"
                COUNT_LABEL="count"
            else
                COUNT_FLAG=""
                COUNT_LABEL="binary"
            fi

            # Create job name w/ all params
            JOB_NAME="regression-${TARGET}-n${N_TRAIN}-r${RADIUS}-${SPARSE_NAME}-${COUNT_LABEL}"
            LOG_DIR="logs/dockstring-regression/${JOB_NAME}"

            mkdir -p "${LOG_DIR}"

            echo "Submitting job array: ${JOB_NAME}"

            sbatch --job-name=${JOB_NAME} \
                    --array=0-9 \
                    --partition=amilan \
                    --qos=normal \
                    --time=8:00:00 \
                    --nodes=1 \
                    --mem=128G \
                    --output=logs/dockstring-regression/${JOB_NAME}/%a.out \
                    --mail-type=ALL \
                    --mail-user=waltervirany@gmail.com \
                    --wrap="module purge && module load python && module load anaconda && \
                            conda activate tanimoto-gp && \
                            python3 dockstring-regression.py \
                            --target ${TARGET} \
                            --n_train ${N_TRAIN} \
                            --radius ${RADIUS} \
                            ${SPARSE_FLAG} \
                            ${COUNT_FLAG}"
                
                sleep 1
            done
        done
    done
done

echo "All jobs submitted"