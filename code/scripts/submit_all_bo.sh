#!/bin/bash

# Run from "code/" directory with ./submit_all_bo.sh

TARGETS=("PARP1" "F2" "ESR2" "PGR")
N_INITS=(100 1000)
RADII=(2 4)
SPARSE_OPTIONS=("true" "false")
POOL=1000000
BUDGET=1000
NUM_JOBS=10

for TARGET in "${TARGETS[@]}"; do
    for N_INIT in "${N_INITS[@]}"; do
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

                # Create job name w/ all params
                JOB_NAME="bo-${TARGET}-p${POOL}-n${N_INIT}-r${RADIUS}-${SPARSE_NAME}"
                LOG_DIR="logs/dockstring-bo/${JOB_NAME}"

                # Create log directory if it doesn't exist
                mkdir -p "${LOG_DIR}"

                echo "Submitting job array: ${JOB_NAME}"

                sbatch --job-name=${JOB_NAME} \
                    --array=0-9 \
                    --partition=amilan \
                    --qos=normal \
                    --time=8:00:00 \
                    --nodes=1 \
                    --mem=64G \
                    --output=logs/dockstring-bo/${JOB_NAME}/%a.out \
                    --mail-type=ALL \
                    --mail-user=waltervirany@gmail.com \
                    --wrap="module purge && module load python && module load anaconda && \
                            conda activate tanimoto-gp && \
                            python3 dockstring-bo.py \
                            --target ${TARGET} \
                            --pool ${POOL} \
                            --n_init ${N_INIT} \
                            --budget ${BUDGET} \
                            --radius ${RADIUS} \
                            ${SPARSE_FLAG}"
                
                sleep 1
            done
        done
    done
done

echo "All jobs submitted"