#!/bin/bash

# Run from "code/" directory with ./submit_all_regression.sh

TARGETS=("PARP1")
N_TRAINS=(10000)
RADII=(2 4)
FP_SIZES=(2048 1024 512 256)


for TARGET in "${TARGETS[@]}"; do
    for N_TRAIN in "${N_TRAINS[@]}"; do
        for RADIUS in "${RADII[@]}"; do

            # Submit job for sparse fingerprint
            JOB_NAME="regression-${TARGET}-n${N_TRAIN}-sparse-r${RADIUS}"
            LOG_DIR="logs/dockstring-regression/${JOB_NAME}"

            # Create log directory if it doesn't exist
            mkdir -p "${LOG_DIR}"

            echo "Submitting job array: ${JOB_NAME}"

            sbatch --job-name=${JOB_NAME} \
                --array=0-29 \
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
                        --sparse"

            sleep 1

            
            # Submit jobs for compressed fingerprints (varying fp_size)
            for FP_SIZE in "${FP_SIZES[@]}"; do

                JOB_NAME="regression-${TARGET}-n${N_TRAIN}-compressed-r${RADIUS}-${FP_SIZE}"
                LOG_DIR="logs/dockstring-regression/${JOB_NAME}"

                # Create log directory if it doesn't exist
                mkdir -p "${LOG_DIR}"

                echo "Submitting job array: ${JOB_NAME}"

                sbatch --job-name=${JOB_NAME} \
                    --array=0-29 \
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
                            --fp_size ${FP_SIZE}"
                    
                sleep 1
            
            done
        done
    done
done

echo "All jobs submitted"