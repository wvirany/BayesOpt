#!/bin/bash

# Run from "code/" directory with ./submit_all_bo.sh

TARGETS=("PARP1")
N_INITS=(100 1000)
RADII=(2)
POOL=1000000
BUDGET=1000
FP_SIZES=(2048 1024 512 256)

for TARGET in "${TARGETS[@]}"; do
    for N_INIT in "${N_INITS[@]}"; do
        for RADIUS in "${RADII[@]}"; do

            # # Submit job for sparse fingerprint
            # JOB_NAME="bo-${TARGET}-p${POOL}-${N_INIT}-${BUDGET}-sparse-r${RADIUS}"
            # LOG_DIR="logs/dockstring-bo/${JOB_NAME}"

            # # Create log directory if it doesn't exist
            # mkdir -p "${LOG_DIR}"

            # echo "Submitting job array: ${JOB_NAME}"

            # sbatch --job-name=${JOB_NAME} \
            #     --array=0-29 \
            #     --partition=amilan \
            #     --qos=normal \
            #     --time=8:00:00 \
            #     --nodes=1 \
            #     --mem=64G \
            #     --output=logs/dockstring-bo/${JOB_NAME}/%a.out \
            #     --mail-type=ALL \
            #     --mail-user=waltervirany@gmail.com \
            #     --wrap="module purge && module load python && module load anaconda && \
            #             conda activate tanimoto-gp && \
            #             python3 dockstring-bo.py \
            #             --target ${TARGET} \
            #             --pool ${POOL} \
            #             --n_init ${N_INIT} \
            #             --budget ${BUDGET} \
            #             --radius ${RADIUS} \
            #             --sparse"

            # sleep 1


            # # Submit jobs for compressed fingerprints (varying fp_size)
            # for FP_SIZE in "${FP_SIZES[@]}"; do

            #     JOB_NAME="bo-${TARGET}-p${POOL}-${N_INIT}-${BUDGET}-compressed-r${RADIUS}-s${FP_SIZE}"
            #     LOG_DIR="logs/dockstring-bo/${JOB_NAME}"

            #     # Create log directory if it doesn't exist
            #     mkdir -p "${LOG_DIR}"

            #     echo "Submitting job array: ${JOB_NAME}"

            #     sbatch --job-name=${JOB_NAME} \
            #         --array=0-29 \
            #         --partition=amilan \
            #         --qos=normal \
            #         --time=8:00:00 \
            #         --nodes=1 \
            #         --mem=64G \
            #         --output=logs/dockstring-bo/${JOB_NAME}/%a.out \
            #         --mail-type=ALL \
            #         --mail-user=waltervirany@gmail.com \
            #         --wrap="module purge && module load python && module load anaconda && \
            #                 conda activate tanimoto-gp && \
            #                 python3 dockstring-bo.py \
            #                 --target ${TARGET} \
            #                 --pool ${POOL} \
            #                 --n_init ${N_INIT} \
            #                 --budget ${BUDGET} \
            #                 --radius ${RADIUS} \
            #                 --fp_size ${FP_SIZE}"
                    
            #     sleep 1
            # done

            # Submit random baseline
            JOB_NAME="bo-${TARGET}-p${POOL}-${N_INIT}-${BUDGET}-random"
            LOG_DIR="logs/dockstring-bo/${JOB_NAME}"

            # Create log directory if it doesn't exist
            mkdir -p "${LOG_DIR}"

            echo "Submitting job array: ${JOB_NAME}"

            sbatch --job-name=${JOB_NAME} \
                --array=0-29 \
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
                        --random"

            sleep 1


        done
    done
done


echo "All jobs submitted"