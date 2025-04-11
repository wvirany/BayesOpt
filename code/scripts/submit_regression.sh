#!/bin/bash

#SBATCH --job-name=dockstring-regression
#SBATCH --array=0-9
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --output=logs/dockstring-regression/%A/%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=waltervirany@gmail.com

module purge
module load python
module load anaconda
conda activate tanimoto-gp

python3 dockstring-regression.py \
    --target PARP1 \
    --n_train 10000 \
    --radius 2 \
    --sparse \
    --count