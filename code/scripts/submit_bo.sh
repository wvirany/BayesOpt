#!/bin/bash

#SBATCH --job-name=bo
#SBATCH --array=0-9
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --output=logs/dockstring-bo/%A/%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=waltervirany@gmail.com

module purge
module load python
module load anaconda
conda activate tanimoto-gp

python3 dockstring-bo.py \
    --target PARP1 \
    --n_init 100 \
    --budget 1000 \
    --radius 2