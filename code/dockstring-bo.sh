#!/bin/bash

#SBATCH --partition=amilan
#SBATCH --job-name=dockstring-bo-100-F2-r4-compressed
#SBATCH --output=logs/100-F2-r4-compressed.out
#SBATCH --time=4:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=waltervirany@gmail.com

module purge
module load python
module load anaconda
conda activate tanimoto-gp

python3 dockstring-bo.py --n_init 100 --budget 100 --target "F2" --radius 4