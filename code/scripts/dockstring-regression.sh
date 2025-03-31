#!/bin/bash

#SBATCH --job-name=dockstring-regression
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

# Run the Python script with appropriate parameters
python dockstring-regression.py --target PARP1 \
                                --radius 2