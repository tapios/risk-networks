#!/bin/bash

#SBATCH --time=12:00:00                 # walltime
#SBATCH --ntasks=6                      # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=1G
#SBATCH -J "risk_networks_accuracy"
#SBATCH --output="output/slurm.out"
#SBATCH --mail-user=dburov@caltech.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Submit this script with: sbatch run_accuracy_estimation.sh

# main
set -euo pipefail

module load python3/3.7.0

python3 accuracy_estimation_script.py


