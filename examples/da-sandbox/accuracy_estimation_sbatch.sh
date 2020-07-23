#!/bin/bash

#SBATCH --time=12:00:00                 # walltime
#SBATCH --nodes=1                       # number of nodes (per job)
#SBATCH --mem=386G                      # memory per node
#SBATCH --exclusive                     # exclusive use of the node
#SBATCH --ntasks=1                      # number of processes (i.e. tasks)
#SBATCH --cpus-per-task=56              # number of cores per process
#SBATCH -J "risk_networks_accuracy"
#SBATCH --output=slurm_output/%A_%a.out
#SBATCH --error=slurm_output/%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-4

##SBATCH --mem-per-cpu=24G               # 24G is needed for 10k full user base
##SBATCH --constraint=cascadelake        # 'cascadelake' for expansion nodes


# create a directory named "slurm_output" and then submit with:
#       sbatch --mail-user=mail@domain.com accuracy_estimation_sbatch.sh

# preface ######################################################################
set -euo pipefail

OUTPUT_DIR="output_${SLURM_ARRAY_JOB_ID}" # corresponds to "output_%A"


# parameters & constants #######################################################
user_fractions=(0.03 0.05 0.1 0.5 1.0)
fractions_tested=(0.5 0.2 0.1 0.02 0.01)

network_size=1e4
I_min_threshold=0.0
user_fraction=${user_fractions[${SLURM_ARRAY_TASK_ID}]}
tested=${fractions_tested[${SLURM_ARRAY_TASK_ID}]}
output_path="${OUTPUT_DIR}/${user_fraction}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"


# launch #######################################################################
module load python3/3.8.5
srun python3 accuracy_estimation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-fraction-tested=${tested} \
  --observations-I-min-threshold=${I_min_threshold} \
  --network-node-count=${network_size} \
  >${stdout} 2>${stderr}


