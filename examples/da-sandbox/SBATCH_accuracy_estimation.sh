#!/bin/bash

#SBATCH --time=12:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=24G               # 24G is needed for 10k full user base
#SBATCH -J "risk_networks_accuracy"
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-4


# submit with: sbatch --mail-user=mail@domain.com SBATCH_accuracy_estimation.sh

# preface ######################################################################
set -euo pipefail

OUTPUT_DIR_PREFIX="output"
COUNTER=""

while [ -d "${OUTPUT_DIR_PREFIX}${COUNTER}" ]; do
  ((COUNTER+=1))
done

OUTPUT_DIR="${OUTPUT_DIR_PREFIX}${COUNTER}"


# parameters & constants #######################################################
user_fractions=(0.03 0.05 0.1 0.5 1.0)
fractions_tested=(0.5 0.2 0.1 0.02 0.01)

network_size=1e4
user_fraction=${user_fractions[${SLURM_ARRAY_TASK_ID}]}
tested=${fractions_tested[${SLURM_ARRAY_TASK_ID}]}
output_path="${OUTPUT_DIR}/${user_fraction}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"


# launch #######################################################################
module load python3/3.7.0
python3 accuracy_estimation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-fraction-tested=${tested} \
  --network-node-count=${network_size} \
  >${stdout} 2>${stderr}


