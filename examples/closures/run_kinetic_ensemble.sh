#!/bin/bash

#SBATCH --time=48:00:00                 # walltime
#SBATCH --ntasks=1                      # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=24G
#SBATCH -J "kin_ens"
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err

#SBATCH --array=0-10

##################################
# Infectiousness test experiment #
##################################
# set: fractions_test = % tested of I per day
# set: output_path = ${OUTPUT_DIR}/expt_name_${tested}
# ENSURE you have the non-parallel import from _master_eqns_init
# submit with: sbatch run_infectious_test_expt.sh

# preface ######################################################################
set -euo pipefail

OUTPUT_DIR="output"
EXP_NAME="kinetic_ensemble"
array_num=$(printf "%04d" ${SLURM_ARRAY_TASK_ID})
output_path=${OUTPUT_DIR}/${EXP_NAME}


# parameters & constants #######################################################

#parsed parameters
stdout="${output_path}/stdout_${SLURM_ARRAY_TASK_ID}"
stderr="${output_path}/stderr_${SLURM_ARRAY_TASK_ID}"
network_size=1e3
epidemic_storage="epidemic_storage_${array_num}.pkl"

mkdir -p "${output_path}"

# launch #######################################################################
# serial unless --parallel-flag is used
python3 run_kinetic_model.py \
  --constants-output-path=${output_path} \
  --kinetic-seed=${SLURM_ARRAY_TASK_ID} \
  --epidemic-storage-name=${epidemic_storage} \
  --network-node-count=${network_size} \
  >${stdout} 2>${stderr}
