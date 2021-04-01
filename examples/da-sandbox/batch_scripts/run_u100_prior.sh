#!/bin/bash

#SBATCH --time=120:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -J "prior_run"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-0

################################
# Intervention test experiment #
################################
# submit with: sbatch run_intervention_scenario_parallel.sh

# preface ######################################################################
set -euo pipefail

# parallelization
num_cpus=${SLURM_CPUS_PER_TASK}
bytes_of_memory=$((${SLURM_MEM_PER_NODE}*1000000 / 4)) #MB -> bytes
echo "requested ${num_cpus} cores and ray is told ${bytes_of_memory} memory available"
# parameters & constants #######################################################

network_size=1e5

# user base
user_fraction=1.0

#param noise mean - ln((mu+bias)^2 / sqrt((mu+bias)^2 +sig^2))
param_prior_noise_factor=0.25

# network  + sensor wearers
EXP_NAME="noda_u100_prior"
# Experimental series parameters ###############################################
#5% 10% 25%, of 97942
budget=0

# output parameters
OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"

echo "output to be found in: ${output_path}, stdout in $stdout, stderr in $stderr "

cp batch_scripts/run_u100_prior.sh ${output_path}

# launch #######################################################################
python3 joint_iterated_forward_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --network-node-count=${network_size} \
  --parallel-flag \
  --parallel-memory=${bytes_of_memory} \
  --parallel-num-cpus=${num_cpus} \
  --params-learn-transition-rates \
  --params-learn-transmission-rate \
  --params-transmission-rate-noise=${param_prior_noise_factor} \
  --prior-run\
  >${stdout} 2>${stderr}



