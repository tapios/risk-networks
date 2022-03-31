#!/bin/bash

#SBATCH --time=120:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -J "test_and_isolate"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=4-6

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
wearers=0

# user base
user_fraction=1.0

# testing: virus tests
I_min_threshold=0.0
I_max_threshold=1.0

# intervention
intervention_freq='interval'
intervention_type='nothing'
intervention_nodes='test_data_only'
intervention_interval=1.0
intervention_start_time=8

#inflation (No inflation is 1.0)

#param noise mean - ln((mu+bias)^2 / sqrt((mu+bias)^2 +sig^2))
param_prior_noise_factor=0.25

# network  + sensor wearers
#EXP_NAME="1e5_params_WRI_${da_window}_${test_reg}_${test_inflation}" #1e5 = 97942 nodes
EXP_NAME="test_and_no_isolate" #1e5 = 97942 nodes
#EXP_NAME="noda_1e5_parsd0.25_nosd"
# Experimental series parameters ###############################################
#5% 10% 25%, of 97942
test_budgets=(0 979 2448 4897 9794 24485 97942)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}

# output parameters
OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"

echo "output to be found in: ${output_path}, stdout in $stdout, stderr in $stderr "

cp batch_scripts/run_test_and_no_isolate.sh ${output_path}

# launch #######################################################################
# launch #######################################################################
python3 joint_iterated_forward_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-budget=${budget} \
  --observations-sensor-wearers=${wearers} \
  --network-node-count=${network_size} \
  --parallel-flag \
  --parallel-memory=${bytes_of_memory} \
  --parallel-num-cpus=${num_cpus} \
  --intervention-frequency=${intervention_freq} \
  --intervention-nodes=${intervention_nodes}\
  --intervention-type=${intervention_type}\
  --intervention-nodes=${intervention_nodes}\
  --intervention-start-time=${intervention_start_time} \
  --params-learn-transition-rates \
  --params-learn-transmission-rate \
  --params-transmission-rate-noise=${param_prior_noise_factor} \
  >${stdout} 2>${stderr}



