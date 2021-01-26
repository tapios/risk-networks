#!/bin/bash

#SBATCH --time=120:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -J "u100s75d5"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-3

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
wearers=73456

# user base
user_fraction=1.0

# testing: virus tests
I_min_threshold=0.0
I_max_threshold=1.0

# intervention
int_freq='single'
intervention_start_time=18

#da params 
da_window=5.0
n_sweeps=1

#observation_noise
obs_noise=1e-8
#reglarization parameters
sensor_reg=1e-1
test_reg=1e-2
record_reg=1e-4

# observation localization (number of nbhds observations have effect on (integer) 0 = delta at observation, 1= nbhd of observation)
distance_threshold=0

#inflation (No inflation is 1.0)
sensor_inflation=10.0
test_inflation=3.0
record_inflation=2.0
rate_inflation=1e5

#param noise mean - ln((mu+bias)^2 / sqrt((mu+bias)^2 +sig^2))
param_prior_noise=0.5
param_prior_bias=-9.2

# network  + sensor wearers
#EXP_NAME="1e5_params_WRI_${da_window}_${test_reg}_${test_inflation}" #1e5 = 97942 nodes
EXP_NAME="u100_s75_d5" #1e5 = 97942 nodes

# Experimental series parameters ###############################################
#5% 10% 25%, of 97942
test_budgets=(0 4897 9794 24485)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}

# output parameters
OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"

echo "output to be found in: ${output_path}, stdout in $stdout, stderr in $stderr "

cp run_u100_s75_d5.sh ${output_path}

# launch #######################################################################
# launch #######################################################################
python3 joint_iterated_forward_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-noise=${obs_noise} \
  --observations-I-budget=${budget} \
  --observations-sensor-wearers=${wearers} \
  --network-node-count=${network_size} \
  --parallel-flag \
  --parallel-memory=${bytes_of_memory} \
  --parallel-num-cpus=${num_cpus} \
  --intervention-frequency=${int_freq} \
  --intervention-start-time=${intervention_start_time} \
  --sensor-assimilation-joint-regularization=${sensor_reg} \
  --test-assimilation-joint-regularization=${test_reg} \
  --record-assimilation-joint-regularization=${record_reg} \
  --assimilation-sensor-inflation=${sensor_inflation} \
  --assimilation-test-inflation=${test_inflation} \
  --assimilation-record-inflation=${record_inflation} \
  --distance-threshold=${distance_threshold} \
  --assimilation-window=${da_window} \
  --assimilation-sweeps=${n_sweeps} \
  --params-learn-transmission-rate \
  --params-transmission-rate-noise=${param_prior_noise} \
  --params-transmission-rate-bias=${param_prior_bias} \
  --params-transmission-inflation=${rate_inflation} \
  >${stdout} 2>${stderr}



