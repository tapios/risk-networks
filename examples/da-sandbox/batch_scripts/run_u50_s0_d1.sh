#!/bin/bash

#SBATCH --time=120:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH -J "u50s0d1"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-6

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
user_fraction=0.5
user_base_type="neighbor"

# testing: virus tests
I_min_threshold=0.0
I_max_threshold=1.0

#da params 
da_window=1.0
n_sweeps=1

#observation_noise
obs_noise=1e-12
#reglarization parameters
sensor_reg=1e-1
test_reg=5e-2
record_reg=1e-2

# observation localization (number of nbhds observations have effect on (integer) 0 = delta at observation, 1= nbhd of observation)
distance_threshold=0

#inflation (No inflation is 1.0)
sensor_inflation=10.0
test_inflation=3.0
record_inflation=2.0
rate_inflation=1.0
additive_inflation=0.1

#param noise mean - ln((mu+bias)^2 / sqrt((mu+bias)^2 +sig^2))
param_prior_noise_factor=0.25

# network  + sensor wearers
EXP_NAME="u50_s0_d1" #1e5 = 97942 nodes
#EXP_NAME="noda_1e5_parsd0.25_nosd"
# Experimental series parameters ###############################################
#5% 10% 25%, of 48971
test_budgets=(0 489 1224 2448 4897 12242 48971)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}

# output parameters
OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"

echo "output to be found in: ${output_path}, stdout in $stdout, stderr in $stderr "

cp batch_scripts/run_u50_s0_d1.sh ${output_path}

# launch #######################################################################
# launch #######################################################################
python3 joint_iterated_forward_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --user-network-type=${user_base_type} \
  --user-network-weighted \
  --constants-output-path=${output_path} \
  --observations-noise=${obs_noise} \
  --observations-I-budget=${budget} \
  --observations-sensor-wearers=${wearers} \
  --network-node-count=${network_size} \
  --parallel-flag \
  --parallel-memory=${bytes_of_memory} \
  --parallel-num-cpus=${num_cpus} \
  --sensor-assimilation-joint-regularization=${sensor_reg} \
  --test-assimilation-joint-regularization=${test_reg} \
  --record-assimilation-joint-regularization=${record_reg} \
  --assimilation-sensor-inflation=${sensor_inflation} \
  --assimilation-test-inflation=${test_inflation} \
  --assimilation-record-inflation=${record_inflation} \
  --distance-threshold=${distance_threshold} \
  --assimilation-window=${da_window} \
  --assimilation-sweeps=${n_sweeps} \
  --params-learn-transition-rates \
  --params-learn-transmission-rate \
  --params-transmission-rate-noise=${param_prior_noise_factor} \
  --params-transmission-inflation=${rate_inflation} \
  --assimilation-additive-inflation\
  --assimilation-additive-inflation-factor=${additive_inflation}\
  --record-ignore-mass-constraint\
  >${stdout} 2>${stderr}



