#!/bin/bash

#SBATCH --time=120:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -J "1e3test"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-0

################################
# Intervention test experiment #
################################
# submit with: sbatch run_intervention_scenario_parallel.sh

#module load python3/3.8.5
# preface ######################################################################
set -euo pipefail

# parallelization
num_cpus=${SLURM_CPUS_PER_TASK}
bytes_of_memory=$((${SLURM_MEM_PER_NODE}*1000000 / 8)) #MB -> bytes
echo "requested ${num_cpus} cores and ray is told ${bytes_of_memory} memory available"
# parameters & constants #######################################################

# network 
network_size=1e3

# user base
user_fraction=1.0


#sensors
wearers=0
batches_sensors=1
batches_records=1 #195884 nodes

#da params 
da_window=5.0
n_sweeps=1

#observation_noise
obs_noise=1e-8
#reglarization parameters
test_reg=1e-1
record_reg=0.0

# observation localization (number of nbhds observations have effect on (integer) 0 = delta at observation, 1= nbhd of observation)
distance_threshold=0

#inflation (No inflation is 1.0)
test_inflation=1.0
record_inflation=1.0

# intervention
int_freq='single'
intervention_start_time=18

#name
EXP_NAME="WSRIO_${da_window}_${n_sweeps}_${test_reg}_${test_inflation}_${obs_noise}"
### Experimental series parameters ###############################################
#1% 5% 25% of 97942
test_budgets=(49)
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}
batches_tests=(1) #so no batch > 1000 nodes
batches_test=${batches_tests[${SLURM_ARRAY_TASK_ID}]}

# output parameters
OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"

echo "output to be found in: ${output_path}, stdout in $stdout, stderr in $stderr "

# launch #######################################################################
python3 joint_iterated_forward_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-noise=${obs_noise} \
  --observations-I-budget=${budget} \
  --observations-sensor-wearers=${wearers} \
  --network-node-count=${network_size} \
  --assimilation-batches-sensor=${batches_sensors} \
  --assimilation-batches-test=${batches_tests} \
  --assimilation-batches-record=${batches_records} \
  --parallel-flag \
  --parallel-memory=${bytes_of_memory} \
  --parallel-num-cpus=${num_cpus} \
  --intervention-frequency=${int_freq} \
  --intervention-start-time=${intervention_start_time} \
  --test-assimilation-joint-regularization=${test_reg} \
  --record-assimilation-joint-regularization=${record_reg} \
  --assimilation-test-inflation=${test_inflation} \
  --assimilation-record-inflation=${record_inflation} \
  --distance-threshold=${distance_threshold} \
  --assimilation-window=${da_window} \
  --assimilation-sweeps=${n_sweeps} \
  >${stdout} 2>${stderr}


