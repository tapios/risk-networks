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

# testing: virus tests
I_min_threshold=0.0
I_max_threshold=1.0

#localization - [from 0to 1 , of variance preservation e.g 90%]
test_joint_regularization=0.5
record_joint_regularization=0.999
# localization - Stick to 0
test_obs_regularization=0
record_obs_regularization=0

#inflation (No inflation is 1.0)
test_inflation=1.1
record_inflation=2.0

# intervention
int_freq='single'
intervention_start_time=18

#name
n_sweeps_total=3 # should be 2 + n for n sweeps of the H/D assimilator (unlinked to file)
EXP_NAME="truncsvd_T${test_joint_regularization}R${record_joint_regularization}_inflateT${test_inflation}R${record_inflation}"
#EXP_NAME="NYC_1e3_inflate${inflation}_5day_1-${n_sweeps_total}sweep_J${test_joint_regularization}-${record_joint_regularization}_O${test_obs_regularization}-${record_obs_regularization}" 

# Experimental series parameters ###############################################
#1% 5% 25% of 97942
test_budgets=(10)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}
batches_tests=(1) #so no batch > 1000 nodes
batches_test=${batches_tests[${SLURM_ARRAY_TASK_ID}]}

# output parameters
OUTPUT_DIR="output"
output_path="${OUTPUT_DIR}/${EXP_NAME}_${budget}"
stdout="${output_path}/stdout"
stderr="${output_path}/stderr"

mkdir -p "${output_path}"

echo "output to be found in: ${output_path}"

# launch #######################################################################
python3 joint_iterated_forward_assimilation.py \
  --user-network-user-fraction=${user_fraction} \
  --constants-output-path=${output_path} \
  --observations-I-budget=${budget} \
  --observations-I-min-threshold=${I_min_threshold} \
  --observations-I-max-threshold=${I_max_threshold} \
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
  --test-assimilation-joint-regularization=${test_joint_regularization} \
  --record-assimilation-joint-regularization=${record_joint_regularization} \
  --test-assimilation-obs-regularization=${test_obs_regularization} \
  --record-assimilation-obs-regularization=${record_obs_regularization} \
  --assimilation-test-inflation=${test_inflation}\
  --assimilation-record-inflation=${record_inflation}\
  >${stdout} 2>${stderr}


