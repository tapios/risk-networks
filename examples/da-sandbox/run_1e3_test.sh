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

# preface ######################################################################
set -euo pipefail

# parallelization
num_cpus=${SLURM_CPUS_PER_TASK}
bytes_of_memory=$((${SLURM_MEM_PER_NODE}*1000000 / 8)) #MB -> bytes
echo "requested ${num_cpus} cores and ray is told ${bytes_of_memory} memory available"
# parameters & constants #######################################################

# network 
EXP_NAME="NYC_1e3_local_1sweep" 
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

# intervention
int_freq='single'
intervention_start_time=18

#update types
update_sensor_type="local"
update_test_type="local"
update_record_type="local"

# Experimental series parameters ###############################################
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


# launch #######################################################################
python3 joint_epidemic_assimilation.py \
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
  --assimilation-update-sensor=${update_sensor_type}\
  --assimilation-update-test=${update_test_type}\
  --assimilation-update-record=${update_record_type}\
  >${stdout} 2>${stderr}


