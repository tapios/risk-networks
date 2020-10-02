#!/bin/bash

#SBATCH --time=120:00:00                 # walltime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=196G
#SBATCH -J "50u+s"
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err  
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-2

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
EXP_NAME="NYC_1e5_user50_sensors" #1e5 = 97942 nodes
network_size=1e5
# user base
user_fraction=0.5 #48971 users

#sensors
wearers=48971
batches_sensors=98

#records
batches_records=196 #195884/2 nodes
# testing: virus tests
I_min_threshold=0.0
I_max_threshold=1.0

# intervention
int_freq='single'
intervention_start_time=18

# other
update_test="local"

# Experimental series parameters ###############################################
#1% 5% 25% of 48971
test_budgets=(490 2449 12243)  
budget=${test_budgets[${SLURM_ARRAY_TASK_ID}]}
batches_tests=(1 5 25) #so no batch > 1000 nodes
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
  --assimilation-update-test=${update_test} \
  >${stdout} 2>${stderr}


