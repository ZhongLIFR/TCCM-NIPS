#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="" # Disable GPU globally for fairness
PATH_TO_DIR="./datasets/contamination"
DATASETS=($(find "$PATH_TO_DIR" -type f -name "*.npz"))

MAX_CORES=$(nproc --all)  # Get total CPU cores
RANDOM_SEEDs=(0 1 2 3 4)
TIME_LIMIT=7200 # In seconds, 7200 seconds are 2 hours
MEMORY_LIMIT=10 # In gigabytes
k=1  # Number of datasets to process before sleeping
CONTAMINATE_LV=10 # The level of contamination
# SLEEP_TIME=3600  # Sleep for 1 hours (in seconds)

mkdir -p logs_contam
echo "Start contaminating datasets..." >> ./logs_contam/Contaminate_log.log
for RANDOM_SEED in "${RANDOM_SEEDs[@]}"; do
    mkdir -p logs_contam/seed_${RANDOM_SEED}  # Ensure log directory exists
    # Batch processing
    total_datasets=${#DATASETS[@]}
    for (( start_idx=0; start_idx<total_datasets; start_idx+=k )); do
        for (( i=start_idx; i<start_idx+k && i<total_datasets; i++ )); do
            dname="${DATASETS[$i]}"
            dname=$(basename "$dname" .npz)

            for j in {0..10}; do # Top 11 models, 10 baselines + 1 ours

                for ((ci=0; ci<CONTAMINATE_LV; ci+=1)); do
                    core_id=$(( (i * 11 * CONTAMINATE_LV + j * CONTAMINATE_LV + ci) % MAX_CORES ))
                    # Run the process
                    echo "$core_id $dname $RANDOM_SEED" >> ./logs_contam/Contaminate_log.log
                    nohup taskset -c "$core_id" python ContaminationStudies.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" -c "$ci">> "./logs_contam/seed_${RANDOM_SEED}/run_${dname}_model_${j}_contam_${ci}.log" 2>&1 &
                done
            done
            # Wait for all background processes to complete before sleeping
            # echo "Sleeping for 3 hours..." >> ./logs_contam/Contaminate_log.log
            # sleep $SLEEP_TIME # For High-dimensional datasets, we send the task every 3 hours
            wait
            break
        done
        wait # For small datasets we wait until they finish
        break
    done
    wait
done
echo "Contaminating datasets processed." >> ./logs_contam/Contaminate_log.log