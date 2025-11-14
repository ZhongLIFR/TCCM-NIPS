#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="" # Disable GPU globally for fairness
PATH_TO_DIR="./datasets/small" 
DATASETS=($(find "$PATH_TO_DIR" -type f -name "*.npz"))

MAX_CORES=$(nproc --all)  # Get total CPU cores
RANDOM_SEEDs=(0 1 2 3 4)
TIME_LIMIT=259200 # In seconds, 259200 seconds are three days
MEMORY_LIMIT=10 # In gigabytes
k=16  # Number of datasets to process before sleeping

mkdir -p logs
echo "Start SMALL datasets..." >> ./logs/All_log.log
for RANDOM_SEED in "${RANDOM_SEEDs[@]}"; do
    mkdir -p logs/seed_${RANDOM_SEED}  # Ensure log directory exists
    # Batch processing
    total_datasets=${#DATASETS[@]}
    for (( start_idx=0; start_idx<total_datasets; start_idx+=k )); do
        for (( i=start_idx; i<start_idx+k && i<total_datasets; i++ )); do
            dname="${DATASETS[$i]}"
            dname=$(basename "$dname" .npz)

            for j in {45..50}; do # 7 models
                core_id=$(( (i * 6 + j) % MAX_CORES ))

                # Run the process
                echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
                nohup taskset -c "$core_id" python FullExperiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
            done
        done
        wait # For small datasets we wait until they finish
    done
    wait
done
echo "SMALL datasets processed." >> ./logs/All_log.log

echo "Start MEDIUM datasets..." >> ./logs/All_log.log
PATH_TO_DIR="./datasets/medium" # Put the absolute path to the datasets directory here
DATASETS=($(find "$PATH_TO_DIR" -type f -name "*.npz"))

MAX_CORES=$(nproc --all)  # Get total CPU cores
RANDOM_SEEDs=(0 1 2 3 4)
TIME_LIMIT=259200 # In seconds, 259200 seconds are three days
MEMORY_LIMIT=10 # In gigabytes
k=16 # Number of datasets to process before sleeping
SLEEP_TIME=3600  # Sleep for 1 hours (in seconds)

# mkdir -p logs
for RANDOM_SEED in "${RANDOM_SEEDs[@]}"; do
    # mkdir -p logs/seed_${RANDOM_SEED}  # Ensure log directory exists
    # Batch processing
    total_datasets=${#DATASETS[@]}
    for (( start_idx=0; start_idx<total_datasets; start_idx+=k )); do
        for (( i=start_idx; i<start_idx+k && i<total_datasets; i++ )); do
            dname="${DATASETS[$i]}"
            dname=$(basename "$dname" .npz)

            for j in {45..50}; do # 7 models
                core_id=$(( (i * 6 + j) % MAX_CORES ))

                # Run the process
                echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
                nohup taskset -c "$core_id" python FullExperiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
            done
        done
        # echo "Sleeping for 1 hour..." >> ./logs/All_log.log
        # sleep $SLEEP_TIME
        wait
    done
    wait
done
echo "MEDIUM datasets processed." >> ./logs/All_log.log
wait

echo "Start LARGE datasets..." >> ./logs/All_log.log
PATH_TO_DIR="./datasets/large" # Put the absolute path to the datasets directory here
DATASETS=($(find "$PATH_TO_DIR" -type f -name "*.npz"))

MAX_CORES=$(nproc --all)  # Get total CPU cores
RANDOM_SEEDs=(0 1 2 3 4)
TIME_LIMIT=259200 # In seconds, 259200 seconds are three days
MEMORY_LIMIT=10 # In gigabytes
k=12 # Number of datasets to process before sleeping
SLEEP_TIME=10800  # Sleep for 3 hours (in seconds)

# mkdir -p logs
for RANDOM_SEED in "${RANDOM_SEEDs[@]}"; do
    # mkdir -p logs/seed_${RANDOM_SEED}  # Ensure log directory exists
    # Batch processing
    total_datasets=${#DATASETS[@]}
    for (( start_idx=0; start_idx<total_datasets; start_idx+=k )); do
        # echo "Processing batch starting at index $start_idx..."

        for (( i=start_idx; i<start_idx+k && i<total_datasets; i++ )); do
            dname="${DATASETS[$i]}"
            dname=$(basename "$dname" .npz)

            for j in {45..50}; do # 7 models
                core_id=$(( (i * 6 + j) % MAX_CORES ))

                # Run the process
                echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
                nohup taskset -c "$core_id" python FullExperiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
            done
        done
        wait
        # # Wait for all background processes to complete before sleeping
        # echo "Sleeping for 3 hours..." >> ./logs/All_log.log
        # sleep $SLEEP_TIME # For High-dimensional datasets, we send the task every 3 hours
    done
    wait
done
echo "LARGE datasets processed." >> ./logs/All_log.log

echo "Start high-dimensional datasets..." >> ./logs/All_log.log
PATH_TO_DIR="./datasets/high_dim" # Put the absolute path to the datasets directory here
DATASETS=($(find "$PATH_TO_DIR" -type f -name "*.npz"))

MAX_CORES=$(nproc --all)  # Get total CPU cores
RANDOM_SEEDs=(0 1 2 3 4)
TIME_LIMIT=259200 # In seconds, 259200 seconds are three days
MEMORY_LIMIT=10 # In gigabytes
k=12 # Number of datasets to process before sleeping
SLEEP_TIME=10800  # Sleep for 3 hours (in seconds)

# mkdir -p logs
for RANDOM_SEED in "${RANDOM_SEEDs[@]}"; do
    # mkdir -p logs/seed_${RANDOM_SEED}  # Ensure log directory exists
    # Batch processing
    total_datasets=${#DATASETS[@]}
    for (( start_idx=0; start_idx<total_datasets; start_idx+=k )); do
        # echo "Processing batch starting at index $start_idx..."

        for (( i=start_idx; i<start_idx+k && i<total_datasets; i++ )); do
            dname="${DATASETS[$i]}"
            dname=$(basename "$dname" .npz)

            for j in {45..50}; do # 7 models
                core_id=$(( (i * 6 + j) % MAX_CORES ))

                # Run the process
                echo "$core_id $dname $RANDOM_SEED" >> ./logs/All_log.log
                nohup taskset -c "$core_id" python FullExperiments.py -d "$dname" -i "$j" -r "$RANDOM_SEED" -t "$TIME_LIMIT" -m "$MEMORY_LIMIT" >> "./logs/seed_${RANDOM_SEED}/run_${dname}_model_${j}.log" 2>&1 &
            done
        done
        # Wait for all background processes to complete before sleeping
        # echo "Sleeping for 3 hours..." >> ./logs/All_log.log
        # sleep $SLEEP_TIME # For High-dimensional datasets, we send the task every 3 hours
        wait
    done
    wait
done
echo "HIGH-DIMENSIONAL datasets processed." >> ./logs/All_log.log
wait