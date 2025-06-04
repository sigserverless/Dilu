#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`
export CUDA_VISIBLE_DEVICES=0

parallelism=1
cores_per_process=4
temp_file=$(mktemp)
SERVER_PID=$(echo get_server_list | nvidia-cuda-mps-control)

low=0
high=100

export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
echo set_active_thread_percentage $SERVER_PID $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE | nvidia-cuda-mps-control

for i in $(seq 1 $parallelism); do
    start_core=$(( (i-1)*cores_per_process ))
    end_core=$(( start_core + cores_per_process - 1 ))
    taskset -c $start_core-$end_core python ./scripts/train_model_a.py --model_name_or_path /cluster/models/model_a_files/ >> $temp_file &
done
wait

read -r T1 < <(tail -n 1 $temp_file)
echo "Throughput at SM=100%: T1=$T1"

while true; do
    mid=$(( (low + high) / 2 / 5 * 5 ))

    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mid
    echo set_active_thread_percentage $SERVER_PID $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE | nvidia-cuda-mps-control

    for i in $(seq 1 $parallelism); do
        taskset -c $start_core-$end_core python ./scripts/train_model_a.py --model_name_or_path /cluster/models/model_a_files/ >> $temp_file &
    done
    wait

    read -r T2 < <(tail -n 1 $temp_file)
    echo "Throughput at SM=$mid%: T2=$T2"

    if (( $(echo "$T2 >= 0.98 * $T1" | bc -l) )) || (( high == low )); then
        echo "Optimal SM rate found: $mid%"
        best_sm_rate=$mid
        break
    fi

    if (( $(echo "$T2 < 0.98 * $T1" | bc -l) )); then
        low=$mid
    else
        high=$mid
    fi
done

echo "Best SM rate: $best_sm_rate%"
rm $temp_file
