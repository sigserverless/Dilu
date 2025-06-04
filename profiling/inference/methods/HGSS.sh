#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`
export CUDA_VISIBLE_DEVICES=0

QoS=0.05  
parallelism=1
cores_per_process=4
temp_file=$(mktemp)
SERVER_PID=$(echo get_server_list | nvidia-cuda-mps-control)


sm_rate=10  
batch_size=1  


best_sm_rate=$sm_rate
best_batch_size=$batch_size
best_efficiency=0

last_sm_rate=$sm_rate
last_batch_size=$batch_size
last_efficiency=$best_efficiency


# initialize: find the first configuration that meets QoS.
while true; do
    result=$(get_simulated_result $batch_size $sm_rate)
    read -r curr_batch curr_sm elapsed_time throughput <<< $(echo $result | tr ',' ' ')
    efficiency=$(echo "$throughput / $curr_sm" | bc -l)


    curr_batch=$(format_float $curr_batch)
    curr_sm=$(format_float $curr_sm)

    echo "Initial Check: SM=$curr_sm, Batch Size=$curr_batch, Time=$elapsed_time, Efficiency=$efficiency"

    if (( $(echo "$elapsed_time <= $QoS" | bc -l) )); then
        best_sm_rate=$curr_sm
        best_batch_size=$curr_batch
        best_efficiency=$efficiency

        last_sm_rate=$curr_sm
        last_batch_size=$curr_batch
        last_efficiency=$efficiency

        batch_size=$((batch_size * 2))
        break
    fi

    sm_rate=$((sm_rate + 10))
done


while true; do
    export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
    export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$sm_rate
    echo set_active_thread_percentage $SERVER_PID $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE | nvidia-cuda-mps-control

    for i in $(seq 1 $parallelism); do
        start_core=$(( (i-1)*cores_per_process ))
        end_core=$(( start_core + cores_per_process - 1 ))
        taskset -c $start_core-$end_core python ./scripts/llama2.py \
            --batch_size $batch_size --sm $sm_rate --model_name_or_path /cluster/models/llama-2-7b-hf/ \
            --device 0 >> $temp_file &
    done
    wait

    read -r curr_batch curr_sm elapsed_time throughput < <(tail -n 1 $temp_file)
    efficiency=$(echo "$throughput / $curr_sm" | bc -l)

    echo "Current: SM=$curr_sm, Batch Size=$curr_batch, Time=$elapsed_time, Efficiency=$efficiency"

    if (( $(echo "$elapsed_time > $QoS" | bc -l) )); then
        sm_rate=$((sm_rate + 10))
        export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$sm_rate
        echo set_active_thread_percentage $SERVER_PID $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE | nvidia-cuda-mps-control
        for i in $(seq 1 $parallelism); do
            taskset -c $start_core-$end_core python ./scripts/llama2.py \
                --batch_size $batch_size --sm $sm_rate --model_name_or_path /cluster/models/llama-2-7b-hf/ \
                --device 0 >> $temp_file &
        done
        wait

        read -r next_batch next_sm next_elapsed next_throughput < <(tail -n 1 $temp_file)
        next_efficiency=$(echo "$next_throughput / $next_sm" | bc -l)

        if (( $(echo "$next_elapsed > $QoS" | bc -l) )); then
            echo "QoS violated after increase. Returning best configuration: SM=$best_sm_rate, Batch Size=$best_batch_size"
            break
        fi

        if (( $(echo "$next_efficiency < ($last_efficiency-0.2)" | bc -l) )); then
            echo "Efficiency dropped. Returning best configuration: SM=$best_sm_rate, Batch Size=$best_batch_size"
            break
        fi

        best_sm_rate=$next_sm
        best_batch_size=$next_batch
        best_efficiency=$next_efficiency

        last_sm_rate=$next_sm
        last_batch_size=$next_batch
        last_efficiency=$next_efficiency

        batch_size=$((batch_size * 2))
    else
        best_sm_rate=$curr_sm
        best_batch_size=$curr_batch
        best_efficiency=$efficiency

        last_sm_rate=$curr_sm
        last_batch_size=$curr_batch
        last_efficiency=$efficiency

        batch_size=$((batch_size * 2))
    fi
done

echo "Best Configuration: SM=$best_sm_rate, Batch Size=$best_batch_size, Efficiency=$best_efficiency"
rm $temp_file
