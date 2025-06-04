#!/bin/bash


# data_file="../observations/results/bs_sm_resnet152.csv"
# QoS=0.04  

# data_file="../observations/results/bs_sm_roberta_large.csv"
# QoS=0.06 

data_file="../observations/results/bs_sm_gpt2_large.csv"
QoS=0.06 

# data_file="../observations/results/bs_sm_llama2.csv"
# QoS=0.042 

sm_rate=10  
batch_size=1  

format_float() {
    printf "%.6f" "$1"
}

best_sm_rate=$sm_rate
best_batch_size=$batch_size
best_efficiency=0

last_sm_rate=$sm_rate
last_batch_size=$batch_size
last_efficiency=$best_efficiency

function get_simulated_result() {
    grep "^$1.000000,$2.000000" $data_file | head -n 1
}


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
    result=$(get_simulated_result $batch_size $sm_rate)
    read -r curr_batch curr_sm elapsed_time throughput <<< $(echo $result | tr ',' ' ')
    efficiency=$(echo "$throughput / $curr_sm" | bc -l)

    echo "Current: SM=$curr_sm, Batch Size=$curr_batch, Time=$elapsed_time, Efficiency=$efficiency"

    if (( $(echo "$elapsed_time > $QoS" | bc -l) )); then
        sm_rate=$((sm_rate + 10))
        # batch_size=$((batch_size * 2))

        result=$(get_simulated_result $batch_size $sm_rate)
        read -r next_batch next_sm next_elapsed next_throughput <<< $(echo $result | tr ',' ' ')
        next_efficiency=$(echo "$next_throughput / $next_sm" | bc -l)

        next_batch=$(format_float $next_batch)
        next_sm=$(format_float $next_sm)
        echo "Current: SM=$next_sm, Batch Size=$next_batch, Time=$next_elapsed, Efficiency=$next_efficiency"
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
