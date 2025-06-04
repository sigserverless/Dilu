#!/bin/bash
export PYTHONPATH=$PYTHONPATH:`pwd`
export CUDA_VISIBLE_DEVICES=0

parallelism=1
cores_per_process=4
temp_file=$(mktemp)
SERVER_PID=$(echo get_server_list | nvidia-cuda-mps-control)

# traverse the values of CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
for percentage in {10,20,30,40,50,60,70,80,90,100}; do
# for percentage in {10,20}; do
    # for batch_size in 16 32; do
    for batch_size in 1 2 4 8 16 32 64; do
        export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
        export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percentage
        echo set_active_thread_percentage $SERVER_PID $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE | nvidia-cuda-mps-control
        for i in $(seq 1 $parallelism); do
            start_core=$(( (i-1)*cores_per_process ))
            end_core=$(( start_core + cores_per_process - 1 ))
            # taskset -c $start_core-$end_core python ./scripts/roberta_large.py --batch_size $batch_size --sm $percentage --model_name_or_path /cluster/datasets/roberta-large/  --device 0 >> $temp_file &
            # taskset -c $start_core-$end_core python ./scripts/resnet152.py --batch_size $batch_size --sm $percentage --device 0 >> $temp_file &
            # taskset -c $start_core-$end_core python ./scripts/gpt2_large.py --batch_size $batch_size --sm $percentage --device 0 --model_name_or_path /cluster/datasets/gpt2-large/  >> $temp_file &
            # taskset -c $start_core-$end_core python ./scripts/bert_base.py --batch_size $batch_size --sm $percentage --model_name_or_path /cluster/datasets/bert-config/ --device 0 >> $temp_file &
            taskset -c $start_core-$end_core python ./scripts/llama2.py --batch_size $batch_size --sm $percentage --model_name_or_path /cluster/models/llama-2-7b-hf/ --device 0 >> $temp_file & 
        done
        wait
    done
done

cat $temp_file
rm $temp_file
