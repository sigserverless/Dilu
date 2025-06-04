
import requests
import json
import time

training_jobs = [
    {
        'service_name': 'dp-bert-training',
        'image': 'lvcunchi1999/torch110cu111_ddp:dilu',  
        'num_gpus': 4,
        'sm_requests': 0.3,
        'sm_limits': 0.9,
        'memory': [33,33,33,33],
        'is_llm': 0,
        'priority': 'low',
        'task_type': 'training',
        'throughput': 0, 
        'commands': 'NCCL_SOCKET_IFNAME=eno1 python -u /cluster/workloads/dp_bert.py --nodes 1 --gpus 4 --nr 0  --iteration 600  --batch_size 192',
    },
    # 2-workers
    {
        'service_name': 'dp-roberta-training',
        'image': 'lvcunchi1999/torch110cu111_ddp:dilu', 
        'num_gpus': 2,
        'sm_requests': 0.5,
        'sm_limits': 0.9,
        'memory': [29, 29],
        'is_llm': 0,
        'priority': 'low',
        'task_type': 'training',
        'throughput': 0,
        'commands': 'NCCL_SOCKET_IFNAME=eno1 python -u /cluster/workloads/dp_roberta.py --nodes 1 --gpus 2 --nr 0  --iteration 1000  --batch_size 64',
    },
    # 2-workers
    {
        'service_name': 'dp-resnet152-training',
        'image': 'lvcunchi1999/torch110cu111_ddp:dilu',  
        'num_gpus': 2,
        'sm_requests': 0.3, 
        'sm_limits': 0.6, 
        'memory': [28, 28],
        'is_llm': 0,
        'priority': 'low',
        'task_type': 'training',
        'throughput': 0,
        'commands': 'NCCL_SOCKET_IFNAME=eno1 python -u /cluster/workloads/dp_resnet152.py --nodes 1 --gpus 2 --nr 0  --iteration 100 --batch_size 128', 
    },
    {
        'service_name': 'deepspeed-llama2-training',
        'image': 'lvcunchi1999/torch110cu111_deepspeed:cluster',  
        'num_gpus': 4,
        'sm_requests': 0.5, 
        'sm_limits': 0.9, 
        'memory': [35, 39, 39, 35],
        'is_llm': 0,
        'priority': 'low',
        'task_type': 'training',
        'throughput': 0,
        'commands': 'NCCL_SOCKET_IFNAME=eno1 deepspeed /gdGPT_container/train_ds.py --config /gdGPT_container/configs/ds_config_pp_llama2.yml', 
    }
]

register_url = "http://localhost:14999/register_service"
for new_service_data in training_jobs:
    time.sleep(5)
    response = requests.post(register_url, data=json.dumps(new_service_data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        print("Service registered successfully:", response.json()["message"])
    else:
        print("Failed to register service. Status code:", response.status_code, "Message:", response.text)