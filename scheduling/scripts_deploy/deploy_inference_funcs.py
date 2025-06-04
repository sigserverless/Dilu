import requests
import json
import time

inference_services = [
    {
        'service_name': 'llama2-7B-inf',
        'image': 'lvcunchi1999/torch110cu111_ddp:dilu',  
        'num_gpus': 1,
        'sm_requests': 0.5,
        'sm_limits': 0.6,
        'memory': 28,
        'is_llm': 1,
        'priority': 'high',
        'task_type': 'llm-inference',
        'throughput': 10, 
        'commands': 'python /cluster/workloads/run_LLaMA2_INF.py --model_name_or_path /cluster/models/llama-2-7b-hf/ ',
    },
    {
        'service_name': 'RoBERTa-inf',
        'image': 'lvcunchi1999/torch110cu111_ddp:dilu', 
        'num_gpus': 1,
        'sm_requests': 0.3,
        'sm_limits': 0.6,
        'memory': 6,
        'is_llm': 0,
        'priority': 'high',
        'task_type': 'inference',
        'throughput': 48, 
        'commands': 'python /cluster/workloads/run_Roberta_INF_batch.py --model_name_or_path /cluster/datasets/roberta-large/',
    },
    {
        'service_name': 'Bert-inf',
        'image': 'lvcunchi1999/torch110cu111_ddp:dilu',  
        'num_gpus': 1,
        'sm_requests': 0.1,
        'sm_limits': 0.2,
        'memory': 5,
        'is_llm': 0,
        'priority': 'high',
        'task_type': 'inference',
        'throughput': 48,
        'commands': 'python /cluster/workloads/run_BERT_INF_batch.py --model_name_or_path /cluster/datasets/bert-config/ ',
    },
]


register_url = "http://localhost:14999/register_service"
for new_service_data in inference_services:
    time.sleep(5)
    response = requests.post(register_url, data=json.dumps(new_service_data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        print("Service registered successfully:", response.json()["message"])
    else:
        print("Failed to register service. Status code:", response.status_code, "Message:", response.text)

