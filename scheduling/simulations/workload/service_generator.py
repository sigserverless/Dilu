import random
import string
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42) 

def generate_service_name(prefix):
    return f"{prefix}-{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}"

def generate_instances(total_instances, train_ratio):
    instances = []
    llm_inference_count = int(total_instances * 0.2)
    train_count = int(total_instances * train_ratio / 10) 
    inference_count = total_instances - train_count - llm_inference_count

    for _ in range(train_count):
        service_name = generate_service_name("training")
        gpu_num = random.choice([2, 4])
        memory = random.choice([25, 30, 35, 40])
        memory_list = [memory] * gpu_num
        sm_requests = round(random.uniform(0.4, 0.8), 2)
        sm_limits = max(sm_requests + 0.2, round(random.uniform(0.6, 1.0), 2))
        instance = {
            "service_name": service_name,
            "type": "training",
            "gpu_num": gpu_num,
            "sm_requests": sm_requests,
            "sm_limits": sm_limits,
            "memory": memory_list
        }
        instances.append(instance)
    
    for _ in range(inference_count):
        service_name = generate_service_name("inference")
        gpu_num = 1
        memory = random.choice([3, 4, 5, 6, 7, 8, 9, 10])
        sm_requests = round(random.uniform(0.2, 0.4), 2)
        sm_limits = max(sm_requests + 0.2, round(random.uniform(0.3, 0.6), 2))
        instance = {
            "service_name": service_name,
            "type": "inference",
            "gpu_num": gpu_num,
            "sm_requests": sm_requests,
            "sm_limits": sm_limits,
            "memory": [memory]
        }
        instances.append(instance)
    
    #  LLM 
    for _ in range(llm_inference_count):
        service_name = generate_service_name("llm-inference")
        gpu_num = 1
        memory = random.choice(range(20, 31))
        sm_requests = round(random.uniform(0.4, 0.6), 2)
        sm_limits = random.uniform(sm_requests + 0.2, min(sm_requests + 0.99, 0.8))
        instance = {
            "service_name": service_name,
            "type": "llm-inference",
            "gpu_num": gpu_num,
            "sm_requests": sm_requests,
            "sm_limits": round(sm_limits, 2),
            "memory": [memory]
        }
        instances.append(instance)
    
    return instances

def shuffle_and_delete(instances, total_instances):
    # Shuffle instances
    random.shuffle(instances)
    # Start events
    start_time = datetime.now()
    events = []
    for i, instance in enumerate(instances):
        events.append({
            "time": start_time + timedelta(seconds=i),
            "action": "start",
            "instance": instance
        })

    # Select deletable instances and schedule delete events for inference and LLM inference
    deletable = [event for event in events if event["instance"]["type"] in ["llm-inference", "inference"]]
    num_deletes = int(total_instances * 0.6)
    delete_events = random.sample(deletable, min(num_deletes, len(deletable)))
    for event in delete_events:
        delete_time = event["time"] + timedelta(minutes=random.randint(1, 3))  # Adjusted time for deletion
        events.append({
            "time": delete_time,
            "action": "delete",
            "instance": event["instance"]
        })
    
    # Select deletable training instances and schedule their delete events
    deletable_training = [event for event in events if event["instance"]["type"] == "training"]
    num_training_deletes = int(total_instances * 0.1)
    delete_training_events = random.sample(deletable_training, min(num_training_deletes, len(deletable_training)))
    for event in delete_training_events:
        delete_time = event["time"] + timedelta(minutes=random.randint(10, 15))  # 10 minutes after start for training deletes
        events.append({
            "time": delete_time,
            "action": "delete",
            "instance": event["instance"]
        })

    # Sort all events by time
    events.sort(key=lambda x: x["time"])

    return events

total_instances = 100
train_ratio = 2
instances = generate_instances(total_instances, train_ratio)
events = shuffle_and_delete(instances, total_instances)
for event in events:
    print({'Time':  event['time'].strftime('%Y-%m-%d %H:%M:%S'), 
           'Action': event['action'],
           'Instance': event['instance']
           })
