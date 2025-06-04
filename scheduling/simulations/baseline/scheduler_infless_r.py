from flask import Flask, request, jsonify
import threading
import uuid


class PortManager:
    def __init__(self, start=15000, end=20000):
        self.available_ports = set(range(start, end + 1))
        self.lock = threading.Lock()

    def allocate_port(self):
        with self.lock:
            if not self.available_ports:
                return None  
            return self.available_ports.pop()

    def release_port(self, port):
        with self.lock:
            if 15000 <= port <= 20000:
                self.available_ports.add(port)

class GPU:
    def __init__(self, id, total_memory, sm_total, ip_address, index):
        self.id = id
        self.total_memory = total_memory
        self.sm_total = sm_total
        self.current_sm_req = 0
        self.current_sm_lim = 0
        self.current_memory = 0
        self.instances = {}
        self.ip_address = ip_address
        self.index = index

    def update_resources(self, instance, sm_req, sm_lim, memory):
        self.current_sm_req += sm_req
        self.current_sm_lim += sm_lim
        self.current_memory += memory
        self.instances[instance.instance_id] = instance

    def can_allocate(self, sm_req, sm_lim, memory, omega, gamma):
        return (self.current_sm_req + sm_req <= omega * self.sm_total and
                self.current_sm_lim + sm_lim <= gamma * self.sm_total and
                self.current_memory + memory <= self.total_memory)

    def calculate_score(self, sm_req, memory, alpha, beta):
        sm_fragmentation = 1 - (self.current_sm_req + sm_req) / self.sm_total
        return sm_fragmentation



class Instance:
    def __init__(self, id, memory, sm_requests, sm_limits, image, type, service_name, allocated_port):
        self.instance_id = id
        self.memory = memory
        self.sm_req = sm_requests
        self.sm_lim = sm_limits
        self.image = image
        self.type = type
        self.service_name = service_name
        self.deployed_gpus = []
        self.port = allocated_port

    def assign_to_gpu(self, deployed_gpu):
        self.deployed_gpus.append(deployed_gpu)


nodes_info = []
for node_ip in range(1000): 
    for i in range(4):
        nodes_info.append({"ip": str(node_ip), "index": i})


alpha = 0.6
beta = 0.4
omega = 1
gamma = 1.5 
new_gpus = [GPU(i, 40, 1, node['ip'], node['index']) for i, node in enumerate(nodes_info)]
active_gpus = []
lock = threading.Lock()

def find_colocated_GPUs(service_name):
    candidate_gpus = set()
    for active_gpu in active_gpus:
        early_instnace_existance = False
        # to find early instance
        for _, instance in active_gpu.instances.items():
            if instance.service_name == service_name:
                early_instnace_existance = True
                break
        if early_instnace_existance:
            # judge that whether having colocated training task
            for _, instance in active_gpu.instances.items():
                if instance.type == 'training':
                    for gpu in instance.deployed_gpus:
                        candidate_gpus.add(gpu)
                    candidate_gpus.remove(active_gpu) # at the same time, remove current active_gpu
        # otherwise, current active_gpu does not have the early instance
    return list(candidate_gpus)



def select_optimal_GPU(candidate_gpus, sm_req, sm_lim, memory, omega, gamma, alpha, beta):
    best_score = float('inf')
    best_gpu = None
    for gpu in candidate_gpus:
        if gpu.can_allocate(sm_req, sm_lim, memory, omega, gamma):
            score = gpu.calculate_score(sm_req, memory, alpha, beta)
            if score < best_score:
                best_score = score
                best_gpu = gpu
    return best_gpu


def update_deploy_info(instance, sm_req, sm_lim, memory, best_gpu, selected_gpus):
    best_gpu.update_resources(instance, sm_req, sm_lim, memory)
    instance.assign_to_gpu(best_gpu)
    selected_gpus.append({'id': best_gpu.id, 'ip': best_gpu.ip_address, 'index': best_gpu.index})




def schedule_instances(data):
    
    instance_id = data['service_name']
    allocated_port = port_manager.allocate_port()
    if not allocated_port:
        return jsonify({'error': 'No available ports'}), 503
    
    n_gpus_needed = data['gpu_num']
    sm_req = data['sm_requests']
    sm_lim = sm_req
    memory = data['memory']
    image = "None"
    type = data['type']
    service_name = data['service_name']
    selected_gpus = []
    instance = Instance(instance_id, memory, sm_req, sm_lim, image, type, service_name, allocated_port)

    
    if type == 'inference' or type == 'llm-inference': # non-llm model execute best-fit allocation algorithm
        memory = memory[0]
        best_gpu = None
        # find the GPUs which coloates the eary instances of current instance with the dp/pp training instances
        # first, LB_gpus
        LB_gpus = find_colocated_GPUs(service_name)
        
        best_gpu = select_optimal_GPU(LB_gpus, sm_req, sm_lim, memory, omega, gamma, alpha, beta)
        if best_gpu:
            update_deploy_info(instance, sm_req, sm_lim, memory, best_gpu, selected_gpus)
        else:
            # second, active GPUs
            left_active_gpus = set(active_gpus) - set(LB_gpus)
            best_gpu = select_optimal_GPU(left_active_gpus, sm_req, sm_lim, memory, omega, gamma, alpha, beta)
            if best_gpu:
                update_deploy_info(instance, sm_req, sm_lim, memory, best_gpu, selected_gpus)
            else:
                # third, new_gpus/the whole new gpus
                # best_gpu = select_optimal_GPU(new_gpus, sm_req, sm_lim, memory, omega, gamma, alpha, beta)
                best_gpu = new_gpus[0] if len(new_gpus)>0 else None
                if best_gpu:
                    update_deploy_info(instance, sm_req, sm_lim, memory, best_gpu, selected_gpus)
                    # update avtivate_gpu list and new_gpu
                    new_gpus.remove(best_gpu)
                    active_gpus.append(best_gpu)
                else: # no enough GPUs
                    return jsonify({'error': 'Not enough resources'}), 400
        
 
    elif type == 'training':
        node_gpus = {}
        # Group GPUs by node and check if each can accommodate its part of the task
        for gpu in active_gpus:
            if all(gpu.can_allocate(sm_req, sm_lim, mem, omega, gamma) for mem in memory):
                node_gpus.setdefault(gpu.ip_address, []).append(gpu)

        # Check if there's a node with enough GPUs
        allocated_gpus = None
        for node_ip, gpus_on_node in node_gpus.items():
            if len(gpus_on_node) >= n_gpus_needed:
                # Ensure these GPUs can be allocated according to memory requirements
                if all(len([gpu for gpu in gpus_on_node if gpu.can_allocate(sm_req, sm_lim, mem, omega, gamma)]) >= 1 for mem in memory):
                    allocated_gpus = gpus_on_node[:n_gpus_needed]
                    break

        # If not enough GPUs on any active node, check new GPUs
        if not allocated_gpus:
            for node_ip in nodes_info:  # Iterate over all possible node IPs directly from nodes_info
                new_gpus_on_node = [gpu for gpu in new_gpus if gpu.ip_address == node_ip['ip'] and all(gpu.can_allocate(sm_req, sm_lim, mem, omega, gamma) for mem in memory)]
                if len(new_gpus_on_node) + len(node_gpus.get(node_ip['ip'], [])) >= n_gpus_needed:
                    # Take needed GPUs from active and new GPUs
                    allocated_gpus = node_gpus.get(node_ip['ip'], []) + new_gpus_on_node[:n_gpus_needed - len(node_gpus.get(node_ip['ip'], []))]
                    # Update active and new GPU lists
                    for gpu in allocated_gpus:
                        if gpu in new_gpus:
                            new_gpus.remove(gpu)
                            active_gpus.append(gpu)
                    break  # Found a node with enough resources, break the loop

        # Update resource information and collect the selected GPUs
        if allocated_gpus and len(allocated_gpus) == n_gpus_needed:
            for gpu, mem in zip(allocated_gpus, memory):
                gpu.update_resources(instance, sm_req, sm_lim, mem)
                instance.assign_to_gpu(gpu)
                selected_gpus.append({'id': gpu.id, 'ip': gpu.ip_address, 'index': gpu.index})
        else:
            return jsonify({'error': 'Not enough resources'}), 400


    return {'selected_gpus': selected_gpus, 'instance_id': instance_id, 'port': allocated_port}




def delete_instance(instance_data):
    
    instance_id = instance_data['service_name']
    
    allocated_port = None 
    service_name = None
    found = False
    new_empty_gpus = []
    with lock:
        for gpu in active_gpus:
            if instance_id in gpu.instances: 
                instance = gpu.instances.pop(instance_id)
                gpu.current_sm_req -= instance.sm_req
                gpu.current_sm_lim -= instance.sm_lim
                gpu.current_memory -= instance.memory[0]
                
                found = True
                # print(gpu.current_memory, gpu.current_sm_req, gpu.current_sm_lim, len(gpu.instances))
                # Check if the GPU is now free and if so, move it to new_gpus
                if len(gpu.instances)==0:
                    
                    new_empty_gpus.append(gpu)
        
        for gpu in new_empty_gpus:
            active_gpus.remove(gpu)
            new_gpus.append(gpu)
        
    if not found:
        print("Not found:", instance_id)



def calculate_average_fragmentation(active_gpus):
    total_sm_fragmentation = 0
    total_memory_fragmentation = 0
    num_gpus = len(active_gpus)

    for gpu in active_gpus:
        sm_fragmentation = 1 - (gpu.current_sm_req / gpu.sm_total)
        memory_fragmentation = 1 - (gpu.current_memory / gpu.total_memory)

        total_sm_fragmentation += sm_fragmentation
        total_memory_fragmentation += memory_fragmentation

    average_sm_fragmentation = total_sm_fragmentation / num_gpus
    average_memory_fragmentation = total_memory_fragmentation / num_gpus

    return average_sm_fragmentation, average_memory_fragmentation




import ast
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

if __name__ == '__main__':
    port_manager = PortManager()
    with open("instances-100.txt", "r") as f:
        lines = f.readlines()
    instances = [ast.literal_eval(line.strip()) for line in lines]

    timestamps = []
    gpu_counts = []
    max_nums = -1

    base_time = datetime.strptime(instances[0]['Time'], "%Y-%m-%d %H:%M:%S")
    for event in instances:
        event_time = datetime.strptime(event['Time'], "%Y-%m-%d %H:%M:%S")
        if event['Action'] == 'start':
            schedule_instances(event['Instance'])
        else:
            delete_instance(event['Instance'])
        time_diff = (event_time - base_time).total_seconds()/60 
        timestamps.append(time_diff)
        gpu_counts.append(len(active_gpus))
        if len(active_gpus)>max_nums:
            max_nums = len(active_gpus) 
        if max_nums >=20 and  max_nums <=24 :
            average_sm_fragmentation, average_memory_fragmentation = calculate_average_fragmentation(active_gpus)
            print(f"SM Frag: {average_sm_fragmentation:.2f}")
            print(f"Memory Frag: {average_memory_fragmentation:.2f}")
    

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, gpu_counts, linestyle='-')
    plt.xlabel('Time (hour)')
    plt.ylabel('Active GPUs')
    plt.title('Active GPU Count Over Time')
    plt.xticks()
    plt.tight_layout()
    plt.savefig('infless-r-timeline.png', dpi=300)
    print("max allocated GPUs:", max_nums)
