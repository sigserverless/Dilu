from flask import Flask, request, jsonify
import threading
import uuid
import utils_docker


class PortManager:
    def __init__(self, start=15000, end=20000):
        self.available_ports = set(range(start, end + 1))
        self.lock = threading.Lock()

    def allocate_port(self):
        with self.lock:
            if not self.available_ports:
                return None  #       
            return self.available_ports.pop()

    def release_port(self, port):
        with self.lock:
            if 15000 <= port <= 20000:
                self.available_ports.add(port)

class GPU:
    def __init__(self, id, total_memory, total_sm, ip_address, index):
        self.id = id
        self.total_memory = total_memory
        self.total_sm = total_sm
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
        return (self.current_sm_req + sm_req <= omega * self.total_sm and
                self.current_sm_lim + sm_lim <= gamma * self.total_sm and
                self.current_memory + memory <= self.total_memory)

    def calculate_score(self, sm_req, memory, alpha, beta):
        sm_fragmentation = 1 - (self.current_sm_req + sm_req) / self.total_sm
        memory_fragmentation = 1 - (self.current_memory + memory) / self.total_memory
        return alpha * sm_fragmentation + beta * memory_fragmentation



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

app = Flask(__name__)
nodes_info = [
    {"ip": "172.16.32.46", "index": 0},
    {"ip": "172.16.32.46", "index": 1},
    {"ip": "172.16.32.46", "index": 2},
    {"ip": "172.16.32.46", "index": 3},
    
    {"ip": "172.16.32.45", "index": 0}, 
    {"ip": "172.16.32.45", "index": 1},
    {"ip": "172.16.32.45", "index": 2},
    {"ip": "172.16.32.45", "index": 3},
    
    {"ip": "172.16.120.44", "index": 0}, 
    {"ip": "172.16.120.44", "index": 1},
    {"ip": "172.16.120.44", "index": 2},
    {"ip": "172.16.120.44", "index": 3},
    
    {"ip": "172.16.32.6", "index": 0}, 
    {"ip": "172.16.32.6", "index": 1},
    {"ip": "172.16.32.6", "index": 2},
    {"ip": "172.16.32.6", "index": 3},
    
]


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



def print_cluster_resources():
    print("Current Cluster Resource Usage:")
    print("==============active_gpus==================")
    for gpu in active_gpus:  # Assuming new_gpus should also be monitored
        print(f"GPU {gpu.id} on Node {gpu.index} of [{gpu.ip_address}]:")
        print(f"  Total Memory: {gpu.total_memory} GB")
        print(f"  Used Memory: {gpu.current_memory} GB")
        print(f"  SM Total: {gpu.total_sm}")
        print(f"  Current SM Requests: {gpu.current_sm_req}")
        print(f"  Current SM Limits: {gpu.current_sm_lim}")
        print(f"  Instances: {len(gpu.instances)} running")
        for instance_id, details in gpu.instances.items():
            print(f"    Instance {instance_id}: Req {details.sm_req}, Lim {details.sm_lim}, Mem {details.memory} GB")
    print("")
    
    print("==============new_gpus==================")
    for gpu in new_gpus: 
        print(f"GPU {gpu.id} on Node {gpu.index} of [{gpu.ip_address}]:")
        print(f"  Total Memory: {gpu.total_memory} GB")
        print(f"  Used Memory: {gpu.current_memory} GB")
        print(f"  SM Total: {gpu.total_sm}")
        print(f"  Current SM Requests: {gpu.current_sm_req}")
        print(f"  Current SM Limits: {gpu.current_sm_lim}")
        print(f"  Instances: {len(gpu.instances)} running")
        for instance_id, details in gpu.instances.items():
            print(f"    Instance {instance_id}: Req {details.sm_req}, Lim {details.sm_lim}, Mem {details.memory} GB")
    print("")


    

def start_instance(selected_gpus, instance_id, args, allocated_port):
    ip_address = selected_gpus[0]['ip']
    utils_docker.start_instance(selected_gpus, instance_id, args['image'], args['service_name'], args, allocated_port, ip_address)


def stop_instance(service_name, instance_id, ip_address):
    utils_docker.stop_instance(service_name, instance_id, ip_address)



@app.route('/schedule', methods=['POST'])
def schedule_instances():
    data = request.get_json()
    instance_id = str(uuid.uuid4())  # Generate a unique instance ID
    allocated_port = port_manager.allocate_port()
    if not allocated_port:
        return jsonify({'error': 'No available ports'}), 503
    
    n_gpus_needed = data['num']
    sm_req = data['sm_requests']
    sm_lim = data['sm_limits']
    memory = data['memory']
    image = data['image']
    type = data['type']
    service_name = data['service_name']
    selected_gpus = []

    instance = Instance(instance_id, memory, sm_req, sm_lim, image, type, service_name, allocated_port)

    with lock:
        if type == 'inference': # non-llm model execute best-fit allocation algorithm
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
                    best_gpu = select_optimal_GPU(new_gpus, sm_req, sm_lim, memory, omega, gamma, alpha, beta)
                    if best_gpu:
                        update_deploy_info(instance, sm_req, sm_lim, memory, best_gpu, selected_gpus)
                        # update avtivate_gpu list and new_gpu
                        new_gpus.remove(best_gpu)
                        active_gpus.append(best_gpu)
                    else: # no enough GPUs
                        return jsonify({'error': 'Not enough resources'}), 400
          
        elif type == 'llm-inference':  # llm model worst-fit algorithm
            best_gpu = None
            best_fit_score = float('inf') 
            for gpu in active_gpus:
                if gpu.can_allocate(sm_req, sm_lim, memory, omega, gamma):
                    score = gpu.current_memory 
                    if score < best_fit_score:
                        best_fit_score = score
                        best_gpu = gpu

            if best_gpu:
                best_gpu.update_resources(instance.id, sm_req, sm_lim, memory)
                instance.assign_to_gpu(best_gpu)
                selected_gpus.append({'id': best_gpu.id, 'ip': best_gpu.ip_address, 'index': best_gpu.index})
            else:
                node_memory_groups = {}
                for gpu in active_gpus:
                    if gpu.can_allocate(sm_req, sm_lim, 0, omega, gamma):
                        node_memory_groups.setdefault(gpu.ip_address, []).append(gpu)
                found = False
                
                for node, gpus_on_node in node_memory_groups.items():
                    gpus_on_node.sort(key=lambda x: x.current_memory, reverse=True)           
                    total_available_memory = sum(gpu.total_memory - gpu.current_memory for gpu in gpus_on_node)
                    if total_available_memory >= memory:
                        remaining_memory = memory
                        for gpu in gpus_on_node:
                            if remaining_memory <= 0:
                                break
                            alloc_memory = min(gpu.total_memory - gpu.current_memory, remaining_memory)
                            gpu.update_resources(instance, sm_req, sm_req, alloc_memory) 
                            instance.assign_to_gpu(gpu)
                            selected_gpus.append({'id': gpu.id, 'ip': gpu.ip_address, 'index': gpu.index})
                            remaining_memory -= alloc_memory
                        found = True
                        break
                    
                if not found:
                    # If no active GPUs can handle it, start a new GPU instance
                    if new_gpus:
                        new_gpu = new_gpus.pop(0)
                        new_gpu.update_resources(instance, sm_req, sm_lim, memory)
                        instance.assign_to_gpu(new_gpu)
                        active_gpus.append(new_gpu)
                        selected_gpus.append({'id': new_gpu.id, 'ip': new_gpu.ip_address, 'index': new_gpu.index})
                    else:
                        return jsonify({'error': 'Unable to allocate resources, and no new GPUs available'}), 400
                    
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

    thrd = threading.Thread(target=start_instance, args=(selected_gpus, instance_id, data, allocated_port))
    thrd.start()
    print_cluster_resources()
    return jsonify({'selected_gpus': selected_gpus, 'instance_id': instance_id, 'port': allocated_port}), 200


@app.route('/delete_instance', methods=['POST'])
def delete_instance():
    data = request.get_json()
    instance_id = data['instance_id']
    
    allocated_port = None 
    service_name = None
    ip_address = None
    found = False
    with lock:
        for gpu in active_gpus:
            if instance_id in gpu.instances:
                instance = gpu.instances.pop(instance_id)
                gpu.current_sm_req -= instance.sm_req
                gpu.current_sm_lim -= instance.sm_lim
                gpu.current_memory -= instance.memory
                found = True
                ip_address = instance.deployed_gpus[0].ip_address
                
                allocated_port = instance.port
                service_name = instance.service_name
                # Check if the GPU is now free and if so, move it to new_gpus
                if not gpu.instances and gpu.current_memory == 0 and gpu.current_sm_req == 0 and gpu.current_sm_lim == 0:
                    active_gpus.remove(gpu)
                    new_gpus.append(gpu)
                break

    if not found:
        return jsonify({'error': 'Instance not found'}), 404

    port_manager.release_port(allocated_port)  #                 
    thrd = threading.Thread(target=stop_instance, args=(service_name, instance_id, ip_address,))
    thrd.start()
    print_cluster_resources()
    return jsonify({'status': 'deleted', 'instance_id': instance_id}), 200


if __name__ == '__main__':
    port_manager = PortManager()
    app.run(debug=False, host='0.0.0.0', port=5000)
