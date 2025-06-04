import subprocess
import shlex   

def start_instance(selected_gpus, instance_id, image_name, service_name, args, allocated_port, ip_address):
    gpus = ','.join([str(gpu['index']) for gpu in selected_gpus])

    volumes = {
        '/home/lvcunchi/Dilu_sys/Dilu_node_scheduler': '/cluster', # for dilu
        '/home/lvcunchi/Dilu_sys/gsharing': '/etc/gsharing',
        '/home/lvcunchi/benchmarks/dilu_benchmarks/gdGPT_container/': '/gdGPT_container',
        '/home/lvcunchi/benchmarks/models/': '/cluster/models', 
        '/home/lvcunchi/Dilu_sys/cluster_logs': '/cluster/workloads/job_logs',
        '/usr/local/cuda-11.7': '/usr/local/cuda',
    }
    environment = {
        'CUDA_MPS_PIPE_DIRECTORY': '/tmp/nvidia-mps', # for MPS
        'CUDA_VISIBLE_DEVICES': gpus, # for MPS
        'requests_rate': str(args.get('sm_requests', 0.25)),
        'limits_rate': str(args.get('sm_limits', 0.75)),
        'is_llm': str(args.get('is_llm', 0)),
        'priority': args.get('priority', 'low'),
    }
    
    if "deepspeed" not in service_name:
        command = args.get('COMMAND', '') + " --port {} ".format(allocated_port)
    else:
        command = args.get('COMMAND', '') 
        
    docker_command = f"docker run --name {service_name}-{instance_id} --gpus device=all --ipc host --network host --cap-add SYS_NICE -u root --shm-size 32g -d"
    for host_path, container_path in volumes.items():
        docker_command += f" -v {shlex.quote(host_path)}:{shlex.quote(container_path)}"
    for key, value in environment.items():
        docker_command += f" -e {shlex.quote(key)}={shlex.quote(value)}"
    output_file = f'/cluster/workloads/job_logs/{service_name}-{instance_id}.log'
    docker_command += f" {image_name} bash -c '{command} > {output_file} 2>&1'"

    print(f"Generated Docker command: {docker_command}")

    ssh_command = f"ssh {ip_address} {shlex.quote(docker_command)}"
    print(f"Executing SSH command: {ssh_command}")
    result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True)

    print(f"SSH command output: {result.stdout}")
    print(f"SSH command error: {result.stderr}")
    print(f"SSH command return code: {result.returncode}")

    if result.returncode != 0:
        raise RuntimeError(f"SSH command failed with return code {result.returncode}")


def stop_instance(service_name, instance_id, ip_address):
    docker_name = f"{service_name}-{instance_id}"
    ssh_command = f"ssh {ip_address} 'docker stop {docker_name} && docker rm {docker_name}  '"
    subprocess.run(ssh_command, shell=True)
