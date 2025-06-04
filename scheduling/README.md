# Cluster Scheduling

It implements cluster scheduling for training and inference workloads, including deploying funcs, routing inference requests and horizontal scaling logic of inference instances to manage GPU resources effectively.


## Component Overview

1. **scheduler.py**  
   Receives user-submitted training and inference tasks and deploys them according to scheduling principles.

2. **scaler.py**  
   - **Router**: Load-balances incoming inference requests and routes them to the appropriate Docker-based inference instances.  
   - **Horizontal Scaling Logic**: Monitors each service’s requests-per-second (RPS) with a sliding window and performs lazy scale-in/scale-out.  
     
        python scheduler.py  
        python scaler.py  

3. **scripts_deploy/**  
   - Scripts for deploying inference and training tasks.  
   - Script for sending inference requests.

4. **scripts_tasks/**  
   - Each script corresponds to a workload definition for each training/inference task.

  
5. **simulations/**  
   Cluster-scheduling logic and test scripts for large-scale simulations.

6. **baselines/**  
   Baseline scheduling implementations.

7. **data/**  
   Sample datasets or configuration files used by training/inference tasks.


---

## Image Preparation for Deployment

The deployment scripts in **scripts_deploy/** expect a custom image such as  
`lvcunchi1999/torch110cu111_ddp:dilu`.  
You can build this image yourself using the client libraries under
`adaptive_2D_scaling/vertical_scaling/client-libs` (helpful for understanding the construction process of the Dilu system), or use the pre-built image from Docker Hub.

### Step-by-Step Build Guide

1. Pull the base image from Docker Hub  
-  docker pull lvcunchi1999/torch110cu111_ddp:cluster  

2. Start a container and copy the required shared libraries  
   (e.g. xxx.so, xxx.so.1, …) into `/` or your preferred directory:  

- docker cp ./client-libs/*.so <container_id>:/  

3. Replace the default preload list  

- docker cp ./client-libs/ld.so.preload <container_id>:/etc/ld.so.preload  

4. Copy the task scripts into the container

- docker cp ./scripts_tasks/ <container_id>:/scripts_tasks/

5. Commit or save the container as a new image  

- docker commit <container_id> lvcunchi1999/torch110cu111_ddp:dilu  
    
The resulting `lvcunchi1999/torch110cu111_ddp:dilu` image is now ready for use by the deployment scripts. As for the DeepSpeed pipeline-parallel fine-tuning, please repeat the process above based on  the `lvcunchi1999/torch110cu111_deepspeed:latest` image from Docker Hub.

Before submitting the task, we should first start the Real time CUDA Kernel Manager (RCKM). The core code of this component is the local_stcaler in adaptive_2D_scaling/vertical_Scaling/interpept/server. The specific execution method can be found in corresponding README.md file.