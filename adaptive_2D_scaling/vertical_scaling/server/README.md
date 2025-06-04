## Starting Real-Time CUDA Kernel Manager (RCKM)

Before submitting any tasks, please start the Real-time CUDA Kernel Manager (RCKM).

### How to Run RCKM

#### Launch a container
- docker run --name RCKM --gpus "device=all" --ipc host --cap-add SYS_PTRACE -u root --shm-size 32g -v /home/lvcunchi/Dilu_sys/gsharing:/etc/gsharing -v /usr/local/cuda:/usr/local/cuda lvcunchi1999/torch110cu111_ddp:cluster bash

#### Containerized launch: copy RCKM into the container
-  docker cp adaptive_2D_scaling/vertical_scaling/intercept/server/RCKM <container_id>:/root/RCKM

#### Inside the container, start RCKM
- bash /root/RCKM

Once RCKM is running, you can efficiently submit training or inference tasks.
