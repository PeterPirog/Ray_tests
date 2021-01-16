import os

file='tune_keras_functional.py' #run_PPO_hard_walker.py python3  run_PPO_hard_walker.py run_agent.py



#Set2
local_result_dir='~/ray_results/' # path in physical machine /home/peterpirog/PycharmProjects/Ray_tests/ray_results/
docker_result_dir='/root/ray_results/'  #path in docker container

#docker_cmd=f'sudo docker run -p 8265:8265 -p 6379:6379 -v `pwd`:`pwd` -v {local_result_dir}:{docker_result_dir} -w `pwd` --gpus all  --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest'
docker_cmd=f'sudo docker run' \
           f' -p 8265:8265' \
           f' -p 6379:6379' \
           f' --expose=10000-10999' \
           f'  -v `pwd`:`pwd`' \
           f' -v {local_result_dir}:{docker_result_dir}' \
           f' -w `pwd`' \
           f' --gpus all' \
           f'  --cpus 8' \
           f' -it --rm' \
           f' --shm-size=16g' \
           f' --network=host' \
           f' peterpirogtf/ray_tf2:latest'

rllib_cmd=f'python3 {file}'

cmd=docker_cmd + ' ' + rllib_cmd



os.system(cmd)

"""
If You want use docker, you have to install docker application at first !!!
    WINDOWS: https://docs.docker.com/docker-for-windows/install/
    UBUNTU: https://docs.docker.com/engine/install/ubuntu/

comments for docker run options (full description here https://docs.docker.com/engine/reference/commandline/run/):
    General:
-it - option to communicate with docker image by console (you need this option to see results of training in console)
--rm - remove container after using, this option delete container with all data inside it but without it you will create a lot of trashes
 
-d - container works in background (typically container is removed after using) you can log in inside container by command:
    "docker ps"
    which returns CONTAINER_ID, copy it and use command
    "docker attach ijhweifhb0wqnih768327483y8erfhnnkfjnkfnk"   <- use id of your container, not this !!!

    Directories

    Network:
--network=host - option to connect container with physical machine network, without this option container use internal interface with address 127.0.0.1, while errors check if ports are busy    
-p 8265:8265 - option for expose dashboard port
-p 6379:6379 - option for expose ray head port
--expose=10000-10999 - option to expose workers ports

    Assets:
IMPORTANT: You can't use asset in script, if You don't declare it for Your docker container !!!
--gpus all - get access for your GPU, but:
        Check is CUDA available for Your graphic card model https://www.geforce.co.uk/hardware/technology/cuda/supported-gpus
        WINDOWS - needs correct installation and configuration of WSL2 before, it can be problematic
        UBUNTU - needs only nvidia drivers and docker image with GPU build inside
        
        You can check is CUDA available by adding in Your code:

            import tensorflow as tf
            print('Is cuda available:', tf.test.is_gpu_available())
            
--cpus 8 - number of CPU available for docker. If You declare 4, ray can use only 4 CPU even in physical machine are 8
--shm-size=16g' - it means get access to 16 GB of RAM
"""
