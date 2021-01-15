import os
#Set 1
docker_cmd='docker run -v `./ray_results/`:`/root/ray_results/` -w `pwd` --cpus 8 --gpus all -it -p 8265:8265 --rm peterpirogtf/ray_tf2'
rllib_cmd='rllib train --run=TD3 --env=BipedalWalker-v3'



#Set2
local_result_dir='/home/peterpirog/ray_results/' # path in physical machine /home/peterpirog/PycharmProjects/Ray_tests/ray_results/
docker_result_dir='/root/ray_results/'  #path in docker container

#docker_cmd=f'sudo docker run -p 8265:8265 -p 6379:6379 -v `pwd`:`pwd` -v {local_result_dir}:{docker_result_dir} -w `pwd` --gpus all  --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest'
docker_cmd=f'sudo docker run -p 8265:8265 -p 6379:6379 -p 6006:6006 --expose=10000-10999  -v `pwd`:`pwd` -v {local_result_dir}:{docker_result_dir} -w `pwd`' \
           f' --gpus all  --cpus 8 -it --rm --shm-size=16g --network=host peterpirogtf/ray_tf2:latest'

rllib_cmd='python3 pb2_ppo_example.py'  #run_PPO_hard_walker.py python3  run_PPO_hard_walker.py run_agent.py

cmd=docker_cmd + ' ' + rllib_cmd

cmd_head=f'sudo docker run -d -p 8265:8265 -p 6379:6379 --expose=10000-10999  -v `pwd`:`pwd` -v {local_result_dir}:{docker_result_dir} -w `pwd` ' \
         f'--gpus all  --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest ' \
         f'ray start --head --port=6379 --include-dashboard=True --dashboard-host=0.0.0.0'

cmd_head2=f'sudo docker run -d -p 8265:8265 -p 6379:6379 --expose=10000-10999  -v `pwd`:`pwd` -v {local_result_dir}:{docker_result_dir} -w `pwd` ' \
         f'--gpus all  --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest'

cmd_start=f'sudo docker run -p 8265:8265 -p 6379:6379 --expose=10000-10999  -v `pwd`:`pwd` -v {local_result_dir}:{docker_result_dir} -w `pwd` ' \
         f'--gpus all  --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest ' \

#         f' ray start --address='192.168.1.16:6379' --redis-password='5241590000000000'

os.system(cmd)

# DOCKER COMMAND TU RUN DOCKER CONTAINER IN BACKGROUND FOR UBUNTU:
#f'sudo docker run -d -p 8265:8265 -p 6379:6379 --expose=10000-10999  -v `pwd`:`pwd` -v {local_result_dir}:{docker_result_dir} -w `pwd` ' \
#         f'--gpus all  --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest'

# DOCKER COMMAND TU RUN DOCKER CONTAINER IN BACKGROUND FOR WINDOWS:
#docker run -d -p 8265:8265 -p 6379:6379 --expose=10000-10999  -v ${PWD}:`pwd` -w `pwd` --gpus all  --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest
#docker run -d -p 8265:8265 -p 6379:6379 --expose=10000-10999 --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest

#create or update ray cluster 'ray up [OPTIONS] CLUSTER_CONFIG_FILE' https://docs.ray.io/en/master/package-ref.html

#tensorboard --logdir ~/ray_results