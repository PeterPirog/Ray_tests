import os
#Set 1
docker_cmd='docker run -v `./ray_results/`:`/root/ray_results/` -w `pwd` --cpus 8 --gpus all -it -p 8265:8265 --rm peterpirogtf/ray_tf2'
rllib_cmd='rllib train --run=TD3 --env=BipedalWalker-v3'



#Set2
local_result_dir='/home/peterpirog/PycharmProjects/Ray_tests/ray_results/'
docker_result_dir='/root/ray_results/'

docker_cmd=f'sudo docker run -p 8265:8265 -p 6379:6379 -v `pwd`:`pwd` -v {local_result_dir}:{docker_result_dir} -w `pwd` --gpus all  --cpus 8 -it --rm --shm-size=16g peterpirogtf/ray_tf2:latest'
rllib_cmd='python3 rllib_test.py'

cmd=docker_cmd + ' ' + rllib_cmd


os.system(cmd)

